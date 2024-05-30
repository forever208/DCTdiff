import sde
import ml_collections
import torch
from torch import multiprocessing as mp
from datasets import get_dataset
import utils
import einops
from torch.utils._pytree import tree_map
import accelerate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
import tempfile
from tools.fid_score import calculate_fid_given_paths
from absl import logging
import builtins
import os
from datetime import timedelta
from accelerate import InitProcessGroupKwargs
import numpy as np


def train(config):
    mp.set_start_method('spawn')
    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))  # 1 hour
    accelerator = accelerate.Accelerator(kwargs_handlers=[process_group_kwargs])
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    # torch.cuda.manual_seed(config.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)

    assert config.train.batch_size % accelerator.num_processes == 0
    mini_batch_size = config.train.batch_size // accelerator.num_processes  # batch per GPU
    logging.info(f'use {accelerator.num_processes} GPUs with batch size {mini_batch_size}/GPU')

    # log setting
    if accelerator.is_main_process:
        os.makedirs(config.ckpt_root, exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        utils.set_logger(log_level='info', fname=os.path.join(config.workdir, 'output.log'))
        logging.info(config)
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None

    # Dataset and DataLoader
    dataset = get_dataset(**config.dataset)
    assert os.path.exists(dataset.fid_stat)
    train_dataset = dataset.get_split(split='train', labeled=config.train.mode == 'cond')
    train_dataset_loader = DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True, drop_last=True,
                                      num_workers=16, pin_memory=False, persistent_workers=True)
    logging.info(f'dataset samples: {len(train_dataset)}')

    # keep track of training states (lr, opt, model)
    train_state = utils.initialize_train_state(config, device)

    # wrap data_loader and model with accelerator for distributed training
    nnet, nnet_ema, optimizer, train_dataset_loader = accelerator.prepare(
        train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_dataset_loader
    )
    lr_scheduler = train_state.lr_scheduler
    train_state.resume(config.ckpt_root)

    # variables for loss reweighting
    Y_std = np.array(config.dataset.Y_std)
    logging.info(f'using {Y_std} for Y_loss reweighting')
    Y_reweight = Y_std[list(config.dataset.low2high_order)][:config.dataset.low_freqs]
    Y_reweight = Y_reweight / (Y_reweight.sum() / Y_reweight.shape[0])  # normalization
    Y_reweight = torch.from_numpy(Y_reweight).to(device=device).float()

    Cb_std = np.array(config.dataset.Cb_std)
    logging.info(f'using {Cb_std} for Cb_loss reweighting')
    Cb_reweight = Cb_std[list(config.dataset.low2high_order)][:config.dataset.low_freqs]
    Cb_reweight = Cb_reweight / (Cb_reweight.sum() / Cb_reweight.shape[0])  # normalization
    Cb_reweight = torch.from_numpy(Cb_reweight).to(device=device).float()

    Cr_std = np.array(config.dataset.Cr_std)
    logging.info(f'using {Cr_std} for Cr_loss reweighting')
    Cr_reweight = Cr_std[list(config.dataset.low2high_order)][:config.dataset.low_freqs]
    Cr_reweight = Cr_reweight / (Cr_reweight.sum() / Cr_reweight.shape[0])  # normalization
    Cr_reweight = torch.from_numpy(Cr_reweight).to(device=device).float()

    reweight_by_std = torch.cat((Y_reweight, Y_reweight, Y_reweight, Y_reweight, Cb_reweight, Cr_reweight)).to(device=device)
    assert reweight_by_std.shape[0] == config.dataset.low_freqs * 6
    # cb_index = [i for i in range(4, config.dataset.tokens, 6)]
    # cr_index = [i for i in range(5, config.dataset.tokens, 6)]
    # y_index = [i for i in range(0, config.dataset.tokens) if i not in cb_index and i not in cr_index]

    def get_data_generator():
        while True:
            for data in tqdm(train_dataset_loader, disable=not accelerator.is_main_process, desc='epoch'):
                yield data

    data_generator = get_data_generator()

    # wrap network with diffusion framework
    score_model = sde.ScoreModel(nnet, pred=config.pred, sde=sde.VPSDE())
    score_model_ema = sde.ScoreModel(nnet_ema, pred=config.pred, sde=sde.VPSDE())

    def train_step(_batch):
        _metrics = dict()
        optimizer.zero_grad()

        if config.train.mode == 'uncond':
            loss = sde.LSimple(score_model, _batch, pred=config.pred, reweight=reweight_by_std)
        elif config.train.mode == 'cond':
            loss = sde.LSimple(score_model, _batch[0], pred=config.pred, y=_batch[1], reweight=reweight_by_std)
        else:
            raise NotImplementedError(config.train.mode)

        _metrics['loss'] = accelerator.gather(loss.detach()).mean()
        accelerator.backward(loss.mean())

        if 'grad_clip' in config and config.grad_clip > 0:
            accelerator.clip_grad_norm_(nnet.parameters(), max_norm=config.grad_clip)

        optimizer.step()
        lr_scheduler.step()
        train_state.ema_update(config.get('ema_rate', 0.9999))
        train_state.step += 1
        return dict(lr=train_state.optimizer.param_groups[0]['lr'], **_metrics)


    def eval_step(n_samples, sample_steps, algorithm, Y_bound):
        logging.info(f'eval_step: n_samples={n_samples}, sample_steps={sample_steps}, algorithm={algorithm}, '
                     f'mini_batch_size={config.sample.mini_batch_size}')

        def sample_fn(_n_samples):
            _x_init = torch.randn(_n_samples, *dataset.data_shape, device=device)
            if config.train.mode == 'uncond':
                kwargs = dict()
            elif config.train.mode == 'cond':
                kwargs = dict(y=dataset.sample_label(_n_samples, device=device))
            else:
                raise NotImplementedError

            if algorithm == 'euler_maruyama_sde':
                return sde.euler_maruyama(sde.ReverseSDE(score_model_ema), _x_init, sample_steps, **kwargs)
            elif algorithm == 'euler_maruyama_ode':
                return sde.euler_maruyama(sde.ODE(score_model_ema), _x_init, sample_steps, **kwargs)
            elif algorithm == 'dpm_solver':
                noise_schedule = NoiseScheduleVP(schedule='linear')
                model_fn = model_wrapper(
                    score_model_ema.noise_pred,
                    noise_schedule,
                    time_input_type='0',
                    model_kwargs=kwargs
                )
                dpm_solver = DPM_Solver(model_fn, noise_schedule)
                return dpm_solver.sample(
                    _x_init,
                    steps=sample_steps,
                    eps=1e-4,
                    adaptive_step_size=False,
                    fast_version=True,
                )
            else:
                raise NotImplementedError

        with tempfile.TemporaryDirectory() as temp_path:  # files will be deleted after 'with' context
            path = config.sample.path or temp_path
            if accelerator.is_main_process:
                os.makedirs(path, exist_ok=True)

            # generate samples
            utils.DCTsample2dir(
                accelerator, path, n_samples, config.sample.mini_batch_size, sample_fn,
                tokens=config.dataset.tokens, low_freqs=config.dataset.low_freqs,
                reverse_order=config.dataset.reverse_order, resolution=config.dataset.resolution,
                block_sz=config.dataset.block_sz, Y_bound=Y_bound
            )

            # FID computation
            _fid = 0
            if accelerator.is_main_process:
                _fid = calculate_fid_given_paths((dataset.fid_stat, path))
                logging.info(f'step={train_state.step} fid{n_samples}={_fid}')
                with open(os.path.join(config.workdir, 'eval.log'), 'a') as f:
                    print(f'step={train_state.step} fid{n_samples}={_fid}', file=f)

            _fid = torch.tensor(_fid, device=device)
            _fid = accelerator.reduce(_fid, reduction='sum')

        return _fid.item()

    logging.info(f'Start fitting, step={train_state.step}, mixed_precision={config.mixed_precision}')
    step_fid = []
    while train_state.step < config.train.n_steps:
        nnet.train()
        batch = tree_map(lambda x: x.to(device), next(data_generator))
        metrics = train_step(batch)

        nnet.eval()
        if accelerator.is_main_process and train_state.step % config.train.log_interval == 0:
            logging.info(utils.dct2str(dict(step=train_state.step, **metrics)))
            logging.info(config.workdir)
        accelerator.wait_for_everyone()

        if accelerator.is_main_process and train_state.step % config.train.eval_interval == 0:
            grid_img_path = os.path.join(config.sample_dir, f'{train_state.step}.png')
            logging.info(f'Save a grid of 16 samples into {grid_img_path}...')
            x_init = torch.randn(16, *dataset.data_shape, device=device)

            if config.train.mode == 'uncond':
                kwargs = dict()
            elif config.train.mode == 'cond':
                kwargs = dict(y=dataset.sample_label(16, device=device))
            else:
                raise NotImplementedError

            if config.sample.algorithm == 'euler_maruyama_sde':
                samples = sde.euler_maruyama(sde.ReverseSDE(score_model_ema), x_init, config.sample.sample_steps,
                                             **kwargs)
            elif config.sample.algorithm == 'euler_maruyama_ode':
                samples = sde.euler_maruyama(sde.ODE(score_model_ema), x_init, config.sample.sample_steps, **kwargs)
            elif config.sample.algorithm == 'dpm_solver':
                noise_schedule = NoiseScheduleVP(schedule='linear')
                model_fn = model_wrapper(
                    score_model_ema.noise_pred,
                    noise_schedule,
                    time_input_type='0',
                    model_kwargs=kwargs
                )
                dpm_solver = DPM_Solver(model_fn, noise_schedule)
                samples = dpm_solver.sample(
                    x_init,
                    steps=config.sample.sample_steps,
                    eps=1e-4,
                    adaptive_step_size=False,
                    fast_version=True,
                )
            else:
                raise NotImplementedError

            utils.DCTsamples_to_grid_image(
                samples, tokens=config.dataset.tokens, low_freqs=config.dataset.low_freqs,
                block_sz=config.dataset.block_sz, reverse_order=config.dataset.reverse_order,
                resolution=config.dataset.resolution, grid_sz=4, path=grid_img_path, Y_bound=config.dataset.Y_bound
            )
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

        if train_state.step >= 50000 and train_state.step % config.train.save_interval == 0:
            logging.info(f'Save and eval checkpoint {train_state.step}...')
            if accelerator.local_process_index == 0:
                train_state.save(os.path.join(config.ckpt_root, f'{train_state.step}.ckpt'))
            accelerator.wait_for_everyone()

            # calculate fid of the saved checkpoint
            fid = eval_step(n_samples=config.sample.n_samples, sample_steps=config.sample.sample_steps,
                            algorithm=config.sample.algorithm, Y_bound=config.dataset.Y_bound)
            step_fid.append((train_state.step, fid))
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

    logging.info(f'Finish fitting, step={train_state.step}')
    logging.info(f'step_fid: {step_fid}')
    step_best = sorted(step_fid, key=lambda x: x[1])[0][0]
    logging.info(f'step_best: {step_best}')
    train_state.load(os.path.join(config.ckpt_root, f'{step_best}.ckpt'))
    del metrics
    accelerator.wait_for_everyone()
    eval_step(n_samples=config.sample.n_samples, sample_steps=config.sample.sample_steps, algorithm=config.sample.algorithm)
    logging.info(f'all done!')


from absl import flags
from absl import app
from ml_collections import config_flags


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("workdir", None, "Work unit directory.")


def main(argv):
    config = FLAGS.config
    config.workdir = FLAGS.workdir or 'exp_train'
    config.ckpt_root = os.path.join(config.workdir, 'ckpts')
    config.sample_dir = os.path.join(config.workdir, 'samples')
    train(config)


if __name__ == "__main__":
    app.run(main)
