import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.pred = 'noise_pred'

    config.train = d(
        n_steps=500000,
        batch_size=256,
        mode='uncond',
        log_interval=100,
        eval_interval=25000,
        save_interval=25000,
    )

    config.optimizer = d(
        name='adamw',
        lr=0.0002,
        weight_decay=0.03,
        betas=(0.99, 0.99),
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=5000
    )

    config.nnet = d(
        name='uvit',
        tokens=256,  # number of tokens to the network
        low_freqs=4,  # B**2 - m
        embed_dim=512,
        depth=12,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=False,
        num_classes=-1,
    )

    config.dataset = d(
        name='celeba',
        path='/data/scratch/datasets/celeba64',
        resolution=64,
        tokens=256,  # number of tokens to the network
        low_freqs=4,  # B**2 - m
        block_sz=2,  # size of DCT block
        Y_bound=[244.925],  # eta
        Y_std=[6.507, 3.297, 3.063, 1.844],
        Cb_std=[4.013, 1.287, 1.06, 1.0],
        Cr_std=[4.239, 1.422, 1.204, 1.0],
        SNR_scale=4.0,  # set 4 when sing euler_maruyama_ode Sample, set 1 when using DPM-Solver
    )

    config.sample = d(
        sample_steps=100,
        n_samples=10000,
        mini_batch_size=500,
        algorithm='euler_maruyama_ode',
        path='/data/scratch/samples',  # must be specified for distributed image saving
        save_npz=''  # save generated sample if not None (used for precision/recall computation)
    )

    return config
