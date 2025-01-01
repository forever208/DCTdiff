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
        lr=0.0003,
        weight_decay=0.03,
        betas=(0.99, 0.999),
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=50000
    )

    config.nnet = d(
        name='uvit',
        img_size=32,
        patch_size=2,
        embed_dim=512,
        depth=12,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=False,
        num_classes=-1,
    )

    config.dataset = d(
        name='cifar10',
        path='/data/scratch/datasets/cifar10',  # /home/mang/Downloads/cifar10
    )

    config.sample = d(
        sample_steps=100,
        n_samples=50000,
        mini_batch_size=500,
        algorithm='euler_maruyama_ode',
        path='',  # if not none, generated images will be saved into this folder
        save_npz=''  # leave it none in training; set the npz file path during eval
    )

    return config
