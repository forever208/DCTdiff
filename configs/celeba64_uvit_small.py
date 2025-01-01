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
        save_interval=50000,
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
        img_size=64,
        patch_size=4,
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
        path='/data/scratch/datasets/celeba64',  # /gpfs/work4/0/prjs0865/DCT/celeba
        resolution=64,
    )

    config.sample = d(
        sample_steps=100,
        n_samples=10000,
        mini_batch_size=500,
        algorithm='euler_maruyama_ode',
        path='',  # if not none, generated images will be saved into this folder
        save_npz=''  # leave it none in training; set the npz file path during eval
    )

    return config
