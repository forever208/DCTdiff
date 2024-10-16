import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.pred = 'noise_pred'

    config.train = d(
        n_steps=1000000,
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
        img_size=128,
        patch_size=8,
        embed_dim=512,
        depth=12,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=False,
        num_classes=-1,
    )

    config.dataset = d(
        name='ffhq128',
        path='/data/scratch/datasets/ffhq128',  # /data/clusterfs/mld/users/lanliu/mang/datasets/ffhq128
        resolution=128,
    )

    config.sample = d(
        sample_steps=50,
        n_samples=10000,
        mini_batch_size=500,
        algorithm='dpm_solver',
        path=''
    )

    return config
