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
        eval_interval=50000,
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
        img_size=128,
        patch_size=8,
        embed_dim=768,
        depth=16,
        num_heads=12,
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
        path='',  # if not none, generated images will be saved into this folder
        save_npz=''  # leave it none in training; set the npz file path during eval
    )

    return config
