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
        tokens=1024,  # number of tokens to the network
        low_freqs=8,  # <=16
        embed_dim=768,
        depth=16,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=False,
        num_classes=-1,
    )

    config.dataset = d(
        name='ffhq256',
        path='/data/clusterfs/mld/users/lanliu/mang/datasets/ffhq256',  # /home/mang/Downloads/ffhq256_jpg
        fid_stat='assets/fid_stats/fid_stats_ffhq256_jpg.npz',
        resolution=256,
        tokens=1024,  # number of tokens to the network
        low_freqs=8,  # could be 10, 13, 15, 16 (<=16)
        block_sz=4,  # size of DCT block
        low2high_order=[0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15],
        reverse_order=[0, 1, 5, 6, 2, 4, 7, 12, 3, 8, 11, 13, 9, 10, 14, 15],
    )

    config.sample = d(
        sample_steps=50,
        n_samples=10000,
        mini_batch_size=100,
        algorithm='dpm_solver',
        path=None
    )

    return config
