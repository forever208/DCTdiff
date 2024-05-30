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
        low_freqs=4,  # 15, 21, 28, 36, 43
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
        path='/gpfs/work4/0/prjs0865/DCT/celeba',  # /home/mang/Downloads/celeba, /data/clusterfs/mld/users/lanliu/mang/datasets/celeba64, ../datasets/celeba
        fid_stat='assets/fid_stats/fid_stats_celeba64_all.npz',
        resolution=64,
        tokens=256,  # number of tokens to the network
        low_freqs=4,  # could be 10, 13, 15, 16 (<=16)
        block_sz=2,  # size of DCT block
        low2high_order=[0, 1, 2, 3],
        reverse_order= [0, 1, 2, 3],
    )

    config.sample = d(
        sample_steps=50,
        n_samples=10000,
        mini_batch_size=500,
        algorithm='dpm_solver',
        path=None
    )

    return config