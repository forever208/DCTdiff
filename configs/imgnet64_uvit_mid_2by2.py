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
        batch_size=1024,
        mode='cond',
        log_interval=100,
        eval_interval=25000,
        save_interval=25000,
    )

    config.optimizer = d(
        name='adamw',
        lr=0.0003,
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
        embed_dim=768,
        depth=16,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=False,
        num_classes=1000,
        use_checkpoint=False
    )

    config.dataset = d(
        name='imgnet64',
        path='/data/clusterfs/mld/users/lanliu/mang/datasets/imagenet64/train',  # /home/mang/Downloads/train
        resolution=64,
        tokens=256,  # number of tokens to the network
        low_freqs=4,  # could be 10, 13, 15, 16 (<=16)
        block_sz=2,  # size of DCT block
        low2high_order=[0, 1, 2, 3],
        reverse_order= [0, 1, 2, 3],
        Y_bound=[248.875, 248.875, 248.875, 248.875],
        Y_std=[6.523, 3.402, 3.407, 2.42],
        Cb_std = [4.286, 1.343, 1.365, 0.988],
        Cr_std=[4.096, 1.309, 1.319, 0.987],
    )

    config.sample = d(
        sample_steps=50,
        n_samples=50000,
        mini_batch_size=250,
        algorithm='dpm_solver',
        path=None
    )

    return config
