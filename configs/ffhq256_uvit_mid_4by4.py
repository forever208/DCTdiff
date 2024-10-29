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
        path='/data/scratch/datasets/ffhq256',  # /home/mang/Downloads/ffhq256_jpg
        resolution=256,
        tokens=1024,  # number of tokens to the network
        low_freqs=8,  # could be 10, 13, 15, 16 (<=16)
        block_sz=4,  # size of DCT block
        low2high_order=[0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15],
        reverse_order=[0, 1, 5, 6, 2, 4, 7, 12, 3, 8, 11, 13, 9, 10, 14, 15],
        Y_bound=[485.0],
        Y_std=[6.543, 2.894, 1.769, 1.197, 2.8, 1.916, 1.458, 1.029, 1.647, 1.386, 1.106, 0.994, 1.107, 0.996, 0.994, 0.996],
        Cb_std=[4.173, 1.392, 0.992, 0.989, 1.338, 0.991, 0.99, 0.991, 0.992, 0.992, 0.989, 0.991, 0.99, 0.991, 0.991, 0.991],
        Cr_std=[4.417, 1.52, 0.993, 0.99, 1.473, 0.991, 0.99, 0.991, 0.993, 0.991, 0.989, 0.99, 0.997, 0.991, 0.989, 0.991],
        SNR_scale=1.0,
    )

    config.sample = d(
        sample_steps=50,
        n_samples=10000,
        mini_batch_size=100,
        algorithm='dpm_solver',
        path=None
    )

    return config
