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
        low_freqs=9,  # <=16
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
        path='/data/scratch/datasets/ffhq128',  # /home/mning2/datasets/ffhq128
        resolution=128,
        tokens=256,  # number of tokens to the network
        low_freqs=9,  # <=16
        block_sz=4,  # size of DCT block
        low2high_order=[0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15],
        reverse_order= [0, 1, 5, 6, 2, 4, 7, 12,3, 8,  11, 13, 9, 10, 14, 15],
        Y_bound=[480.25],
        Y_std=[6.53, 3.478, 2.196, 1.405, 3.378, 2.381, 1.765, 1.185, 2.072, 1.697, 1.291, 0.999, 1.291, 1.083, 0.999, 0.999],
        Cb_std=[4.138, 1.741, 0.999, 0.998, 1.589, 0.998, 0.998, 0.998, 0.998, 0.998, 0.997, 0.998, 0.997, 0.998, 0.998, 0.998],
        Cr_std=[4.391, 1.904, 0.998, 0.998, 1.758, 1.082, 0.998, 0.998, 0.999, 0.998, 0.998, 0.998, 1.0, 0.998, 0.998, 0.998],
        SNR_scale=3.0,
    )

    config.sample = d(
        sample_steps=50,
        n_samples=10000,
        mini_batch_size=500,
        algorithm='dpm_solver',
        path=None
    )

    return config
