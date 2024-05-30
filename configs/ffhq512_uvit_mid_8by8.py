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
        batch_size=128,
        mode='uncond',
        log_interval=100,
        eval_interval=25000,
        save_interval=25000,
    )

    config.optimizer = d(
        name='adamw',
        lr=0.0001,
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
        low_freqs=18,  # B**2 - m
        embed_dim=768,
        depth=16,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=False,
        num_classes=-1,
    )

    config.dataset = d(
        name='ffhq512',
        path='/data/scratch/datasets/ffhq512',
        resolution=512,
        tokens=1024,  # number of tokens to the network
        low_freqs=18,  # B**2 - m
        block_sz=8,  # size of DCT block
        Y_bound=[969.5],  # eta
        Y_std=[6.53, 2.866, 1.871, 1.368, 1.186, 1.252, 1.0, 0.996, 2.766, 1.865, 1.569, 1.366, 1.07, 1.048, 0.996, 0.993, 1.742, 1.506, 1.265, 1.172, 1.311, 0.994, 0.996, 0.994, 1.334, 1.108, 1.185, 0.993, 1.094, 0.993, 0.996, 0.993, 0.995, 1.086, 0.995, 1.026, 0.995, 0.993, 0.995, 0.993, 0.996, 0.993, 0.996, 0.993, 0.996, 0.993, 0.996, 0.993, 0.994, 0.994, 0.996, 0.994, 0.995, 0.994, 0.996, 0.994, 0.996, 0.993, 0.996, 0.993, 0.996, 0.993, 0.996, 0.993],
        Cb_std=[4.167, 1.364, 1.001, 0.99, 0.999, 0.989, 0.999, 0.99, 1.303, 1.025, 0.993, 0.992, 0.992, 0.992, 0.992, 0.992, 0.993, 0.996, 0.994, 0.998, 0.993, 0.998, 0.994, 0.995, 0.991, 0.992, 0.992, 0.992, 0.992, 0.992, 0.992, 0.992, 0.994, 0.997, 0.993, 1.0, 0.991, 1.0, 0.993, 0.997, 0.991, 0.992, 0.992, 0.992, 0.992, 0.992, 0.992, 0.992, 0.992, 0.998, 0.994, 1.0, 0.993, 1.0, 0.994, 0.998, 0.992, 0.992, 0.992, 0.992, 0.992, 0.992, 0.992, 0.992],
        Cr_std=[4.403, 1.482, 1.07, 0.991, 0.998, 0.99, 0.998, 0.991, 1.445, 1.105, 0.993, 0.992, 0.992, 0.992, 0.993, 0.992, 1.041, 0.995, 0.993, 0.997, 0.993, 0.998, 0.993, 0.994, 0.995, 0.992, 0.993, 0.992, 0.992, 0.992, 0.992, 0.992, 0.995, 0.997, 0.993, 0.999, 0.993, 1.0, 0.993, 0.997, 0.996, 0.992, 0.993, 0.992, 0.992, 0.992, 0.992, 0.992, 0.992, 0.998, 0.994, 0.999, 0.994, 1.0, 0.994, 0.997, 0.994, 0.992, 0.992, 0.992, 0.992, 0.992, 0.992, 0.992],
        SNR_scale=12.0,
    )

    config.sample = d(
        sample_steps=100,
        n_samples=10000,
        mini_batch_size=100,
        algorithm='dpm_solver',
        path='/data/scratch/samples',  # must be specified for distributed image saving
        save_npz=''  # save generated sample if not None (used for precision/recall computation)
    )

    return config
