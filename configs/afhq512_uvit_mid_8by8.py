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
        name='afhq512',
        path='/data/scratch/datasets/afhq512_jpg',
        resolution=512,
        tokens=1024,  # number of tokens to the network
        low_freqs=18,  # B**2 - m
        block_sz=8,  # size of DCT block
        Y_bound=[928.0],  # eta
        Y_std=[6.524, 3.25, 2.206, 1.607, 1.368, 1.495, 1.173, 0.999, 3.218, 2.332, 1.931, 1.682, 1.288, 1.277, 1.088, 0.997, 2.266, 1.974, 1.649, 1.449, 1.621, 1.257, 0.999, 0.998, 1.758, 1.493, 1.533, 1.225, 1.42, 0.997, 0.999, 0.997, 1.326, 1.446, 1.684, 1.4, 1.19, 0.998, 0.999, 0.998, 1.25, 1.068, 1.331, 1.198, 0.999, 0.997, 0.999, 0.997, 1.255, 1.12, 0.999, 0.998, 0.999, 0.998, 0.999, 0.998, 0.999, 0.997, 0.999, 0.997, 0.999, 0.997, 0.999, 0.997],
        Cb_std=[4.153, 1.415, 1.0, 0.99, 1.0, 0.989, 0.999, 0.991, 1.372, 1.091, 0.995, 0.994, 0.994, 0.994, 0.995, 0.994, 0.996, 0.997, 0.996, 0.999, 0.995, 0.999, 0.995, 0.996, 0.995, 0.994, 0.995, 0.994, 0.994, 0.994, 0.995, 0.994, 0.997, 0.998, 0.995, 1.0, 0.993, 1.0, 0.995, 0.998, 0.995, 0.994, 0.995, 0.994, 0.994, 0.994, 0.994, 0.994, 0.995, 0.999, 0.996, 1.0, 0.995, 1.0, 0.996, 0.999, 0.995, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994],
        Cr_std=[3.906, 1.291, 0.999, 0.989, 0.998, 0.988, 0.998, 0.991, 1.246, 0.994, 0.995, 0.994, 0.994, 0.994, 0.994, 0.994, 0.996, 0.997, 0.995, 0.999, 0.995, 0.999, 0.995, 0.997, 0.996, 0.994, 0.995, 0.994, 0.994, 0.994, 0.994, 0.994, 0.996, 0.999, 0.995, 1.0, 0.995, 1.0, 0.994, 0.999, 0.996, 0.994, 0.995, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.999, 0.996, 1.0, 0.996, 1.0, 0.996, 0.999, 0.995, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994, 0.994],
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
