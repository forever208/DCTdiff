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
        tokens=64,  # number of tokens to the network
        low_freqs=4,  # 15, 21, 28, 36, 43
        embed_dim=768,
        depth=16,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=False,
        num_classes=-1,
    )

    config.dataset = d(
        name='cifar10',
        path='/data/scratch/datasets/cifar10',  # /home/mning2/datasets/cifar10  /home/mang/Downloads/cifar10
        resolution=32,
        tokens=64,  # number of tokens to the network
        low_freqs=4,  # could be 10, 13, 15, 16 (<=16)
        block_sz=2,  # size of DCT block
        low2high_order=[0, 1, 2, 3],
        reverse_order= [0, 1, 2, 3],
        Y_bound=[242.382, 242.382, 242.382, 242.382],
        Y_std=[6.471, 3.588, 3.767, 2.411],
        Cb_std=[4.308, 1.315, 1.487, 1.0],
        Cr_std=[4.014, 1.284, 1.435, 1.0],
        SNR_scale=2.0,
    )

    config.sample = d(
        sample_steps=50,
        n_samples=50000,
        mini_batch_size=500,
        algorithm='dpm_solver',
        path=None
    )

    return config
