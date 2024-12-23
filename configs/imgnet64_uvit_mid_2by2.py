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
        low_freqs=4,  # B*B - m
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
        path='/data/scratch/datasets/imagenet64/train',
        resolution=64,
        tokens=256,  # number of tokens to the network
        low_freqs=4,  # B*B - m
        block_sz=2,  # B
        low2high_order=[0, 1, 2, 3],
        reverse_order=[0, 1, 2, 3],
        Y_bound=[247.125],  # eta
        Y_std=[6.522, 3.377, 3.386, 2.389],  # Entropy-Based Frequency Reweighting (EBFR)
        Cb_std=[4.27, 1.329, 1.351, 0.988],
        Cr_std=[4.078, 1.292, 1.303, 0.987],
        SNR_scale=4.0,
    )

    config.sample = d(
        sample_steps=50,
        n_samples=10000,
        mini_batch_size=250,
        algorithm='dpm_solver',
        path=None
    )

    return config
