import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.pred = 'noise_pred'
    config.z_shape = (4, 64, 64)

    config.autoencoder = d(
        pretrained_path='/data/scratch/U-ViT2/assets/stable-diffusion/autoencoder_kl.pth'
    )

    config.train = d(
        n_steps=800000,
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
        img_size=64,
        patch_size=4,
        in_chans=4,
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
    )

    config.sample = d(
        sample_steps=100,
        n_samples=10000,
        mini_batch_size=25,  # the decoder is large
        algorithm='dpm_solver',
        path='',  # if not none, generated images will be saved into this folder
        save_npz='/data/scratch/afhq512_base_dpm_NFE100_50k.npz'  # leave it none in training; set the npz file path during eval
    )

    return config
