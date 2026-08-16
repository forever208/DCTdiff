import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.pred = 'noise_pred'
    config.z_shape = (4, 32, 32)

    config.autoencoder = d(
        pretrained_path='/home/ningmang/vae/exps/celeba256_SDVAE/checkpoints/98000.ckpt',
        scaler=0.07029,
    )

    config.train = d(
        n_steps=200000,
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
        img_size=32,
        patch_size=2,
        in_chans=4,
        embed_dim=512,
        depth=12,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=False,
        num_classes=-1,
    )

    config.dataset = d(
        name='celeba256_features',
        path='/mnt/vepfs/base2/stream/ningmang/datasets/celeba256_latents/SDVAE_98k',
        resolution=256,
    )

    config.sample = d(
        sample_steps=100,
        n_samples=50000,
        mini_batch_size=50,  # the decoder is large
        algorithm='euler_maruyama_ode',
        path='/home/ningmang/DCTdiff/samples1',  # generated images will be saved into this folder for FID eval
        save_npz=''  # save generated sample if not None (used for precision/recall computation)
    )

    return config
