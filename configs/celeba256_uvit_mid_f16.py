import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.pred = 'noise_pred'
    config.z_shape = (16, 16, 16)

    config.autoencoder = d(
        pretrained_path='/home/mning/LDM_exps/celeba256_SDVAE_bf16_b48_f16d16_flip/SDVAE/checkpoint_610000/model.safetensors',
        scaler=0.18475,  # 99.99pct=0.21697, 99.999pct=0.18475, 99.9999pct=0.15928
        ldm_config_path='configs/ldm_f16d16.yaml',
    )

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
        img_size=16,
        patch_size=1,
        in_chans=16,
        embed_dim=768,
        depth=16,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=False,
        num_classes=-1,
    )

    config.dataset = d(
        name='celeba256_features',
        path='/projects/prjs0865/datasets/celeba256_SDVAE_f16_latents',
        resolution=256,
    )

    config.sample = d(
        sample_steps=50,
        n_samples=50000,
        mini_batch_size=25,  # the decoder is large
        algorithm='dpm_solver',
        path='/projects/prjs0865/samples',  # generated images will be saved into this folder for FID eval
        save_npz=''  # save generated sample if not None (used for precision/recall computation)
    )

    return config
