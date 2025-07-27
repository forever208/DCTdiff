import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.pred = 'noise_pred'
    config.z_shape = (64, 8, 8)

    config.autoencoder = d(
        pretrained_path='/data/clusterfs/mld/users/lanliu/mang/LDM_exps/celeba256_SDVAE_bf16_b48_f32d64_flip/SDVAE/checkpoint_530000/model.safetensors',
        scaler=0.162145, # stdx6
        ldm_config_path='/data/scratch/U-ViT2/configs/ldm_f32d64.yaml',
    )

    config.train = d(
        n_steps=300000,
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
        img_size=8,
        patch_size=1,
        in_chans=64,
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
        path='/data/scratch/datasets/celeba256_latents/celeba256_SDVAE_f32_latents_530k',
        resolution=256,
    )

    config.sample = d(
        sample_steps=50,
        n_samples=10000,
        mini_batch_size=25,  # the decoder is large
        algorithm='dpm_solver',
        path='/data/scratch/samples1',  # generated images will be saved into this folder for FID eval
        save_npz=''  # save generated sample if not None (used for precision/recall computation)
    )

    return config
