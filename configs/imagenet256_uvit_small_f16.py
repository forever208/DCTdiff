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
        pretrained_path='/leonardo_work/EUHPC_B29_014/LDM_exps/imagenet256_SDVAE_bf16_b128_f16_flip_400k/SDVAE/checkpoint_280000/model.safetensors',
        scaler=0.9296,
        ldm_config_path='/leonardo_work/EUHPC_B29_014/U-ViT2/configs/ldm_f16d16.yaml',
    )

    config.train = d(
        n_steps=300000,
        batch_size=1024,
        mode='cond',
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
        embed_dim=512,
        depth=12,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=False,
        num_classes=1001,
        use_checkpoint=False,
    )

    config.dataset = d(
        name='imagenet256_features',
        path='/leonardo_work/EUHPC_B29_014/datasets/imagenet256_latents/imagenet256_SDVAE_f16_280k',
        cfg=True,
        p_uncond=0.15
    )

    config.sample = d(
        sample_steps=100,
        n_samples=10000,
        mini_batch_size=50,  # the decoder is large
        algorithm='euler_maruyama_ode',
        path='/leonardo_work/EUHPC_B29_014/samples10',  # generated images will be saved into this folder for FID eval
        save_npz='',  # save generated sample if not None (used for precision/recall computation)
        cfg=True,
        scale=0.4,
    )

    return config
