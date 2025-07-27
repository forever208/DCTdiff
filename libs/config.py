from dataclasses import dataclass
from typing import Tuple


@dataclass
class LDMConfig:
    ######################################
    ### VARIATIONAL AUTOENCODER CONFIG ###
    ######################################

    ### Input/Latent/Output Channels ###
    img_size: int = 256
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 4

    ### Encoder/Decoder Config ###
    residual_layers_per_block: int = 2
    attention_layers: int = 1
    attention_residual_connections: bool = True
    post_encoder_latent_proj: bool = True
    pre_decoder_latent_proj: bool = True

    ### Block Config ###
    vae_channels_per_block: Tuple = (128, 256, 512, 512)
    vae_up_down_factor: int = 2
    vae_up_down_kernel_size: int = 3

    ### Quantization Config ###
    quantize: bool = False
    codebook_size: int = 16384
    vq_embed_dim: float = 4
    commitment_beta: float = 0.25

    ### VAE Scaling (Variance Normalization for Diffusion) ###
    vae_scale_factor: float = None

    ###################
    ### UNET CONFIG ###
    ###################

    ### UNET Parts Config ###
    down_block_types: Tuple = ("AttnDown", "AttnDown", "AttnDown", "Down")  # AttnDown/Down
    mid_block_types: str = "AttnMid"  # AttnMid/Mid
    up_block_types: Tuple = ("Up", "AttnUp", "AttnUp", "AttnUp")  # AttnUp/Up
    unet_channels_per_block: Tuple = (320, 640, 1280, 1280)
    unet_residual_layers_per_block: int = 2
    unet_up_down_factor: int = 2
    unet_up_down_kernel_size: int = 3

    ### Time Embeddings Config ###
    time_embed_start_dim: int = 320
    time_embed_proj_dim: int = 1280
    time_embed_requires_grad: bool = False

    ### Attention Config ###
    transformer_blocks_per_layer: int = 1
    transformer_dim_mult: int = 4
    attention_bias: bool = False
    attention_head_dim: int = 8

    ### Text Embeddings Config ###
    text_conditioning_hf_model: bool = "openai/clip-vit-large-patch14"
    text_conditioning: bool = False
    text_embed_dim: int = 768

    ### Class Embeddings Config ###
    class_conditioning: bool = False
    class_embed_dim: int = 512
    num_classes: int = 1000

    ################################
    ### DIFFUSION SAMPLER CONFIG ###
    ################################

    ### Diffusion Config ###
    num_diffusion_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    cfg_dropout_prob: float = 0.1
    diffusion_loss_fn: str = "mse"

    ######################
    ### GENERAL CONFIG ###
    ######################
    pre_encoded_text: bool = False
    groupnorm_groups: int = 32
    norm_eps: float = 1e-6
    dropout: float = 0.0

