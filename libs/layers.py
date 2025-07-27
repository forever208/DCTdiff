import torch
import torch.nn as nn
import torch.nn.functional as F


class UpSampleBlock2D(nn.Module):
    """
    Upsampling Block that takes

    (B x C x H x W) -> (B x C x H*2 x W*2)

    Args:
        - in_channels: Input channels of images (no change in channels)
        - kernel_size: Kernel size in learnable convolution
        - upsample_factor: By what factor do you want to upsample image by?
    """

    def __init__(self,
                 in_channels,
                 kernel_size=3,
                 upsample_factor=2):
        super(UpSampleBlock2D, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.factor = upsample_factor

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=upsample_factor, mode="nearest"),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding='same')
        )

    def forward(self, x):
        batch, channels, height, width = x.shape

        upsampled = self.upsample(x)

        assert upsampled.shape[2:] == (height * self.factor, width * self.factor)

        return upsampled


class DownSampleBlock2D(nn.Module):
    """
    Downsampling Block that takes

    (B x C x H x W) -> (B x C x H/2 x W/2)

    Args:
        - in_channels: Input channels of images (no change in channels)
        - kernel_size: Kernel size in learnable convolution
        - kernel_size: What stride do you want to use to downsample?

    """

    def __init__(self,
                 in_channels,
                 kernel_size=3,
                 downsample_factor=2):
        super(DownSampleBlock2D, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.factor = downsample_factor

        self.downsample_conv = nn.Conv2d(in_channels=in_channels,
                                         out_channels=in_channels,
                                         kernel_size=kernel_size,
                                         stride=downsample_factor,
                                         padding=1)

    def forward(self, x):
        batch, channels, height, width = x.shape

        downsampled = self.downsample_conv(x)

        assert downsampled.shape[2:] == (height / self.factor, width / self.factor)

        return downsampled


class ResidualBlock2D(nn.Module):
    """
    Core Residual Block of a Stack of Convolutions and a Residual Connection

    Args:
        - in_channels: Input channels to the block
        - out_channels: Output channels from the block
        - dropout_p: Dropout probability you want to use
        - groupnorm_groups: How many groups do you want in Groupnorm
        - time_embed_proj: Do you want to inject time embeddings?
        - time_embed_dim: Time embedding dimension
        - norm_eps: Groupnorm eps
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout_p=0.0,
                 groupnorm_groups=32,
                 time_embed_proj=False,
                 time_embed_dim=1280,
                 class_embed_proj=False,
                 class_embed_dim=512,
                 norm_eps=1e-6):

        super(ResidualBlock2D, self).__init__()

        ### Input Convolutions ###
        self.norm1 = nn.GroupNorm(num_groups=groupnorm_groups, num_channels=in_channels, eps=norm_eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding="same")

        ### Second Set of Convolutions ###
        self.norm2 = nn.GroupNorm(num_groups=groupnorm_groups, num_channels=out_channels, eps=norm_eps, affine=True)
        self.dropout = nn.Dropout(dropout_p)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding="same")

        ### Time Embedding Mapping ###
        self.time_expand = None
        if time_embed_proj:
            self.time_expand = nn.Linear(time_embed_dim, out_channels)

        ### Class Embedding Mapping ###
        self.class_expand = None

        if class_embed_proj:
            self.class_expand = nn.Linear(class_embed_dim, out_channels)

        ### Residual Connection Upchannels ###
        self.identity_conv = nn.Identity()
        if in_channels != out_channels:
            self.identity_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self,
                x,
                time_embed=None,
                class_conditioning=None):

        residual_connection = x

        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)

        if time_embed is not None:
            if self.time_expand is None:
                raise Exception("Passing in Time Embedding into ResidualBlock without time_embed_proj = True")
            else:

                ### Project Time Embedding from (B x time_embed_dim) -> (B x out_channels) ###
                time_embed = self.time_expand(time_embed)

                ### Reshape (B x out_channels) -> (B x out_channels x 1 x 1) ###
                time_embed = time_embed.reshape((*time_embed.shape, 1, 1))

                ### Add Time Information to Images (B x out_channel x h x w) + (B x out_channels x 1 x 1) ###
                x = x + time_embed

        if class_conditioning is not None:

            if self.class_expand is None:
                raise Exception("Passing in Class Conditioning to ResidualBlock without class_embed_proj = True")
            else:

                ### Project Class Embedding from (B x class_embed_dim) -> (B x out_channels) ###
                class_conditioning = self.class_expand(class_conditioning)

                ### Reshape (B x out_channels) -> (B x out_channels x 1 x 1) ###
                class_conditioning = class_conditioning.reshape((*class_conditioning.shape, 1, 1))

                ### Add Time Information to Images (B x out_channel x h x w) + (B x out_channels x 1 x 1) ###
                x = x + class_conditioning

        x = self.norm2(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.conv2(x)

        ### Residual Connection ###
        x = x + self.identity_conv(residual_connection)

        return x