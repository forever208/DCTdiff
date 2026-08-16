"""Self-contained CelebA256 F8 AutoencoderKL used by DCTdiff.

The architecture is fixed to ``SDVAE_config_celeba256.yaml`` from the VAE
project.  Keeping it local avoids coupling LDM training to that project's
configuration and training-only dependencies.
"""

from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


def nonlinearity(x):
    return x * torch.sigmoid(x)


def normalize(in_channels):
    return nn.GroupNorm(
        num_groups=32,
        num_channels=in_channels,
        eps=1e-6,
        affine=True,
    )


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x) if self.with_conv else x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, 3, 2, 0)

    def forward(self, x):
        if self.with_conv:
            x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
            return self.conv(x)
        return F.avg_pool2d(x, kernel_size=2, stride=2)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, 3, 1, 1)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, self.out_channels)
        self.norm2 = normalize(self.out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(
                    in_channels, self.out_channels, 3, 1, 1
                )
            else:
                self.nin_shortcut = nn.Conv2d(
                    in_channels, self.out_channels, 1, 1, 0
                )

    def forward(self, x, temb=None):
        h = self.conv1(nonlinearity(self.norm1(x)))
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        h = self.conv2(self.dropout(nonlinearity(self.norm2(h))))
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        h = self.norm(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        batch, channels, height, width = q.shape
        q = q.reshape(batch, channels, height * width).permute(0, 2, 1)
        k = k.reshape(batch, channels, height * width)
        weights = torch.bmm(q, k) * (int(channels) ** -0.5)
        weights = F.softmax(weights, dim=2).permute(0, 2, 1)
        h = torch.bmm(v.reshape(batch, channels, height * width), weights)
        h = h.reshape(batch, channels, height, width)
        return x + self.proj_out(h)


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch=128,
        ch_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        dropout=0.0,
        in_channels=3,
        resolution=256,
        z_channels=4,
        double_z=True,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.conv_in = nn.Conv2d(in_channels, ch, 3, 1, 1)

        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[level]
            block_out = ch * ch_mult[level]
            for _ in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=0,
                        dropout=dropout,
                    )
                )
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, with_conv=True)
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=0,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=0,
            dropout=dropout,
        )
        self.norm_out = normalize(block_in)
        out_channels = 2 * z_channels if double_z else z_channels
        self.conv_out = nn.Conv2d(block_in, out_channels, 3, 1, 1)

    def forward(self, x):
        h = self.conv_in(x)
        for level in range(self.num_resolutions):
            for block in self.down[level].block:
                h = block(h, None)
            if level != self.num_resolutions - 1:
                h = self.down[level].downsample(h)
        h = self.mid.block_1(h, None)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, None)
        return self.conv_out(nonlinearity(self.norm_out(h)))


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch=128,
        out_ch=3,
        ch_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        dropout=0.0,
        resolution=256,
        z_channels=4,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution

        block_in = ch * ch_mult[self.num_resolutions - 1]
        self.z_shape = (
            1,
            z_channels,
            resolution // 2 ** (self.num_resolutions - 1),
            resolution // 2 ** (self.num_resolutions - 1),
        )
        self.conv_in = nn.Conv2d(z_channels, block_in, 3, 1, 1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=0,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=0,
            dropout=dropout,
        )

        self.up = nn.ModuleList()
        for level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[level]
            for _ in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=0,
                        dropout=dropout,
                    )
                )
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if level != 0:
                up.upsample = Upsample(block_in, with_conv=True)
            self.up.insert(0, up)

        self.norm_out = normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, 3, 1, 1)

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid.block_1(h, None)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, None)
        for level in reversed(range(self.num_resolutions)):
            for block in self.up[level].block:
                h = block(h, None)
            if level != 0:
                h = self.up[level].upsample(h)
        return self.conv_out(nonlinearity(self.norm_out(h)))


class DiagonalGaussianDistribution:
    """Diagonal Gaussian represented by concatenated mean and log variance."""

    def __init__(self, parameters):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self):
        return self.mean + self.std * torch.randn_like(self.mean)

    def mode(self):
        return self.mean


class AutoencoderKLF8(nn.Module):
    """Inference-only SDVAE matching the CelebA256 F8 training config."""

    latent_channels = 4
    downsample_factor = 8

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.quant_conv = nn.Conv2d(8, 8, 1)
        self.post_quant_conv = nn.Conv2d(4, 4, 1)
        self.eval()
        self.requires_grad_(False)

    def train(self, mode=True):
        # This model is a frozen first stage even while the LDM is training.
        return super().train(False)

    def encode(self, x):
        moments = self.quant_conv(self.encoder(x))
        return DiagonalGaussianDistribution(moments)

    def encode_images(self, x, scale_factor=1.0):
        return self.encode(x).sample() * scale_factor

    def sample(self, moments, scale_factor=1.0):
        return DiagonalGaussianDistribution(moments).sample() * scale_factor

    def decode(self, z, scale_factor=1.0):
        z = z / scale_factor
        return self.decoder(self.post_quant_conv(z))


def _checkpoint_state(checkpoint) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("VAE checkpoint must contain a state dictionary")

    state = checkpoint.get("model")
    if state is None:
        state = checkpoint.get("state_dict")
    if state is None and checkpoint and all(
        torch.is_tensor(value) for value in checkpoint.values()
    ):
        state = checkpoint
    if state is None:
        raise KeyError("VAE checkpoint has no 'model' or 'state_dict' entry")

    if state and next(iter(state)).startswith("module."):
        state = {
            key.removeprefix("module."): value for key, value in state.items()
        }
    return state


def load_vae_f8(checkpoint_path, device=None):
    """Build and strictly validate the inference portion of an SDVAE checkpoint."""
    checkpoint_path = Path(checkpoint_path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"VAE checkpoint does not exist: {checkpoint_path}")

    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model = AutoencoderKLF8()
    incompatible = model.load_state_dict(_checkpoint_state(checkpoint), strict=False)
    if incompatible.missing_keys:
        preview = ", ".join(incompatible.missing_keys[:10])
        raise RuntimeError(f"VAE checkpoint is missing inference weights: {preview}")

    unexpected = [
        key for key in incompatible.unexpected_keys if not key.startswith("loss.")
    ]
    if unexpected:
        preview = ", ".join(unexpected[:10])
        raise RuntimeError(f"Unexpected non-loss VAE checkpoint weights: {preview}")

    model.eval().requires_grad_(False)
    if device is not None:
        model = model.to(device)
    return model
