"""
Generator Network Architecture for Synthetic Fashion DCGAN (Stage 3).
Transforms a 100-dimensional latent noise vector z ~ N(0, I) into a 28x28 grayscale fashion image tensor.
"""

import torch
import torch.nn as nn
from utils.config import IMAGE_CHANNELS, LATENT_DIM


def weights_init(m: nn.Module):
    """
    Standard DCGAN weight initialization:
    - Convolutional and Linear layers: Gaussian distribution N(0, 0.02)
    - BatchNorm layers: Gaussian distribution N(1.0, 0.02) for weights, 0 for bias
    """
    classname = m.__class__.__name__
    if "Conv" in classname or "Linear" in classname:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif "BatchNorm" in classname:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    """
    Deep Convolutional Generator (DCGAN Generator).

    Architecture Flow:
        [batch_size, 100]  (Latent Noise Input)
               ↓ Linear Layer + BatchNorm1d + ReLU
        [batch_size, 128 * 7 * 7]  (= 6,272 features)
               ↓ Reshape to 4D Spatial Feature Map
        [batch_size, 128, 7, 7]
               ↓ ConvTranspose2d (128 -> 64, kernel=4, stride=2, padding=1) + BatchNorm2d + ReLU
        [batch_size, 64, 14, 14]
               ↓ ConvTranspose2d (64 -> 1, kernel=4, stride=2, padding=1) + Tanh
        [batch_size, 1, 28, 28]  (Synthetic Grayscale Image Output)
    """

    def __init__(self, latent_dim: int = LATENT_DIM, out_channels: int = IMAGE_CHANNELS):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.init_size = 7

        # 1. Linear Projection Layer: Maps 1D latent vector to 3D spatial representation (128 x 7 x 7)
        self.fc = nn.Sequential(
            nn.Linear(self.latent_dim, 128 * self.init_size * self.init_size),
            nn.BatchNorm1d(128 * self.init_size * self.init_size),
            nn.ReLU(True),
        )

        # 2. Transposed Convolution Upsampling Blocks
        self.conv_blocks = nn.Sequential(
            # Transposed Conv 1: 7x7 -> 14x14
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # Transposed Conv 2: 14x14 -> 28x28
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=self.out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.Tanh(),
        )

        # Apply standard DCGAN weight initialization
        self.apply(weights_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Generator.

        Args:
            z (torch.Tensor): Input latent noise tensor of shape (batch_size, latent_dim).

        Returns:
            torch.Tensor: Generated synthetic image batch of shape (batch_size, out_channels, 28, 28).
        """
        # Step 1: Linear projection -> (batch_size, 128 * 7 * 7)
        out = self.fc(z)
        # Step 2: Reshape to 4D tensor -> (batch_size, 128, 7, 7)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        # Step 3: Transposed Convolutional upsampling -> (batch_size, 1, 28, 28)
        img = self.conv_blocks(out)
        return img
