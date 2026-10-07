"""
Discriminator Network Architecture for Synthetic Fashion DCGAN (Stage 4).
Convolutional binary classifier that maps a 28x28 grayscale image tensor [B, 1, 28, 28]
to a single probability scalar [B, 1] indicating whether the image is real (1) or fake (0).
"""

import torch
import torch.nn as nn
from utils.config import IMAGE_CHANNELS, IMAGE_SIZE


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


class Discriminator(nn.Module):
    """
    Deep Convolutional Discriminator (DCGAN Discriminator).

    Architecture Flow:
        [batch_size, 1, 28, 28]  (Input Image Tensor)
               ↓ Conv2d (1 -> 64, kernel=4, stride=2, padding=1) + LeakyReLU(0.2)
        [batch_size, 64, 14, 14]
               ↓ Conv2d (64 -> 128, kernel=4, stride=2, padding=1) + BatchNorm2d + LeakyReLU(0.2)
        [batch_size, 128, 7, 7]
               ↓ Flatten (128 * 7 * 7 = 6,272 features)
        [batch_size, 6272]
               ↓ Linear Layer (6,272 -> 1) + Sigmoid Activation
        [batch_size, 1]  (Probability of Image Being Real: 0=Fake, 1=Real)
    """

    def __init__(self, in_channels: int = IMAGE_CHANNELS):
        super(Discriminator, self).__init__()
        self.in_channels = in_channels

        # Convolutional Feature Extraction Layers
        self.feature_extractor = nn.Sequential(
            # Block 1: 28x28 -> 14x14 (No BatchNorm on first Discriminator layer as per DCGAN guidelines)
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # Block 2: 14x14 -> 7x7
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Classification Head: Project 7x7 feature map to binary probability [B, 1]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid(),
        )

        # Apply standard DCGAN weight initialization
        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Discriminator.

        Args:
            x (torch.Tensor): Image batch tensor of shape (batch_size, in_channels, 28, 28).

        Returns:
            torch.Tensor: Probability scalar tensor of shape (batch_size, 1).
        """
        features = self.feature_extractor(x)
        validity = self.classifier(features)
        return validity
