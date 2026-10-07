"""
Visualization Utilities for Synthetic Fashion DCGAN.
Functions for formatting, rendering, saving grids of real or synthetic fashion images,
loss curves, side-by-side comparisons, and epoch evolution progression.
"""

import os
import matplotlib
matplotlib.use("Agg")  # Set non-interactive thread-safe backend before importing pyplot
import matplotlib.pyplot as plt

import torch
import torchvision.utils as vutils

from .config import BASE_DIR, SAMPLES_DIR


def plot_sample_grid(
    images: torch.Tensor,
    save_path: str = os.path.join(SAMPLES_DIR, "real_fashion_mnist.png"),
    nrow: int = 4,
    show: bool = False,
    title: str = "Real Fashion-MNIST Samples",
) -> str:
    """
    Renders and saves a grid of images.

    Args:
        images (torch.Tensor): Tensor of shape (N, C, H, W) with pixel values normalized to [-1, 1].
        save_path (str): Output file path for saved image grid.
        nrow (int): Number of images per row in grid.
        show (bool): If True, calls plt.show() for interactive display.
        title (str): Title for figure.

    Returns:
        str: Path to saved image file.
    """
    # Select first 16 images if more provided
    images_to_plot = images[:16]

    # Create grid of images denormalized from [-1, 1] to [0, 1] using value_range=(-1, 1)
    grid_tensor = vutils.make_grid(
        images_to_plot,
        nrow=nrow,
        padding=2,
        normalize=True,
        value_range=(-1, 1),
    )

    # Convert PyTorch tensor format (C, H, W) to NumPy layout (H, W, C)
    grid_np = grid_tensor.cpu().numpy().transpose((1, 2, 0))

    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid_np, cmap="gray" if grid_np.shape[2] == 1 else None)
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=10)
    plt.tight_layout()

    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    return save_path


def plot_real_vs_generated(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    save_path: str = os.path.join(SAMPLES_DIR, "real_vs_generated.png"),
) -> str:
    """
    Renders a side-by-side comparison grid of Real Fashion-MNIST vs. Generated Fashion images.

    Args:
        real_images (torch.Tensor): Real image batch normalized to [-1, 1].
        fake_images (torch.Tensor): Generated image batch normalized to [-1, 1].
        save_path (str): File path to save output comparison grid.

    Returns:
        str: Path to saved image file.
    """
    real_grid = (
        vutils.make_grid(real_images[:16], nrow=4, padding=2, normalize=True, value_range=(-1, 1))
        .cpu()
        .numpy()
        .transpose((1, 2, 0))
    )
    fake_grid = (
        vutils.make_grid(fake_images[:16], nrow=4, padding=2, normalize=True, value_range=(-1, 1))
        .cpu()
        .numpy()
        .transpose((1, 2, 0))
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(real_grid, cmap="gray" if real_grid.shape[2] == 1 else None)
    ax1.axis("off")
    ax1.set_title("REAL FASHION-MNIST", fontsize=14, fontweight="bold", pad=10)

    ax2.imshow(fake_grid, cmap="gray" if fake_grid.shape[2] == 1 else None)
    ax2.axis("off")
    ax2.set_title("GENERATED FASHION", fontsize=14, fontweight="bold", pad=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    return save_path


def plot_loss_curves(
    history_json_path: str = os.path.join(BASE_DIR, "training_history.json"),
    save_path: str = os.path.join(SAMPLES_DIR, "training_loss_curves.png"),
) -> str:
    """
    Plots Generator and Discriminator loss curves across training epochs from training history JSON.

    Args:
        history_json_path (str): Path to training_history.json.
        save_path (str): Output file path for saved loss plot.

    Returns:
        str: Path to saved plot image.
    """
    import json

    with open(history_json_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    epochs = [item["epoch"] for item in history]
    g_losses = [item["generator_loss"] for item in history]
    d_losses = [item["discriminator_loss"] for item in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, g_losses, label="Generator Loss", color="#4f46e5", linewidth=2.2, marker="o", markersize=3)
    ax.plot(epochs, d_losses, label="Discriminator Loss", color="#ef4444", linewidth=2.2, marker="s", markersize=3)

    ax.set_title("DCGAN Adversarial Training Loss Curves", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Binary Cross Entropy Loss", fontsize=12)
    ax.set_xticks(range(1, len(epochs) + 1, 2) if len(epochs) > 15 else epochs)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=11, loc="upper right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    return save_path


def plot_epoch_evolution(
    samples_dir: str = SAMPLES_DIR,
    save_path: str = os.path.join(SAMPLES_DIR, "epoch_evolution.png"),
) -> str:
    """
    Combines saved epoch sample images (Epochs 1, 5, 10, 15, 20, 25) into a progression grid
    demonstrating progressive generator quality improvement.

    Args:
        samples_dir (str): Directory containing saved epoch sample PNGs.
        save_path (str): Output file path for combined evolution figure.

    Returns:
        str: Path to saved image file.
    """
    from PIL import Image

    target_epochs = [1, 5, 10, 15, 20, 25]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for idx, ep in enumerate(target_epochs):
        img_filename = f"epoch_{ep:03d}.png"
        img_path = os.path.join(samples_dir, img_filename)
        if os.path.exists(img_path):
            img = Image.open(img_path)
            axes[idx].imshow(img)
            axes[idx].set_title(f"Epoch {ep}", fontsize=13, fontweight="bold")
        else:
            axes[idx].text(0.5, 0.5, f"Epoch {ep}\n(Not Found)", ha="center", va="center")
        axes[idx].axis("off")

    plt.suptitle("DCGAN Generator Output Evolution Over Training Epochs", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    return save_path
