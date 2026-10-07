"""
Inference & Model Evaluation Module for Synthetic Fashion DCGAN (Stage 6).
Provides functions to load trained checkpoints, generate synthetic samples,
measure GPU/CPU inference latency, and evaluate Discriminator classification accuracy.
"""

import os
import time
from typing import Dict, Tuple
import torch

import utils.config as config
from models import Discriminator, Generator


def load_generator(
    checkpoint_path: str = os.path.join(config.CHECKPOINTS_DIR, "generator.pth"),
    device: torch.device = config.DEVICE,
) -> Generator:
    """
    Loads trained Generator model weights from checkpoint.

    Args:
        checkpoint_path (str): Path to saved generator.pth file.
        device (torch.device): Compute target device.

    Returns:
        Generator: Loaded Generator model in evaluation mode.
    """
    generator = Generator(latent_dim=config.LATENT_DIM, out_channels=config.IMAGE_CHANNELS).to(device)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Generator checkpoint file not found at: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    generator.load_state_dict(state_dict)
    generator.eval()
    return generator


def load_discriminator(
    checkpoint_path: str = os.path.join(config.CHECKPOINTS_DIR, "discriminator.pth"),
    device: torch.device = config.DEVICE,
) -> Discriminator:
    """
    Loads trained Discriminator model weights from checkpoint.

    Args:
        checkpoint_path (str): Path to saved discriminator.pth file.
        device (torch.device): Compute target device.

    Returns:
        Discriminator: Loaded Discriminator model in evaluation mode.
    """
    discriminator = Discriminator(in_channels=config.IMAGE_CHANNELS).to(device)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Discriminator checkpoint file not found at: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    discriminator.load_state_dict(state_dict)
    discriminator.eval()
    return discriminator


def generate_synthetic_images(
    generator: Generator,
    num_samples: int = config.BATCH_SIZE,
    device: torch.device = config.DEVICE,
) -> torch.Tensor:
    """
    Generates synthetic fashion images from random latent noise vectors.

    Args:
        generator (Generator): Trained Generator model instance.
        num_samples (int): Number of synthetic images to generate.
        device (torch.device): Compute device.

    Returns:
        torch.Tensor: Synthetic image batch tensor of shape (num_samples, 1, 28, 28).
    """
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, config.LATENT_DIM, device=device)
        fake_images = generator(z)
    return fake_images


def measure_inference_latency(
    generator: Generator,
    batch_size: int = config.BATCH_SIZE,
    device: torch.device = config.DEVICE,
    num_runs: int = 10,
) -> Dict[str, float]:
    """
    Measures Generator inference latency with proper CUDA synchronization and warm-up.

    Args:
        generator (Generator): Generator model.
        batch_size (int): Batch size for inference test.
        device (torch.device): Active device.
        num_runs (int): Number of timing iterations.

    Returns:
        Dict[str, float]: Dictionary containing total_time_seconds, per_image_latency_ms, images_per_second.
    """
    generator.eval()
    dummy_noise = torch.randn(batch_size, config.LATENT_DIM, device=device)

    # 1. Warm-up run to initialize CUDA context/caches
    with torch.no_grad():
        _ = generator(dummy_noise)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # 2. Timing benchmark iterations
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = generator(dummy_noise)
            if device.type == "cuda":
                torch.cuda.synchronize()
    total_time_seconds = time.perf_counter() - start_time

    avg_batch_time = total_time_seconds / num_runs
    total_images_evaluated = batch_size * num_runs
    per_image_latency_ms = (avg_batch_time / batch_size) * 1000.0
    images_per_second = total_images_evaluated / total_time_seconds

    return {
        "total_time_seconds": round(total_time_seconds, 6),
        "per_image_latency_ms": round(per_image_latency_ms, 4),
        "images_per_second": round(images_per_second, 2),
    }


def evaluate_discriminator_predictions(
    discriminator: Discriminator,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
) -> Dict[str, float]:
    """
    Evaluates trained Discriminator classification predictions and accuracy.

    NOTE: The Discriminator is a binary authenticity classifier (Real vs. Fake),
          NOT a 10-class Fashion-MNIST object classifier.

    Args:
        discriminator (Discriminator): Trained Discriminator model.
        real_images (torch.Tensor): Real image batch [B, 1, 28, 28].
        fake_images (torch.Tensor): Fake generated image batch [B, 1, 28, 28].

    Returns:
        Dict[str, float]: Detailed classification accuracy and prediction statistics.
    """
    discriminator.eval()
    device = next(discriminator.parameters()).device
    real_images = real_images.to(device)
    fake_images = fake_images.to(device)

    with torch.no_grad():
        d_real = discriminator(real_images)
        d_fake = discriminator(fake_images)

    # Convert probability scalars to binary predictions (threshold = 0.5)
    real_correct = (d_real >= 0.5).sum().item()
    fake_correct = (d_fake < 0.5).sum().item()

    num_real = real_images.size(0)
    num_fake = fake_images.size(0)

    real_acc = (real_correct / num_real) * 100.0
    fake_acc = (fake_correct / num_fake) * 100.0
    overall_acc = ((real_correct + fake_correct) / (num_real + num_fake)) * 100.0

    return {
        "average_real_prediction": round(float(d_real.mean().item()), 4),
        "min_real_prediction": round(float(d_real.min().item()), 4),
        "max_real_prediction": round(float(d_real.max().item()), 4),
        "average_fake_prediction": round(float(d_fake.mean().item()), 4),
        "min_fake_prediction": round(float(d_fake.min().item()), 4),
        "max_fake_prediction": round(float(d_fake.max().item()), 4),
        "real_accuracy": round(real_acc, 2),
        "fake_accuracy": round(fake_acc, 2),
        "overall_accuracy": round(overall_acc, 2),
    }


def generate_samples():
    """Wrapper function for backwards compatibility."""
    generator = load_generator()
    return generate_synthetic_images(generator)
