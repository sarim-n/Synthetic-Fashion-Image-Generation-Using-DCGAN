"""
Adversarial DCGAN Training Pipeline (Stage 5 & Experiments).
Trains Generator and Discriminator on Fashion-MNIST using Binary Cross Entropy (BCE) Loss
and Adam Optimizers. Saves intermediate sample grids, model weight checkpoints, and training metrics JSON.
"""

import json
import os
import time
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim

import utils.config as config
from models import Discriminator, Generator
from utils.data_loader import get_data_loader
from utils.visualization import plot_sample_grid


def train_dcgan(
    num_epochs: int = config.NUM_EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    beta1: float = config.BETA1,
    beta2: float = config.BETA2,
    is_sanity_test: bool = False,
    samples_dir: str = config.SAMPLES_DIR,
    checkpoints_dir: str = config.CHECKPOINTS_DIR,
    history_json_path: str = os.path.join(config.BASE_DIR, "training_history.json"),
) -> List[Dict]:
    """
    Executes adversarial DCGAN training loop for Generator and Discriminator.

    Args:
        num_epochs (int): Total number of epochs to train.
        batch_size (int): DataLoader batch size.
        learning_rate (float): Adam optimizer learning rate.
        beta1 (float): Adam beta1 hyperparameter.
        beta2 (float): Adam beta2 hyperparameter.
        is_sanity_test (bool): If True, runs only 1 batch to verify pipeline integrity.
        samples_dir (str): Directory where output sample PNGs will be saved.
        checkpoints_dir (str): Directory where model .pth weight files will be saved.
        history_json_path (str): File path for saved training_history.json.

    Returns:
        List[Dict]: History of per-epoch metrics (epoch, generator_loss, discriminator_loss, epoch_time).
    """
    device = config.DEVICE
    print(f"\n[Hardware Check] Selected PyTorch Device: '{device}'")
    print(f"  - PyTorch Version     : {torch.__version__}")
    print(f"  - CUDA Available      : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - Active GPU Name     : {torch.cuda.get_device_name(0)}")

    # 1. Load Dataset & DataLoader
    print("\n[1] Initializing Fashion-MNIST DataLoader...")
    _, dataloader = get_data_loader(
        data_dir=config.DATA_DIR,
        batch_size=batch_size,
        download=True,
    )

    # 2. Instantiate Models and Move to Device
    print("[2] Instantiating DCGAN Generator & Discriminator Models...")
    generator = Generator(latent_dim=config.LATENT_DIM, out_channels=config.IMAGE_CHANNELS).to(device)
    discriminator = Discriminator(in_channels=config.IMAGE_CHANNELS).to(device)

    # 3. Setup Binary Cross Entropy Loss & Adam Optimizers
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, beta2))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, beta2))

    # 4. Create Fixed Latent Noise Vector for Sample Tracking (16 samples for a 4x4 grid)
    fixed_noise = torch.randn(16, config.LATENT_DIM, device=device)

    # Ensure output directories exist
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    history: List[Dict] = []
    total_start_time = time.time()

    print(f"\n[3] Starting DCGAN Training ({'SANITY TEST Mode (1 Batch)' if is_sanity_test else f'{num_epochs} Epochs'})...")
    print("=" * 70)

    epochs_to_run = 1 if is_sanity_test else num_epochs

    for epoch in range(1, epochs_to_run + 1):
        epoch_start_time = time.time()
        generator.train()
        discriminator.train()

        running_g_loss = 0.0
        running_d_loss = 0.0
        batches_processed = 0

        for i, (real_images, _) in enumerate(dataloader):
            current_batch_size = real_images.size(0)
            real_images = real_images.to(device)

            # Target labels: Real = 1, Fake = 0
            real_targets = torch.ones(current_batch_size, 1, device=device)
            fake_targets = torch.zeros(current_batch_size, 1, device=device)

            # --------------------------------------------------------
            # Step A: Train Discriminator -> max log(D(x)) + log(1 - D(G(z)))
            # --------------------------------------------------------
            discriminator.zero_grad()

            # Discriminator loss on real images
            d_real_output = discriminator(real_images)
            d_real_loss = criterion(d_real_output, real_targets)

            # Discriminator loss on fake images (detach G to avoid updating G parameters)
            noise = torch.randn(current_batch_size, config.LATENT_DIM, device=device)
            fake_images = generator(noise)
            d_fake_output = discriminator(fake_images.detach())
            d_fake_loss = criterion(d_fake_output, fake_targets)

            # Combined Discriminator Loss
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            optimizer_d.step()

            # --------------------------------------------------------
            # Step B: Train Generator -> max log(D(G(z))) <=> min BCE(D(G(z)), 1)
            # --------------------------------------------------------
            generator.zero_grad()

            # Pass fake images through Discriminator (no detach)
            g_fake_output = discriminator(fake_images)
            g_loss = criterion(g_fake_output, real_targets)  # Generator wants D to classify fake as 1 (real)

            g_loss.backward()
            optimizer_g.step()

            # Accumulate batch metrics
            running_g_loss += g_loss.item()
            running_d_loss += d_loss.item()
            batches_processed += 1

            # Break early if running sanity test
            if is_sanity_test:
                break

        # Compute average losses for epoch
        avg_g_loss = running_g_loss / batches_processed
        avg_d_loss = running_d_loss / batches_processed
        epoch_time = time.time() - epoch_start_time

        # Store epoch metrics
        epoch_metrics = {
            "epoch": epoch,
            "generator_loss": round(avg_g_loss, 4),
            "discriminator_loss": round(avg_d_loss, 4),
            "epoch_time": round(epoch_time, 2),
        }
        history.append(epoch_metrics)

        # Print epoch progress
        print(
            f"Epoch [{epoch:02d}/{num_epochs}] | "
            f"Discriminator Loss: {avg_d_loss:.4f} | "
            f"Generator Loss: {avg_g_loss:.4f} | "
            f"Epoch Time: {epoch_time:.2f}s"
        )

        # Generate & save fixed-noise sample grid after each epoch
        generator.eval()
        with torch.no_grad():
            sample_grid_path = os.path.join(samples_dir, f"epoch_{epoch:03d}.png")
            fake_samples = generator(fixed_noise)
            plot_sample_grid(
                images=fake_samples.cpu(),
                save_path=sample_grid_path,
                nrow=4,
                show=False,
                title=f"DCGAN Synthetic Samples - Epoch {epoch:03d}",
            )

    total_training_time = time.time() - total_start_time
    print("=" * 70)
    print(f"Training completed in {total_training_time:.2f} seconds.")

    # Save final model checkpoints
    g_checkpoint_path = os.path.join(checkpoints_dir, "generator.pth")
    d_checkpoint_path = os.path.join(checkpoints_dir, "discriminator.pth")
    torch.save(generator.state_dict(), g_checkpoint_path)
    torch.save(discriminator.state_dict(), d_checkpoint_path)
    print(f"  - Generator Checkpoint Saved to     : {g_checkpoint_path}")
    print(f"  - Discriminator Checkpoint Saved to : {d_checkpoint_path}")

    # Save training history JSON
    os.makedirs(os.path.dirname(history_json_path), exist_ok=True)
    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"  - Metrics History Saved to         : {history_json_path}")

    return history


def train():
    """Default training wrapper."""
    return train_dcgan(num_epochs=config.NUM_EPOCHS, is_sanity_test=False)
