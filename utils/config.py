"""
Centralized Configuration Module for Synthetic Fashion DCGAN.
Contains hyperparameter definitions, dataset specifications, paths, and device selection.
"""

import os
import torch

# Directory Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")

# Dataset Specifications
IMAGE_SIZE = 28
IMAGE_CHANNELS = 1
NUM_CLASSES = 10

# Model Architecture Parameters
LATENT_DIM = 100

# Training Hyperparameters
BATCH_SIZE = 128
NUM_EPOCHS = 25
LEARNING_RATE = 0.0002
BETA1 = 0.5
BETA2 = 0.999

# Compute Device Selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
