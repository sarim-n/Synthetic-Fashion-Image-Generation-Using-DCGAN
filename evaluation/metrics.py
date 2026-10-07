"""
Evaluation Metrics for Synthetic Fashion DCGAN (Stage 6).
Implements lightweight, explainable diversity statistics to monitor mode collapse
and sample variance across generated image batches.
"""

from typing import Dict, Union
import torch


def calculate_pairwise_pixel_diversity(images: torch.Tensor) -> Dict[str, Union[str, float]]:
    """
    Calculates the mean pairwise L1 pixel distance across a batch of generated images.

    Higher values indicate diverse sample generation without mode collapse.
    Extremely low values (<0.05) signal potential mode collapse (identical image outputs).

    Args:
        images (torch.Tensor): Image batch tensor of shape (N, C, H, W) normalized to [-1, 1].

    Returns:
        Dict[str, Union[str, float]]: Dictionary containing metric name, value, and interpretation.
    """
    num_samples = images.size(0)
    if num_samples < 2:
        return {
            "metric": "Mean Pairwise L1 Pixel Distance",
            "value": 0.0,
            "interpretation": "Insufficient samples for pairwise diversity calculation.",
        }

    # Flatten images to 2D matrix: [N, 784]
    flattened = images.view(num_samples, -1)

    # Compute pairwise L1 distance normalized by feature dimension (784)
    pairwise_distances = torch.cdist(flattened, flattened, p=1.0) / flattened.size(1)

    # Extract upper triangular matrix (excluding zero self-distances on main diagonal)
    triu_indices = torch.triu_indices(num_samples, num_samples, offset=1)
    unique_distances = pairwise_distances[triu_indices[0], triu_indices[1]]

    mean_diversity = unique_distances.mean().item()

    if mean_diversity < 0.05:
        interpretation = "Warning: Low pixel diversity detected. Potential mode collapse."
    elif mean_diversity < 0.20:
        interpretation = "Moderate sample diversity across generated fashion items."
    else:
        interpretation = (
            "Rich intra-batch pixel variation indicating healthy sample diversity without mode collapse."
        )

    return {
        "metric": "Mean Pairwise L1 Pixel Distance",
        "value": round(float(mean_diversity), 4),
        "interpretation": interpretation,
    }


def evaluate_model():
    """Legacy placeholder wrapper for backwards compatibility."""
    pass
