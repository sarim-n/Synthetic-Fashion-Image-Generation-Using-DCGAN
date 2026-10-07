"""
Synthetic Fashion DCGAN - Evaluation Pipeline Entry Point (Stage 6).
Loads trained model checkpoints, evaluates inference latency, tests Discriminator
classification accuracy, calculates pixel diversity metrics, renders visualization grids,
and saves evaluation_results.json.
"""

import json
import os
import sys

# Add project root to sys.path if executing script directly
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import torch
import utils.config as config
from evaluation.inference import (
    evaluate_discriminator_predictions,
    generate_synthetic_images,
    load_discriminator,
    load_generator,
    measure_inference_latency,
)
from evaluation.metrics import calculate_pairwise_pixel_diversity
from utils.data_loader import get_data_loader
from utils.visualization import (
    plot_epoch_evolution,
    plot_loss_curves,
    plot_real_vs_generated,
    plot_sample_grid,
)


def run_evaluation() -> dict:
    print("=" * 70)
    print("           SYNTHETIC FASHION GENERATION DCGAN - STAGE 6 EVALUATION          ")
    print("=" * 70)

    device = config.DEVICE
    print(f"\n[1] Environment & Model Checkpoint Setup:")
    print(f"  - Compute Device Target      : '{device}'")
    print(f"  - PyTorch Version            : {torch.__version__}")
    print(f"  - CUDA Available             : {torch.cuda.is_available()}")

    # 1. Load Trained Checkpoints
    g_ckpt = os.path.join(config.CHECKPOINTS_DIR, "generator.pth")
    d_ckpt = os.path.join(config.CHECKPOINTS_DIR, "discriminator.pth")

    generator = load_generator(checkpoint_path=g_ckpt, device=device)
    discriminator = load_discriminator(checkpoint_path=d_ckpt, device=device)
    print(f"  - Generator Checkpoint Loaded: {g_ckpt}")
    print(f"  - Discriminator Checkpoint   : {d_ckpt}")

    # 2. Load Real Dataset Batch & Generate Fresh Fake Batch
    print("\n[2] Data Preparation & Fresh Sample Generation:")
    _, dataloader = get_data_loader(data_dir=config.DATA_DIR, batch_size=config.BATCH_SIZE, download=True)
    real_images, _ = next(iter(dataloader))
    real_images = real_images.to(device)

    fake_images = generate_synthetic_images(generator, num_samples=config.BATCH_SIZE, device=device)

    print(f"  - Real Image Batch Shape     : {list(real_images.shape)}")
    print(f"  - Generated Image Shape     : {list(fake_images.shape)}")
    print(f"  - Generated Min / Max Pixels : {fake_images.min().item():.4f} / {fake_images.max().item():.4f}")

    # Assertions
    assert fake_images.shape == (config.BATCH_SIZE, 1, 28, 28), "Invalid fake image tensor shape."
    assert fake_images.min().item() >= -1.05 and fake_images.max().item() <= 1.05, "Fake pixels out of [-1, 1] range."

    # 3. Measure Inference Latency
    print("\n[3] Measuring Generator Inference Latency...")
    latency_stats = measure_inference_latency(generator, batch_size=config.BATCH_SIZE, device=device, num_runs=10)
    print(f"  - Total Benchmark Time (10x) : {latency_stats['total_time_seconds']:.6f} seconds")
    print(f"  - Per-Image Latency          : {latency_stats['per_image_latency_ms']:.4f} ms")
    print(f"  - Throughput                 : {latency_stats['images_per_second']:.2f} images/sec")

    # 4. Evaluate Discriminator Classification Accuracy
    print("\n[4] Evaluating Discriminator Binary Authenticity Classification...")
    disc_stats = evaluate_discriminator_predictions(discriminator, real_images, fake_images)
    print(f"  - Average D(Real Image)      : {disc_stats['average_real_prediction']:.4f}")
    print(f"  - Average D(Fake Image)      : {disc_stats['average_fake_prediction']:.4f}")
    print(f"  - Real Classification Acc    : {disc_stats['real_accuracy']:.2f}%")
    print(f"  - Fake Classification Acc    : {disc_stats['fake_accuracy']:.2f}%")
    print(f"  - Overall Binary Accuracy    : {disc_stats['overall_accuracy']:.2f}%")

    # 5. Calculate Intra-Batch Diversity Statistic
    print("\n[5] Calculating Sample Diversity Statistic...")
    diversity_stats = calculate_pairwise_pixel_diversity(fake_images)
    print(f"  - Diversity Metric Name      : {diversity_stats['metric']}")
    print(f"  - Diversity Score            : {diversity_stats['value']}")
    print(f"  - Interpretation             : {diversity_stats['interpretation']}")

    # 6. Generate All Required Visualizations
    print("\n[6] Rendering Evaluation Visualizations...")

    eval_gen_path = os.path.join(config.SAMPLES_DIR, "evaluation_generated.png")
    plot_sample_grid(fake_images.cpu(), save_path=eval_gen_path, nrow=4, show=False, title="Fresh Synthetic Evaluation Samples")
    print(f"  - Saved Fresh Samples Grid  : {eval_gen_path}")

    rvg_path = os.path.join(config.SAMPLES_DIR, "real_vs_generated.png")
    plot_real_vs_generated(real_images, fake_images, save_path=rvg_path)
    print(f"  - Saved Real vs. Generated  : {rvg_path}")

    loss_curve_path = os.path.join(config.SAMPLES_DIR, "training_loss_curves.png")
    plot_loss_curves(history_json_path=os.path.join(config.BASE_DIR, "training_history.json"), save_path=loss_curve_path)
    print(f"  - Saved Loss Curves Plot   : {loss_curve_path}")

    evolution_path = os.path.join(config.SAMPLES_DIR, "epoch_evolution.png")
    plot_epoch_evolution(samples_dir=config.SAMPLES_DIR, save_path=evolution_path)
    print(f"  - Saved Epoch Evolution Grid : {evolution_path}")

    # 7. Construct & Save Evaluation Results JSON
    results_dict = {
        "device": str(device),
        "batch_size": config.BATCH_SIZE,
        "inference": latency_stats,
        "discriminator": disc_stats,
        "diversity": diversity_stats,
    }

    results_json_path = os.path.join(config.BASE_DIR, "evaluation", "evaluation_results.json")
    os.makedirs(os.path.dirname(results_json_path), exist_ok=True)
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n[7] Evaluation Results JSON Saved to: {results_json_path}")

    # Concise Summary Terminal Printout
    print("\n" + "=" * 70)
    print("DCGAN EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Device               : {results_dict['device'].upper()}")
    print("Generator            : LOADED (checkpoints/generator.pth)")
    print("Discriminator        : LOADED (checkpoints/discriminator.pth)")
    print("\nInference Performance:")
    print(f"  Batch Size         : {results_dict['batch_size']}")
    print(f"  Generation Time    : {results_dict['inference']['total_time_seconds']:.6f} s (10 runs)")
    print(f"  Per-Image Latency  : {results_dict['inference']['per_image_latency_ms']:.4f} ms")
    print(f"  Throughput         : {results_dict['inference']['images_per_second']:.2f} img/s")
    print("\nDiscriminator Classification:")
    print(f"  Average D(real)    : {results_dict['discriminator']['average_real_prediction']}")
    print(f"  Average D(fake)    : {results_dict['discriminator']['average_fake_prediction']}")
    print(f"  Real Accuracy      : {results_dict['discriminator']['real_accuracy']}%")
    print(f"  Fake Accuracy      : {results_dict['discriminator']['fake_accuracy']}%")
    print(f"  Overall Accuracy   : {results_dict['discriminator']['overall_accuracy']}%")
    print("\nDiversity Analysis:")
    print(f"  Metric             : {results_dict['diversity']['metric']}")
    print(f"  Value              : {results_dict['diversity']['value']}")
    print(f"  Interpretation     : {results_dict['diversity']['interpretation']}")
    print("=" * 70)
    print("Evaluation completed successfully.\n")

    return results_dict


if __name__ == "__main__":
    run_evaluation()
