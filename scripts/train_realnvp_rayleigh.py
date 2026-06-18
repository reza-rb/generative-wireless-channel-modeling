"""Train RealNVP on Rayleigh fading channel samples.

This script runs the first complete end-to-end experiment:

    Rayleigh channel simulation
        -> PyTorch DataLoaders
        -> RealNVP training
        -> likelihood evaluation
        -> sample generation
        -> visualization and metrics export

Run from the repository root:

    python scripts/train_realnvp_rayleigh.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from gwcm.channels.rayleigh import generate_rayleigh_samples
from gwcm.data.channel_dataset import create_dataloaders
from gwcm.evaluation.likelihood import (
    compute_sample_statistics,
    evaluate_likelihood,
    generate_flow_samples,
)
from gwcm.models.flows.realnvp import RealNVP
from gwcm.training.trainer import train_model
from gwcm.visualization.distributions import (
    plot_complex_scatter,
    plot_sample_comparison,
)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed:
            Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device() -> torch.device:
    """Resolve the best available PyTorch device.

    Returns:
        CUDA device if available, then MPS, otherwise CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def save_metrics(metrics: dict, path: str | Path) -> None:
    """Save experiment metrics as JSON.

    Args:
        metrics:
            Metrics dictionary.
        path:
            Output JSON path.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)


def main() -> None:
    """Run the RealNVP Rayleigh fading experiment."""
    seed = 42
    set_seed(seed)

    device = resolve_device()

    results_dir = Path("results")
    figures_dir = results_dir / "figures"
    metrics_dir = results_dir / "metrics"
    checkpoints_dir = results_dir / "checkpoints"

    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # 1. Generate Rayleigh fading samples
    # ---------------------------------------------------------------------
    num_samples = 10_000
    sigma = 1.0

    reference_samples = generate_rayleigh_samples(
        num_samples=num_samples,
        sigma=sigma,
        seed=seed,
    )

    # ---------------------------------------------------------------------
    # 2. Create DataLoaders
    # ---------------------------------------------------------------------
    batch_size = 256

    loaders = create_dataloaders(
        samples=reference_samples,
        batch_size=batch_size,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        seed=seed,
        shuffle_train=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    # ---------------------------------------------------------------------
    # 3. Build RealNVP model
    # ---------------------------------------------------------------------
    model = RealNVP(
        input_dim=2,
        num_coupling_layers=6,
        hidden_dim=128,
        num_hidden_layers=2,
        scale_clamp=2.0,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=5e-4,
        weight_decay=0.0,
    )

    # ---------------------------------------------------------------------
    # 4. Train model
    # ---------------------------------------------------------------------
    checkpoint_path = checkpoints_dir / "realnvp_rayleigh_best.pt"

    history = train_model(
        model=model,
        train_loader=loaders.train,
        val_loader=loaders.val,
        optimizer=optimizer,
        device=device,
        epochs=100,
        grad_clip_norm=5.0,
        checkpoint_path=checkpoint_path,
        show_progress=True,
    )

    # ---------------------------------------------------------------------
    # 5. Evaluate likelihood on test set
    # ---------------------------------------------------------------------
    test_metrics = evaluate_likelihood(
        model=model,
        data_loader=loaders.test,
        device=device,
    )

    # ---------------------------------------------------------------------
    # 6. Generate samples from trained model
    # ---------------------------------------------------------------------
    num_generated_samples = 5_000

    generated_samples = generate_flow_samples(
        model=model,
        num_samples=num_generated_samples,
        device=device,
    )

    # ---------------------------------------------------------------------
    # 7. Compute statistics
    # ---------------------------------------------------------------------
    reference_stats = compute_sample_statistics(reference_samples)
    generated_stats = compute_sample_statistics(generated_samples)

    # ---------------------------------------------------------------------
    # 8. Save plots
    # ---------------------------------------------------------------------
    plot_complex_scatter(
        samples=reference_samples,
        title="Rayleigh Reference Samples",
        save_path=figures_dir / "rayleigh_reference_scatter.png",
    )

    plot_complex_scatter(
        samples=generated_samples,
        title="RealNVP Generated Rayleigh Samples",
        save_path=figures_dir / "realnvp_rayleigh_generated_scatter.png",
    )

    plot_sample_comparison(
        reference_samples=reference_samples,
        generated_samples=generated_samples,
        reference_label="Rayleigh Reference",
        generated_label="RealNVP Generated",
        title="RealNVP on Rayleigh Fading",
        save_path=figures_dir / "realnvp_rayleigh_comparison.png",
    )

    # ---------------------------------------------------------------------
    # 9. Save metrics
    # ---------------------------------------------------------------------
    metrics = {
        "experiment": {
            "name": "realnvp_rayleigh_baseline",
            "seed": seed,
            "device": str(device),
        },
        "channel": {
            "type": "rayleigh",
            "num_samples": num_samples,
            "sigma": sigma,
            "theoretical_average_power": 2.0 * sigma**2,
        },
        "data": {
            "batch_size": batch_size,
            "train_samples": len(loaders.train.dataset),
            "val_samples": len(loaders.val.dataset),
            "test_samples": len(loaders.test.dataset),
        },
        "model": {
            "type": "RealNVP",
            "input_dim": 2,
            "num_coupling_layers": 6,
            "hidden_dim": 128,
            "num_hidden_layers": 2,
            "scale_clamp": 2.0,
        },
        "training": {
            "epochs": 100,
            "learning_rate": 5e-4,
            "weight_decay": 0.0,
            "grad_clip_norm": 5.0,
            "final_train_loss": history.train_losses[-1],
            "final_val_loss": history.val_losses[-1],
            "best_val_loss": history.best_val_loss,
        },
        "test_likelihood": {
            "mean_log_likelihood": test_metrics.mean_log_likelihood,
            "mean_negative_log_likelihood": (
                test_metrics.mean_negative_log_likelihood
            ),
            "std_log_likelihood": test_metrics.std_log_likelihood,
            "num_test_samples": test_metrics.num_samples,
        },
        "reference_statistics": {
            "mean_real": reference_stats.mean_real,
            "mean_imag": reference_stats.mean_imag,
            "std_real": reference_stats.std_real,
            "std_imag": reference_stats.std_imag,
            "average_power": reference_stats.average_power,
            "num_samples": reference_stats.num_samples,
        },
        "generated_statistics": {
            "mean_real": generated_stats.mean_real,
            "mean_imag": generated_stats.mean_imag,
            "std_real": generated_stats.std_real,
            "std_imag": generated_stats.std_imag,
            "average_power": generated_stats.average_power,
            "num_samples": generated_stats.num_samples,
        },
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "reference_scatter": str(
                figures_dir / "rayleigh_reference_scatter.png"
            ),
            "generated_scatter": str(
                figures_dir / "realnvp_rayleigh_generated_scatter.png"
            ),
            "comparison_plot": str(
                figures_dir / "realnvp_rayleigh_comparison.png"
            ),
        },
    }

    save_metrics(
        metrics=metrics,
        path=metrics_dir / "realnvp_rayleigh_metrics.json",
    )

    # ---------------------------------------------------------------------
    # 10. Console summary
    # ---------------------------------------------------------------------
    print("\nExperiment completed.")
    print(f"Device: {device}")
    print(f"Best validation NLL: {history.best_val_loss:.4f}")
    print(
        "Test NLL: "
        f"{test_metrics.mean_negative_log_likelihood:.4f}"
    )
    print(
        "Reference average power: "
        f"{reference_stats.average_power:.4f}"
    )
    print(
        "Generated average power: "
        f"{generated_stats.average_power:.4f}"
    )
    print(f"Checkpoint saved to: {checkpoint_path}")
    print(
        "Metrics saved to: "
        f"{metrics_dir / 'realnvp_rayleigh_metrics.json'}"
    )


if __name__ == "__main__":
    main()