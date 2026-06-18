"""Train RealNVP on Rayleigh, Rician, or multipath fading channels.

Run examples from the repository root:

    python3 scripts/train_realnvp_channel.py --channel rayleigh

    python3 scripts/train_realnvp_channel.py --channel rician --k-factor-db 10

    python3 scripts/train_realnvp_channel.py --channel multipath --num-paths 5 --decay-factor 1.0

This script runs a full experiment:

    channel simulation
        -> DataLoaders
        -> RealNVP training
        -> likelihood evaluation
        -> sample generation
        -> plots and metrics export
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gwcm.channels.multipath import generate_multipath_samples
from gwcm.channels.rayleigh import generate_rayleigh_samples
from gwcm.channels.rician import generate_rician_samples
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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train RealNVP on wireless channel samples."
    )

    parser.add_argument(
        "--channel",
        type=str,
        choices=["rayleigh", "rician", "multipath"],
        required=True,
        help="Wireless channel model to simulate.",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=10_000,
        help="Number of simulated channel samples.",
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Rayleigh Gaussian component standard deviation.",
    )

    parser.add_argument(
        "--k-factor-db",
        type=float,
        default=10.0,
        help="Rician K-factor in dB.",
    )

    parser.add_argument(
        "--los-phase",
        type=float,
        default=0.0,
        help="Rician line-of-sight phase in radians.",
    )

    parser.add_argument(
        "--num-paths",
        type=int,
        default=5,
        help="Number of multipath components.",
    )

    parser.add_argument(
        "--decay-factor",
        type=float,
        default=1.0,
        help="Multipath exponential power decay factor.",
    )

    parser.add_argument(
        "--multipath-los-real",
        type=float,
        default=None,
        help="Optional real part of multipath LOS component.",
    )

    parser.add_argument(
        "--multipath-los-imag",
        type=float,
        default=0.0,
        help="Optional imaginary part of multipath LOS component.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
        help="Adam learning rate.",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Adam weight decay.",
    )

    parser.add_argument(
        "--num-coupling-layers",
        type=int,
        default=6,
        help="Number of RealNVP coupling layers.",
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension of coupling networks.",
    )

    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        default=2,
        help="Number of hidden layers in each coupling network.",
    )

    parser.add_argument(
        "--scale-clamp",
        type=float,
        default=2.0,
        help="Scale clamp value for RealNVP.",
    )

    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=5.0,
        help="Gradient clipping norm.",
    )

    parser.add_argument(
        "--num-generated-samples",
        type=int,
        default=5_000,
        help="Number of generated samples after training.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Training device.",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(device_arg: str) -> torch.device:
    """Resolve training device from argument."""
    if device_arg == "cpu":
        return torch.device("cpu")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")

    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def make_experiment_name(args: argparse.Namespace) -> str:
    """Create a filesystem-safe experiment name."""
    if args.channel == "rayleigh":
        sigma_name = str(args.sigma).replace(".", "p")
        return f"realnvp_rayleigh_sigma{sigma_name}"

    if args.channel == "rician":
        k_name = str(args.k_factor_db).replace(".", "p").replace("-", "m")
        return f"realnvp_rician_k{k_name}db"

    decay_name = str(args.decay_factor).replace(".", "p")
    name = f"realnvp_multipath_l{args.num_paths}_decay{decay_name}"

    if args.multipath_los_real is not None:
        los_real = str(args.multipath_los_real).replace(".", "p").replace("-", "m")
        los_imag = str(args.multipath_los_imag).replace(".", "p").replace("-", "m")
        name = f"{name}_los{los_real}_{los_imag}"

    return name


def generate_channel_samples(args: argparse.Namespace) -> np.ndarray:
    """Generate samples for the selected channel model."""
    if args.channel == "rayleigh":
        return generate_rayleigh_samples(
            num_samples=args.num_samples,
            sigma=args.sigma,
            seed=args.seed,
        )

    if args.channel == "rician":
        return generate_rician_samples(
            num_samples=args.num_samples,
            k_factor_db=args.k_factor_db,
            los_phase=args.los_phase,
            seed=args.seed,
        )

    los_component = None

    if args.multipath_los_real is not None:
        los_component = complex(
            args.multipath_los_real,
            args.multipath_los_imag,
        )

    return generate_multipath_samples(
        num_samples=args.num_samples,
        num_paths=args.num_paths,
        decay_factor=args.decay_factor,
        los_component=los_component,
        normalize_power=True,
        seed=args.seed,
    )


def save_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    """Save experiment metrics as JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)


def main() -> None:
    """Run the selected RealNVP channel modeling experiment."""
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    experiment_name = make_experiment_name(args)

    results_dir = Path("results")
    figures_dir = results_dir / "figures"
    metrics_dir = results_dir / "metrics"
    checkpoints_dir = results_dir / "checkpoints"

    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    reference_samples = generate_channel_samples(args)

    loaders = create_dataloaders(
        samples=reference_samples,
        batch_size=args.batch_size,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        seed=args.seed,
        shuffle_train=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = RealNVP(
        input_dim=2,
        num_coupling_layers=args.num_coupling_layers,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
        scale_clamp=args.scale_clamp,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    checkpoint_path = checkpoints_dir / f"{experiment_name}_best.pt"

    history = train_model(
        model=model,
        train_loader=loaders.train,
        val_loader=loaders.val,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        grad_clip_norm=args.grad_clip_norm,
        checkpoint_path=checkpoint_path,
        show_progress=True,
    )

    test_metrics = evaluate_likelihood(
        model=model,
        data_loader=loaders.test,
        device=device,
    )

    generated_samples = generate_flow_samples(
        model=model,
        num_samples=args.num_generated_samples,
        device=device,
    )

    reference_stats = compute_sample_statistics(reference_samples)
    generated_stats = compute_sample_statistics(generated_samples)

    reference_scatter_path = figures_dir / f"{experiment_name}_reference_scatter.png"
    generated_scatter_path = figures_dir / f"{experiment_name}_generated_scatter.png"
    comparison_path = figures_dir / f"{experiment_name}_comparison.png"

    plot_complex_scatter(
        samples=reference_samples,
        title=f"{experiment_name}: Reference Samples",
        save_path=reference_scatter_path,
    )

    plot_complex_scatter(
        samples=generated_samples,
        title=f"{experiment_name}: Generated Samples",
        save_path=generated_scatter_path,
    )

    plot_sample_comparison(
        reference_samples=reference_samples,
        generated_samples=generated_samples,
        reference_label="Reference",
        generated_label="RealNVP Generated",
        title=f"{experiment_name}: Reference vs RealNVP",
        save_path=comparison_path,
    )

    metrics = {
        "experiment": {
            "name": experiment_name,
            "seed": args.seed,
            "device": str(device),
        },
        "channel": {
            "type": args.channel,
            "num_samples": args.num_samples,
            "sigma": args.sigma if args.channel == "rayleigh" else None,
            "k_factor_db": args.k_factor_db if args.channel == "rician" else None,
            "los_phase": args.los_phase if args.channel == "rician" else None,
            "num_paths": args.num_paths if args.channel == "multipath" else None,
            "decay_factor": args.decay_factor if args.channel == "multipath" else None,
            "multipath_los_real": args.multipath_los_real,
            "multipath_los_imag": (
                args.multipath_los_imag
                if args.multipath_los_real is not None
                else None
            ),
        },
        "data": {
            "batch_size": args.batch_size,
            "train_samples": len(loaders.train.dataset),
            "val_samples": len(loaders.val.dataset),
            "test_samples": len(loaders.test.dataset),
        },
        "model": {
            "type": "RealNVP",
            "input_dim": 2,
            "num_coupling_layers": args.num_coupling_layers,
            "hidden_dim": args.hidden_dim,
            "num_hidden_layers": args.num_hidden_layers,
            "scale_clamp": args.scale_clamp,
        },
        "training": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "grad_clip_norm": args.grad_clip_norm,
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
            "reference_scatter": str(reference_scatter_path),
            "generated_scatter": str(generated_scatter_path),
            "comparison_plot": str(comparison_path),
            "metrics": str(metrics_dir / f"{experiment_name}_metrics.json"),
        },
    }

    metrics_path = metrics_dir / f"{experiment_name}_metrics.json"
    save_metrics(metrics=metrics, path=metrics_path)

    print("\nExperiment completed.")
    print(f"Experiment: {experiment_name}")
    print(f"Device: {device}")
    print(f"Best validation NLL: {history.best_val_loss:.4f}")
    print(f"Test NLL: {test_metrics.mean_negative_log_likelihood:.4f}")
    print(f"Reference average power: {reference_stats.average_power:.4f}")
    print(f"Generated average power: {generated_stats.average_power:.4f}")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()