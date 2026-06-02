"""Visualization utilities for wireless channel distributions.

This module provides reusable plotting functions for complex-valued wireless
channel samples represented as real-valued vectors:

    x = [Re(h), Im(h)]

with shape:

    (num_samples, 2)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def _validate_samples(samples: np.ndarray) -> np.ndarray:
    """Validate and convert samples to NumPy array.

    Args:
        samples:
            Real-valued channel samples with shape (num_samples, 2).

    Returns:
        Validated NumPy array.

    Raises:
        ValueError:
            If samples do not have shape (num_samples, 2).
    """
    samples = np.asarray(samples)

    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError("samples must have shape (num_samples, 2).")

    if samples.shape[0] == 0:
        raise ValueError("samples must contain at least one sample.")

    return samples


def _save_figure(fig: Figure, save_path: str | Path | None) -> None:
    """Save figure if a save path is provided.

    Args:
        fig:
            Matplotlib figure.
        save_path:
            Optional file path.
    """
    if save_path is None:
        return

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=300)


def compute_magnitude(samples: np.ndarray) -> np.ndarray:
    """Compute channel magnitudes from real-valued samples.

    Args:
        samples:
            Real-valued channel samples with shape (num_samples, 2).

    Returns:
        Magnitudes with shape (num_samples,).
    """
    samples = _validate_samples(samples)

    real = samples[:, 0]
    imag = samples[:, 1]

    return np.sqrt(real**2 + imag**2)


def plot_complex_scatter(
    samples: np.ndarray,
    title: str = "Complex Channel Samples",
    max_points: int = 5000,
    alpha: float = 0.5,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot channel samples in the complex plane.

    Args:
        samples:
            Real-valued channel samples with shape (num_samples, 2).
        title:
            Plot title.
        max_points:
            Maximum number of points to display.
        alpha:
            Marker transparency.
        save_path:
            Optional path for saving the figure.

    Returns:
        Matplotlib figure.
    """
    samples = _validate_samples(samples)

    if max_points <= 0:
        raise ValueError("max_points must be positive.")

    displayed_samples = samples[:max_points]

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(
        displayed_samples[:, 0],
        displayed_samples[:, 1],
        s=8,
        alpha=alpha,
    )

    ax.axhline(0.0, linewidth=1.0)
    ax.axvline(0.0, linewidth=1.0)

    ax.set_title(title)
    ax.set_xlabel("Re(h)")
    ax.set_ylabel("Im(h)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    _save_figure(fig, save_path)

    return fig


def plot_real_imag_histograms(
    samples: np.ndarray,
    title: str = "Real and Imaginary Components",
    bins: int = 80,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot histograms of real and imaginary channel components.

    Args:
        samples:
            Real-valued channel samples with shape (num_samples, 2).
        title:
            Figure title.
        bins:
            Number of histogram bins.
        save_path:
            Optional path for saving the figure.

    Returns:
        Matplotlib figure.
    """
    samples = _validate_samples(samples)

    if bins <= 0:
        raise ValueError("bins must be positive.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(samples[:, 0], bins=bins, density=True, alpha=0.7)
    axes[0].set_title("Real Part")
    axes[0].set_xlabel("Re(h)")
    axes[0].set_ylabel("Density")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(samples[:, 1], bins=bins, density=True, alpha=0.7)
    axes[1].set_title("Imaginary Part")
    axes[1].set_xlabel("Im(h)")
    axes[1].set_ylabel("Density")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()

    _save_figure(fig, save_path)

    return fig


def plot_magnitude_histogram(
    samples: np.ndarray,
    title: str = "Channel Magnitude Distribution",
    bins: int = 80,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot histogram of channel magnitudes.

    Args:
        samples:
            Real-valued channel samples with shape (num_samples, 2).
        title:
            Plot title.
        bins:
            Number of histogram bins.
        save_path:
            Optional path for saving the figure.

    Returns:
        Matplotlib figure.
    """
    samples = _validate_samples(samples)

    if bins <= 0:
        raise ValueError("bins must be positive.")

    magnitudes = compute_magnitude(samples)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(magnitudes, bins=bins, density=True, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("|h|")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)

    _save_figure(fig, save_path)

    return fig


def plot_sample_comparison(
    reference_samples: np.ndarray,
    generated_samples: np.ndarray,
    reference_label: str = "Reference",
    generated_label: str = "Generated",
    title: str = "Reference vs Generated Samples",
    max_points: int = 5000,
    save_path: str | Path | None = None,
) -> Figure:
    """Compare reference and generated channel samples in the complex plane.

    This function will be useful after training normalizing flows.

    Args:
        reference_samples:
            True/simulated samples with shape (num_samples, 2).
        generated_samples:
            Model-generated samples with shape (num_samples, 2).
        reference_label:
            Label for reference samples.
        generated_label:
            Label for generated samples.
        title:
            Figure title.
        max_points:
            Maximum number of points to plot from each set.
        save_path:
            Optional path for saving the figure.

    Returns:
        Matplotlib figure.
    """
    reference_samples = _validate_samples(reference_samples)
    generated_samples = _validate_samples(generated_samples)

    if max_points <= 0:
        raise ValueError("max_points must be positive.")

    reference_display = reference_samples[:max_points]
    generated_display = generated_samples[:max_points]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    axes[0].scatter(
        reference_display[:, 0],
        reference_display[:, 1],
        s=8,
        alpha=0.5,
    )
    axes[0].set_title(reference_label)
    axes[0].set_xlabel("Re(h)")
    axes[0].set_ylabel("Im(h)")
    axes[0].axhline(0.0, linewidth=1.0)
    axes[0].axvline(0.0, linewidth=1.0)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect("equal", adjustable="box")

    axes[1].scatter(
        generated_display[:, 0],
        generated_display[:, 1],
        s=8,
        alpha=0.5,
    )
    axes[1].set_title(generated_label)
    axes[1].set_xlabel("Re(h)")
    axes[1].set_ylabel("Im(h)")
    axes[1].axhline(0.0, linewidth=1.0)
    axes[1].axvline(0.0, linewidth=1.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect("equal", adjustable="box")

    fig.suptitle(title)
    fig.tight_layout()

    _save_figure(fig, save_path)

    return fig
