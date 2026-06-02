"""Tests for wireless channel visualization utilities."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from gwcm.channels.rayleigh import generate_rayleigh_samples
from gwcm.visualization.distributions import (
    compute_magnitude,
    plot_complex_scatter,
    plot_magnitude_histogram,
    plot_real_imag_histograms,
    plot_sample_comparison,
)


def test_compute_magnitude_shape() -> None:
    samples = np.array(
        [
            [3.0, 4.0],
            [5.0, 12.0],
        ],
        dtype=np.float32,
    )

    magnitudes = compute_magnitude(samples)

    assert magnitudes.shape == (2,)
    np.testing.assert_allclose(magnitudes, np.array([5.0, 13.0]))


def test_compute_magnitude_rejects_invalid_shape() -> None:
    samples = np.random.randn(100).astype(np.float32)

    with pytest.raises(ValueError, match="samples must have shape"):
        compute_magnitude(samples)


def test_plot_complex_scatter_returns_figure() -> None:
    samples = generate_rayleigh_samples(
        num_samples=100,
        sigma=1.0,
        seed=42,
    )

    fig = plot_complex_scatter(samples)

    assert isinstance(fig, Figure)


def test_plot_real_imag_histograms_returns_figure() -> None:
    samples = generate_rayleigh_samples(
        num_samples=100,
        sigma=1.0,
        seed=42,
    )

    fig = plot_real_imag_histograms(samples)

    assert isinstance(fig, Figure)


def test_plot_magnitude_histogram_returns_figure() -> None:
    samples = generate_rayleigh_samples(
        num_samples=100,
        sigma=1.0,
        seed=42,
    )

    fig = plot_magnitude_histogram(samples)

    assert isinstance(fig, Figure)


def test_plot_sample_comparison_returns_figure() -> None:
    reference_samples = generate_rayleigh_samples(
        num_samples=100,
        sigma=1.0,
        seed=42,
    )

    generated_samples = generate_rayleigh_samples(
        num_samples=100,
        sigma=1.0,
        seed=123,
    )

    fig = plot_sample_comparison(
        reference_samples=reference_samples,
        generated_samples=generated_samples,
    )

    assert isinstance(fig, Figure)


def test_plot_complex_scatter_saves_file(tmp_path) -> None:
    samples = generate_rayleigh_samples(
        num_samples=100,
        sigma=1.0,
        seed=42,
    )

    save_path = tmp_path / "rayleigh_scatter.png"

    _ = plot_complex_scatter(samples=samples, save_path=save_path)

    assert save_path.exists()


def test_invalid_max_points_raises_error() -> None:
    samples = generate_rayleigh_samples(
        num_samples=100,
        sigma=1.0,
        seed=42,
    )

    with pytest.raises(ValueError, match="max_points must be"):
        plot_complex_scatter(samples=samples, max_points=0)


def test_invalid_bins_raises_error() -> None:
    samples = generate_rayleigh_samples(
        num_samples=100,
        sigma=1.0,
        seed=42,
    )

    with pytest.raises(ValueError, match="bins must be"):
        plot_magnitude_histogram(samples=samples, bins=0)
        