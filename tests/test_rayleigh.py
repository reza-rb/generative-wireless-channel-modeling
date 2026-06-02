"""Tests for Rayleigh fading channel simulator."""

from __future__ import annotations

import numpy as np
import pytest

from gwcm.channels.rayleigh import (
    average_channel_power,
    generate_rayleigh_samples,
    rayleigh_magnitude,
)


def test_rayleigh_output_shape_real_format() -> None:
    samples = generate_rayleigh_samples(num_samples=1000, sigma=1.0, seed=42)

    assert samples.shape == (1000, 2)
    assert samples.dtype == np.float32


def test_rayleigh_output_shape_complex_format() -> None:
    samples = generate_rayleigh_samples(
        num_samples=1000,
        sigma=1.0,
        seed=42,
        return_complex=True,
    )

    assert samples.shape == (1000,)
    assert np.iscomplexobj(samples)


def test_rayleigh_reproducibility() -> None:
    samples_1 = generate_rayleigh_samples(num_samples=100, sigma=1.0, seed=123)
    samples_2 = generate_rayleigh_samples(num_samples=100, sigma=1.0, seed=123)

    np.testing.assert_allclose(samples_1, samples_2)


def test_rayleigh_magnitude_shape() -> None:
    samples = generate_rayleigh_samples(num_samples=500, sigma=1.0, seed=42)
    magnitudes = rayleigh_magnitude(samples)

    assert magnitudes.shape == (500,)
    assert np.all(magnitudes >= 0)


def test_rayleigh_average_power_close_to_theory() -> None:
    sigma = 1.0
    samples = generate_rayleigh_samples(
        num_samples=100_000,
        sigma=sigma,
        seed=42,
    )

    estimated_power = average_channel_power(samples)
    theoretical_power = 2.0 * sigma**2

    assert estimated_power == pytest.approx(theoretical_power, rel=0.03)


def test_invalid_num_samples_raises_error() -> None:
    with pytest.raises(ValueError, match="num_samples must be"):
        generate_rayleigh_samples(num_samples=0)


def test_invalid_sigma_raises_error() -> None:
    with pytest.raises(ValueError, match="sigma must be"):
        generate_rayleigh_samples(num_samples=100, sigma=0.0) 