"""Tests for Rician fading channel simulator."""

from __future__ import annotations

import numpy as np
import pytest

from gwcm.channels.rician import (
    average_channel_power,
    db_to_linear,
    generate_rician_samples,
    rician_magnitude,
)


def test_db_to_linear() -> None:
    assert db_to_linear(0.0) == pytest.approx(1.0)
    assert db_to_linear(10.0) == pytest.approx(10.0)
    assert db_to_linear(20.0) == pytest.approx(100.0)


def test_rician_output_shape_real_format() -> None:
    samples = generate_rician_samples(
        num_samples=1000,
        k_factor=1.0,
        seed=42,
    )

    assert samples.shape == (1000, 2)
    assert samples.dtype == np.float32


def test_rician_output_shape_complex_format() -> None:
    samples = generate_rician_samples(
        num_samples=1000,
        k_factor=1.0,
        seed=42,
        return_complex=True,
    )

    assert samples.shape == (1000,)
    assert np.iscomplexobj(samples)


def test_rician_reproducibility() -> None:
    samples_1 = generate_rician_samples(
        num_samples=100,
        k_factor=5.0,
        seed=123,
    )

    samples_2 = generate_rician_samples(
        num_samples=100,
        k_factor=5.0,
        seed=123,
    )

    np.testing.assert_allclose(samples_1, samples_2)


def test_rician_magnitude_shape() -> None:
    samples = generate_rician_samples(
        num_samples=500,
        k_factor=3.0,
        seed=42,
    )

    magnitudes = rician_magnitude(samples)

    assert magnitudes.shape == (500,)
    assert np.all(magnitudes >= 0)


def test_rician_average_power_close_to_one() -> None:
    samples = generate_rician_samples(
        num_samples=100_000,
        k_factor=10.0,
        seed=42,
    )

    estimated_power = average_channel_power(samples)

    assert estimated_power == pytest.approx(1.0, rel=0.03)


def test_rician_k_factor_db_matches_linear() -> None:
    samples_linear = generate_rician_samples(
        num_samples=1000,
        k_factor=10.0,
        seed=42,
    )

    samples_db = generate_rician_samples(
        num_samples=1000,
        k_factor_db=10.0,
        seed=42,
    )

    np.testing.assert_allclose(samples_linear, samples_db)


def test_rician_high_k_has_strong_los_mean() -> None:
    samples = generate_rician_samples(
        num_samples=100_000,
        k_factor=100.0,
        los_phase=0.0,
        seed=42,
    )

    real_mean = samples[:, 0].mean()
    imag_mean = samples[:, 1].mean()

    expected_real_mean = np.sqrt(100.0 / 101.0)

    assert real_mean == pytest.approx(expected_real_mean, rel=0.02)
    assert imag_mean == pytest.approx(0.0, abs=0.02)


def test_invalid_num_samples_raises_error() -> None:
    with pytest.raises(ValueError, match="num_samples must be"):
        generate_rician_samples(num_samples=0)


def test_invalid_k_factor_raises_error() -> None:
    with pytest.raises(ValueError, match="K-factor must be"):
        generate_rician_samples(num_samples=100, k_factor=-1.0)