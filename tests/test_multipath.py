"""Tests for multipath fading channel simulator."""

from __future__ import annotations

import numpy as np
import pytest

from gwcm.channels.multipath import (
    average_channel_power,
    exponential_power_profile,
    generate_multipath_samples,
    multipath_magnitude,
)


def test_exponential_power_profile_shape() -> None:
    powers = exponential_power_profile(num_paths=5, decay_factor=1.0)

    assert powers.shape == (5,)


def test_exponential_power_profile_normalized() -> None:
    powers = exponential_power_profile(
        num_paths=5,
        decay_factor=1.0,
        normalize=True,
    )

    assert np.sum(powers) == pytest.approx(1.0)


def test_exponential_power_profile_is_decreasing() -> None:
    powers = exponential_power_profile(num_paths=5, decay_factor=1.0)

    assert np.all(powers[:-1] >= powers[1:])


def test_multipath_output_shape_real_format() -> None:
    samples = generate_multipath_samples(
        num_samples=1000,
        num_paths=5,
        decay_factor=1.0,
        seed=42,
    )

    assert samples.shape == (1000, 2)
    assert samples.dtype == np.float32


def test_multipath_output_shape_complex_format() -> None:
    samples = generate_multipath_samples(
        num_samples=1000,
        num_paths=5,
        decay_factor=1.0,
        seed=42,
        return_complex=True,
    )

    assert samples.shape == (1000,)
    assert np.iscomplexobj(samples)


def test_multipath_reproducibility() -> None:
    samples_1 = generate_multipath_samples(
        num_samples=100,
        num_paths=5,
        decay_factor=1.0,
        seed=123,
    )

    samples_2 = generate_multipath_samples(
        num_samples=100,
        num_paths=5,
        decay_factor=1.0,
        seed=123,
    )

    np.testing.assert_allclose(samples_1, samples_2)


def test_multipath_magnitude_shape() -> None:
    samples = generate_multipath_samples(
        num_samples=500,
        num_paths=5,
        decay_factor=1.0,
        seed=42,
    )

    magnitudes = multipath_magnitude(samples)

    assert magnitudes.shape == (500,)
    assert np.all(magnitudes >= 0)


def test_multipath_average_power_close_to_one_when_normalized() -> None:
    samples = generate_multipath_samples(
        num_samples=100_000,
        num_paths=8,
        decay_factor=2.0,
        normalize_power=True,
        seed=42,
    )

    estimated_power = average_channel_power(samples)

    assert estimated_power == pytest.approx(1.0, rel=0.02)


def test_multipath_with_los_component_has_nonzero_mean() -> None:
    samples = generate_multipath_samples(
        num_samples=100_000,
        num_paths=5,
        decay_factor=1.0,
        los_component=1.0 + 0.0j,
        normalize_power=False,
        seed=42,
    )

    real_mean = samples[:, 0].mean()
    imag_mean = samples[:, 1].mean()

    assert real_mean == pytest.approx(1.0, rel=0.03)
    assert imag_mean == pytest.approx(0.0, abs=0.03)


def test_invalid_num_samples_raises_error() -> None:
    with pytest.raises(ValueError, match="num_samples must be"):
        generate_multipath_samples(num_samples=0)


def test_invalid_num_paths_raises_error() -> None:
    with pytest.raises(ValueError, match="num_paths must be"):
        generate_multipath_samples(num_samples=100, num_paths=0)


def test_invalid_decay_factor_raises_error() -> None:
    with pytest.raises(ValueError, match="decay_factor must be"):
        generate_multipath_samples(
            num_samples=100,
            num_paths=5,
            decay_factor=0.0,
        )