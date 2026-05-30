"""Rician fading channel simulator.

This module generates complex-valued Rician fading channel coefficients and
represents them as real-valued vectors for machine learning models.

A Rician fading channel contains a deterministic line-of-sight component and
a random non-line-of-sight scattering component:

    h = sqrt(K / (K + 1)) * h_los
        + sqrt(1 / (K + 1)) * h_nlos

where K is the linear Rician K-factor.
"""

from __future__ import annotations

import numpy as np


def db_to_linear(value_db: float) -> float:
    """Convert a decibel value to linear scale.

    Args:
        value_db:
            Value in decibels.

    Returns:
        Linear-scale value.
    """
    return float(10.0 ** (value_db / 10.0))


def generate_rician_samples(
    num_samples: int,
    k_factor: float = 1.0,
    k_factor_db: float | None = None,
    los_phase: float = 0.0,
    seed: int | None = None,
    return_complex: bool = False,
) -> np.ndarray:
    """Generate normalized Rician fading channel samples.

    The normalized Rician channel model is:

        h = sqrt(K / (K + 1)) * h_los
            + sqrt(1 / (K + 1)) * h_nlos

    where:

        h_los = exp(j * los_phase)

    and:

        h_nlos = (n_real + j * n_imag) / sqrt(2)

    with:

        n_real, n_imag ~ N(0, 1)

    This normalization gives approximately:

        E[|h|^2] = 1

    Args:
        num_samples:
            Number of independent channel samples to generate.
        k_factor:
            Linear Rician K-factor. Ignored if k_factor_db is provided.
            Must be non-negative.
        k_factor_db:
            Optional Rician K-factor in decibels. If provided, this is
            converted to linear scale and used instead of k_factor.
        los_phase:
            Phase of the deterministic line-of-sight component in radians.
        seed:
            Optional random seed for reproducibility.
        return_complex:
            If True, return a complex-valued NumPy array with shape
            (num_samples,). If False, return a real-valued NumPy array with
            shape (num_samples, 2).

    Returns:
        Generated Rician fading samples.

        If return_complex is False:
            Array with shape (num_samples, 2), where column 0 is Re(h)
            and column 1 is Im(h).

        If return_complex is True:
            Complex-valued array with shape (num_samples,).

    Raises:
        ValueError:
            If num_samples is not positive or the K-factor is negative.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be a positive integer.")

    if k_factor_db is not None:
        k_factor = db_to_linear(k_factor_db)

    if k_factor < 0:
        raise ValueError("Rician K-factor must be non-negative.")

    rng = np.random.default_rng(seed)

    h_los = np.exp(1j * los_phase)

    n_real = rng.normal(loc=0.0, scale=1.0, size=num_samples)
    n_imag = rng.normal(loc=0.0, scale=1.0, size=num_samples)

    h_nlos = (n_real + 1j * n_imag) / np.sqrt(2.0)

    los_scale = np.sqrt(k_factor / (k_factor + 1.0))
    nlos_scale = np.sqrt(1.0 / (k_factor + 1.0))

    h = los_scale * h_los + nlos_scale * h_nlos

    if return_complex:
        return h

    samples = np.stack([h.real, h.imag], axis=1)

    return samples.astype(np.float32)


def rician_magnitude(samples: np.ndarray) -> np.ndarray:
    """Compute the magnitude of Rician fading samples.

    Args:
        samples:
            Either a complex-valued array with shape (num_samples,) or a
            real-valued array with shape (num_samples, 2).

    Returns:
        Magnitudes with shape (num_samples,).

    Raises:
        ValueError:
            If the real-valued input shape is invalid.
    """
    if np.iscomplexobj(samples):
        return np.abs(samples)

    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError(
            "Real-valued samples must have shape (num_samples, 2)."
        )

    return np.sqrt(samples[:, 0] ** 2 + samples[:, 1] ** 2)


def average_channel_power(samples: np.ndarray) -> float:
    """Compute the average channel power E[|h|^2].

    Args:
        samples:
            Either a complex-valued array with shape (num_samples,) or a
            real-valued array with shape (num_samples, 2).

    Returns:
        Estimated average channel power.
    """
    magnitudes = rician_magnitude(samples)
    power = np.mean(magnitudes**2)

    return float(power)