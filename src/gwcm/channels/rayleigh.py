"""Rayleigh fading channel simulator.

This module provides utilities for generating complex-valued Rayleigh fading
channel coefficients and representing them as real-valued vectors suitable
for machine learning models.

A complex channel coefficient is represented as:

    h = h_real + j * h_imag

For PyTorch models, we convert this to:

    x = [Re(h), Im(h)]

with shape:

    (num_samples, 2)
"""

from __future__ import annotations

import numpy as np


def generate_rayleigh_samples(
    num_samples: int,
    sigma: float = 1.0,
    seed: int | None = None,
    return_complex: bool = False,
) -> np.ndarray:
    """Generate Rayleigh fading channel samples.

    Rayleigh fading assumes no dominant line-of-sight component. The real and
    imaginary parts of the complex channel coefficient are modeled as
    independent zero-mean Gaussian random variables:

        h_real ~ N(0, sigma^2)
        h_imag ~ N(0, sigma^2)

    The complex channel is:

        h = h_real + j * h_imag

    By default, the output is represented as a real-valued array:

        x = [Re(h), Im(h)]

    Args:
        num_samples:
            Number of independent channel samples to generate.
        sigma:
            Standard deviation of the real and imaginary Gaussian components.
            Must be positive.
        seed:
            Optional random seed for reproducibility.
        return_complex:
            If True, return a complex-valued NumPy array with shape
            (num_samples,). If False, return a real-valued NumPy array with
            shape (num_samples, 2).

    Returns:
        Generated Rayleigh fading samples.

        If return_complex is False:
            Array with shape (num_samples, 2), where column 0 is Re(h)
            and column 1 is Im(h).

        If return_complex is True:
            Complex-valued array with shape (num_samples,).

    Raises:
        ValueError:
            If num_samples is not positive or sigma is not positive.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be a positive integer.")

    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    rng = np.random.default_rng(seed)

    h_real = rng.normal(loc=0.0, scale=sigma, size=num_samples)
    h_imag = rng.normal(loc=0.0, scale=sigma, size=num_samples)

    if return_complex:
        return h_real + 1j * h_imag

    samples = np.stack([h_real, h_imag], axis=1)

    return samples.astype(np.float32)


def rayleigh_magnitude(samples: np.ndarray) -> np.ndarray:
    """Compute the magnitude of Rayleigh fading samples.

    Args:
        samples:
            Either a complex-valued array with shape (num_samples,) or a
            real-valued array with shape (num_samples, 2), where the two
            columns represent real and imaginary parts.

    Returns:
        Magnitudes of the complex channel coefficients with shape
        (num_samples,).

    Raises:
        ValueError:
            If the input shape is invalid.
    """
    if np.iscomplexobj(samples):
        return np.abs(samples)

    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError(
            "Real-valued samples must have shape (num_samples, 2)."
        )

    h_real = samples[:, 0]
    h_imag = samples[:, 1]

    return np.sqrt(h_real**2 + h_imag**2)


def average_channel_power(samples: np.ndarray) -> float:
    """Compute the average channel power E[|h|^2].

    For Rayleigh fading with:

        h_real ~ N(0, sigma^2)
        h_imag ~ N(0, sigma^2)

    the theoretical average power is:

        E[|h|^2] = 2 * sigma^2

    Args:
        samples:
            Either a complex-valued array with shape (num_samples,) or a
            real-valued array with shape (num_samples, 2).

    Returns:
        Estimated average channel power.
    """
    magnitudes = rayleigh_magnitude(samples)
    power = np.mean(magnitudes**2)

    return float(power)