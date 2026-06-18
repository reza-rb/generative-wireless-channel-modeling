"""Likelihood evaluation and sampling utilities for normalizing flows.

This module evaluates trained normalizing-flow models on wireless channel
samples represented as:

    x = [Re(h), Im(h)]

with shape:

    (num_samples, 2)

The model is expected to implement:

    log_prob(x) -> Tensor
    sample(num_samples, device=None) -> Tensor
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class LikelihoodMetrics:
    """Likelihood metrics computed over a dataset."""

    mean_log_likelihood: float
    mean_negative_log_likelihood: float
    std_log_likelihood: float
    num_samples: int


@dataclass(frozen=True)
class SampleStatistics:
    """Basic statistics for generated or reference channel samples."""

    mean_real: float
    mean_imag: float
    std_real: float
    std_imag: float
    average_power: float
    num_samples: int


def evaluate_likelihood(
    model: nn.Module,
    data_loader: DataLoader[Tensor],
    device: torch.device | str,
) -> LikelihoodMetrics:
    """Evaluate log-likelihood metrics on a dataloader.

    Args:
        model:
            Normalizing-flow model implementing log_prob(x).
        data_loader:
            Dataloader containing batches with shape (batch_size, input_dim).
        device:
            Evaluation device.

    Returns:
        LikelihoodMetrics object.

    Raises:
        AttributeError:
            If the model does not implement log_prob.
    """
    if not hasattr(model, "log_prob"):
        raise AttributeError("model must implement a log_prob method.")

    device = torch.device(device)
    model.to(device)
    model.eval()

    all_log_probs: list[Tensor] = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            log_prob = model.log_prob(batch)

            if log_prob.ndim != 1:
                raise ValueError(
                    "model.log_prob(batch) must return shape (batch_size,)."
                )

            all_log_probs.append(log_prob.detach().cpu())

    log_probs = torch.cat(all_log_probs, dim=0)

    mean_log_likelihood = float(log_probs.mean().item())
    std_log_likelihood = float(log_probs.std(unbiased=False).item())

    return LikelihoodMetrics(
        mean_log_likelihood=mean_log_likelihood,
        mean_negative_log_likelihood=-mean_log_likelihood,
        std_log_likelihood=std_log_likelihood,
        num_samples=int(log_probs.shape[0]),
    )


def generate_flow_samples(
    model: nn.Module,
    num_samples: int,
    device: torch.device | str,
) -> np.ndarray:
    """Generate samples from a trained flow model.

    Args:
        model:
            Normalizing-flow model implementing sample(num_samples, device).
        num_samples:
            Number of samples to generate.
        device:
            Sampling device.

    Returns:
        Generated samples as a NumPy array with shape (num_samples, input_dim).

    Raises:
        AttributeError:
            If the model does not implement sample.
        ValueError:
            If num_samples is not positive.
    """
    if not hasattr(model, "sample"):
        raise AttributeError("model must implement a sample method.")

    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    device = torch.device(device)
    model.to(device)
    model.eval()

    with torch.no_grad():
        samples = model.sample(num_samples=num_samples, device=device)

    return samples.detach().cpu().numpy().astype(np.float32)


def compute_sample_statistics(samples: np.ndarray | Tensor) -> SampleStatistics:
    """Compute basic statistics for channel samples.

    Args:
        samples:
            Real-valued channel samples with shape (num_samples, 2).

    Returns:
        SampleStatistics object.

    Raises:
        ValueError:
            If samples do not have shape (num_samples, 2).
    """
    if isinstance(samples, Tensor):
        samples_array = samples.detach().cpu().numpy()
    else:
        samples_array = np.asarray(samples)

    if samples_array.ndim != 2 or samples_array.shape[1] != 2:
        raise ValueError("samples must have shape (num_samples, 2).")

    if samples_array.shape[0] == 0:
        raise ValueError("samples must contain at least one sample.")

    real = samples_array[:, 0]
    imag = samples_array[:, 1]

    power = np.mean(real**2 + imag**2)

    return SampleStatistics(
        mean_real=float(np.mean(real)),
        mean_imag=float(np.mean(imag)),
        std_real=float(np.std(real)),
        std_imag=float(np.std(imag)),
        average_power=float(power),
        num_samples=int(samples_array.shape[0]),
    )


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device | str,
) -> dict:
    """Load model weights from a checkpoint.

    Args:
        model:
            Model instance with the same architecture as the checkpoint.
        checkpoint_path:
            Path to checkpoint file.
        device:
            Device used for loading.

    Returns:
        Loaded checkpoint dictionary.

    Raises:
        FileNotFoundError:
            If checkpoint file does not exist.
    """
    path = Path(checkpoint_path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    device = torch.device(device)

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    return checkpoint