"""Loss functions for normalizing-flow training."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def negative_log_likelihood(model: nn.Module, batch: Tensor) -> Tensor:
    """Compute mean negative log-likelihood for a batch.

    The model is expected to implement:

        log_prob(batch) -> Tensor

    where the returned tensor has shape:

        (batch_size,)

    Args:
        model:
            Normalizing-flow model with a log_prob method.
        batch:
            Input batch with shape (batch_size, input_dim).

    Returns:
        Scalar tensor containing the mean negative log-likelihood.

    Raises:
        AttributeError:
            If the model does not implement log_prob.
        ValueError:
            If log_prob output shape is invalid.
    """
    if not hasattr(model, "log_prob"):
        raise AttributeError("model must implement a log_prob method.")

    log_prob = model.log_prob(batch)

    if log_prob.ndim != 1:
        raise ValueError("model.log_prob(batch) must return shape (batch_size,).")

    if log_prob.shape[0] != batch.shape[0]:
        raise ValueError("log_prob batch dimension must match input batch size.")

    return -torch.mean(log_prob)