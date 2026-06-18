"""Training utilities for normalizing-flow models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from gwcm.training.losses import negative_log_likelihood


@dataclass
class TrainingHistory:
    """Training history for a normalizing-flow experiment."""

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    best_val_loss: float = float("inf")


def move_batch_to_device(batch: Tensor, device: torch.device) -> Tensor:
    """Move one batch to the target device.

    Args:
        batch:
            Input batch.
        device:
            Target device.

    Returns:
        Batch on target device.
    """
    return batch.to(device)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader[Tensor],
    optimizer: Optimizer,
    device: torch.device,
    grad_clip_norm: float | None = None,
) -> float:
    """Train the model for one epoch.

    Args:
        model:
            Normalizing-flow model.
        train_loader:
            Training dataloader.
        optimizer:
            PyTorch optimizer.
        device:
            Training device.
        grad_clip_norm:
            Optional maximum gradient norm.

    Returns:
        Mean training loss for the epoch.

    Raises:
        ValueError:
            If grad_clip_norm is not positive.
    """
    if grad_clip_norm is not None and grad_clip_norm <= 0:
        raise ValueError("grad_clip_norm must be positive if provided.")

    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch in train_loader:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        loss = negative_log_likelihood(model, batch)

        loss.backward()

        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip_norm,
            )

        optimizer.step()

        batch_size = batch.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    data_loader: DataLoader[Tensor],
    device: torch.device,
) -> float:
    """Evaluate the model for one epoch.

    Args:
        model:
            Normalizing-flow model.
        data_loader:
            Validation or test dataloader.
        device:
            Evaluation device.

    Returns:
        Mean negative log-likelihood.
    """
    model.eval()

    total_loss = 0.0
    total_samples = 0

    for batch in data_loader:
        batch = move_batch_to_device(batch, device)

        loss = negative_log_likelihood(model, batch)

        batch_size = batch.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    val_loss: float,
    checkpoint_path: str | Path,
) -> None:
    """Save a model checkpoint.

    Args:
        model:
            Trained model.
        optimizer:
            Optimizer.
        epoch:
            Current epoch index.
        val_loss:
            Validation loss.
        checkpoint_path:
            Output checkpoint path.
    """
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        },
        path,
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader[Tensor],
    val_loader: DataLoader[Tensor],
    optimizer: Optimizer,
    device: torch.device | str,
    epochs: int = 100,
    grad_clip_norm: float | None = 5.0,
    checkpoint_path: str | Path | None = None,
    show_progress: bool = True,
) -> TrainingHistory:
    """Train a normalizing-flow model.

    Args:
        model:
            Normalizing-flow model.
        train_loader:
            Training dataloader.
        val_loader:
            Validation dataloader.
        optimizer:
            PyTorch optimizer.
        device:
            Training device.
        epochs:
            Number of training epochs.
        grad_clip_norm:
            Optional maximum gradient norm.
        checkpoint_path:
            Optional path for saving the best validation checkpoint.
        show_progress:
            If True, show tqdm progress bar.

    Returns:
        TrainingHistory object containing train and validation losses.

    Raises:
        ValueError:
            If epochs is not positive.
    """
    if epochs <= 0:
        raise ValueError("epochs must be positive.")

    device = torch.device(device)
    model.to(device)

    history = TrainingHistory()

    epoch_iterator = range(1, epochs + 1)

    if show_progress:
        epoch_iterator = tqdm(epoch_iterator, desc="Training", leave=True)

    for epoch in epoch_iterator:
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=grad_clip_norm,
        )

        val_loss = evaluate_one_epoch(
            model=model,
            data_loader=val_loader,
            device=device,
        )

        history.train_losses.append(train_loss)
        history.val_losses.append(val_loss)

        if val_loss < history.best_val_loss:
            history.best_val_loss = val_loss

            if checkpoint_path is not None:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    val_loss=val_loss,
                    checkpoint_path=checkpoint_path,
                )

        if show_progress and hasattr(epoch_iterator, "set_postfix"):
            epoch_iterator.set_postfix(
                train_loss=f"{train_loss:.4f}",
                val_loss=f"{val_loss:.4f}",
                best_val_loss=f"{history.best_val_loss:.4f}",
            )

    return history