"""Tests for normalizing-flow training utilities."""

from __future__ import annotations

import torch
import pytest

from gwcm.channels.rayleigh import generate_rayleigh_samples
from gwcm.data.channel_dataset import create_dataloaders
from gwcm.models.flows.realnvp import RealNVP
from gwcm.training.losses import negative_log_likelihood
from gwcm.training.trainer import (
    evaluate_one_epoch,
    train_model,
    train_one_epoch,
)


def test_negative_log_likelihood_returns_scalar() -> None:
    model = RealNVP(
        input_dim=2,
        num_coupling_layers=2,
        hidden_dim=16,
        num_hidden_layers=1,
    )

    batch = torch.randn(32, 2)

    loss = negative_log_likelihood(model, batch)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_train_one_epoch_returns_finite_loss() -> None:
    samples = generate_rayleigh_samples(
        num_samples=512,
        sigma=1.0,
        seed=42,
    )

    loaders = create_dataloaders(
        samples=samples,
        batch_size=64,
        seed=42,
    )

    model = RealNVP(
        input_dim=2,
        num_coupling_layers=2,
        hidden_dim=16,
        num_hidden_layers=1,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss = train_one_epoch(
        model=model,
        train_loader=loaders.train,
        optimizer=optimizer,
        device=torch.device("cpu"),
        grad_clip_norm=5.0,
    )

    assert isinstance(loss, float)
    assert loss > 0
    assert torch.isfinite(torch.tensor(loss))


def test_evaluate_one_epoch_returns_finite_loss() -> None:
    samples = generate_rayleigh_samples(
        num_samples=512,
        sigma=1.0,
        seed=42,
    )

    loaders = create_dataloaders(
        samples=samples,
        batch_size=64,
        seed=42,
    )

    model = RealNVP(
        input_dim=2,
        num_coupling_layers=2,
        hidden_dim=16,
        num_hidden_layers=1,
    )

    loss = evaluate_one_epoch(
        model=model,
        data_loader=loaders.val,
        device=torch.device("cpu"),
    )

    assert isinstance(loss, float)
    assert loss > 0
    assert torch.isfinite(torch.tensor(loss))


def test_train_model_history_lengths() -> None:
    samples = generate_rayleigh_samples(
        num_samples=512,
        sigma=1.0,
        seed=42,
    )

    loaders = create_dataloaders(
        samples=samples,
        batch_size=64,
        seed=42,
    )

    model = RealNVP(
        input_dim=2,
        num_coupling_layers=2,
        hidden_dim=16,
        num_hidden_layers=1,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    history = train_model(
        model=model,
        train_loader=loaders.train,
        val_loader=loaders.val,
        optimizer=optimizer,
        device="cpu",
        epochs=3,
        grad_clip_norm=5.0,
        show_progress=False,
    )

    assert len(history.train_losses) == 3
    assert len(history.val_losses) == 3
    assert history.best_val_loss == min(history.val_losses)


def test_train_model_saves_checkpoint(tmp_path) -> None:
    samples = generate_rayleigh_samples(
        num_samples=512,
        sigma=1.0,
        seed=42,
    )

    loaders = create_dataloaders(
        samples=samples,
        batch_size=64,
        seed=42,
    )

    model = RealNVP(
        input_dim=2,
        num_coupling_layers=2,
        hidden_dim=16,
        num_hidden_layers=1,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    checkpoint_path = tmp_path / "best_model.pt"

    _ = train_model(
        model=model,
        train_loader=loaders.train,
        val_loader=loaders.val,
        optimizer=optimizer,
        device="cpu",
        epochs=2,
        checkpoint_path=checkpoint_path,
        show_progress=False,
    )

    assert checkpoint_path.exists()


def test_train_model_rejects_invalid_epochs() -> None:
    samples = generate_rayleigh_samples(
        num_samples=512,
        sigma=1.0,
        seed=42,
    )

    loaders = create_dataloaders(
        samples=samples,
        batch_size=64,
        seed=42,
    )

    model = RealNVP(input_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    with pytest.raises(ValueError, match="epochs must be"):
        train_model(
            model=model,
            train_loader=loaders.train,
            val_loader=loaders.val,
            optimizer=optimizer,
            device="cpu",
            epochs=0,
            show_progress=False,
        )


def test_train_one_epoch_rejects_invalid_grad_clip() -> None:
    samples = generate_rayleigh_samples(
        num_samples=512,
        sigma=1.0,
        seed=42,
    )

    loaders = create_dataloaders(
        samples=samples,
        batch_size=64,
        seed=42,
    )

    model = RealNVP(input_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    with pytest.raises(ValueError, match="grad_clip_norm must be"):
        train_one_epoch(
            model=model,
            train_loader=loaders.train,
            optimizer=optimizer,
            device=torch.device("cpu"),
            grad_clip_norm=0.0,
        )