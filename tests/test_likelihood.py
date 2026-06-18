"""Tests for likelihood evaluation and flow sampling utilities."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gwcm.channels.rayleigh import generate_rayleigh_samples
from gwcm.data.channel_dataset import create_dataloaders
from gwcm.evaluation.likelihood import (
    LikelihoodMetrics,
    SampleStatistics,
    compute_sample_statistics,
    evaluate_likelihood,
    generate_flow_samples,
    load_model_checkpoint,
)
from gwcm.models.flows.realnvp import RealNVP


def test_evaluate_likelihood_returns_metrics() -> None:
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

    metrics = evaluate_likelihood(
        model=model,
        data_loader=loaders.test,
        device="cpu",
    )

    assert isinstance(metrics, LikelihoodMetrics)
    assert metrics.num_samples == len(loaders.test.dataset)
    assert np.isfinite(metrics.mean_log_likelihood)
    assert np.isfinite(metrics.mean_negative_log_likelihood)
    assert np.isfinite(metrics.std_log_likelihood)


def test_generate_flow_samples_shape() -> None:
    model = RealNVP(
        input_dim=2,
        num_coupling_layers=2,
        hidden_dim=16,
        num_hidden_layers=1,
    )

    samples = generate_flow_samples(
        model=model,
        num_samples=100,
        device="cpu",
    )

    assert samples.shape == (100, 2)
    assert samples.dtype == np.float32
    assert np.isfinite(samples).all()


def test_compute_sample_statistics_from_numpy() -> None:
    samples = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ],
        dtype=np.float32,
    )

    stats = compute_sample_statistics(samples)

    assert isinstance(stats, SampleStatistics)
    assert stats.num_samples == 4
    assert stats.mean_real == pytest.approx(0.0)
    assert stats.mean_imag == pytest.approx(0.0)
    assert stats.average_power == pytest.approx(1.0)


def test_compute_sample_statistics_from_tensor() -> None:
    samples = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=torch.float32,
    )

    stats = compute_sample_statistics(samples)

    assert stats.num_samples == 2
    assert stats.mean_real == pytest.approx(2.0)
    assert stats.mean_imag == pytest.approx(3.0)


def test_compute_sample_statistics_rejects_invalid_shape() -> None:
    samples = np.random.randn(100).astype(np.float32)

    with pytest.raises(ValueError, match="samples must have shape"):
        compute_sample_statistics(samples)


def test_generate_flow_samples_rejects_invalid_num_samples() -> None:
    model = RealNVP(input_dim=2)

    with pytest.raises(ValueError, match="num_samples must be"):
        generate_flow_samples(
            model=model,
            num_samples=0,
            device="cpu",
        )


def test_load_model_checkpoint(tmp_path) -> None:
    model = RealNVP(
        input_dim=2,
        num_coupling_layers=2,
        hidden_dim=16,
        num_hidden_layers=1,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    checkpoint_path = tmp_path / "checkpoint.pt"

    torch.save(
        {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": 1.23,
        },
        checkpoint_path,
    )

    loaded_model = RealNVP(
        input_dim=2,
        num_coupling_layers=2,
        hidden_dim=16,
        num_hidden_layers=1,
    )

    checkpoint = load_model_checkpoint(
        model=loaded_model,
        checkpoint_path=checkpoint_path,
        device="cpu",
    )

    assert checkpoint["epoch"] == 1
    assert checkpoint["val_loss"] == pytest.approx(1.23)


def test_load_model_checkpoint_rejects_missing_file() -> None:
    model = RealNVP(input_dim=2)

    with pytest.raises(FileNotFoundError):
        load_model_checkpoint(
            model=model,
            checkpoint_path="missing_checkpoint.pt",
            device="cpu",
        )