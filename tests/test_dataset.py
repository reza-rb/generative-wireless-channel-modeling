"""Tests for wireless channel PyTorch datasets and dataloaders."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gwcm.channels.rayleigh import generate_rayleigh_samples
from gwcm.data.channel_dataset import (
    ChannelDataset,
    compute_split_sizes,
    create_dataloaders,
)


def test_channel_dataset_from_numpy() -> None:
    samples = np.random.randn(100, 2).astype(np.float32)
    dataset = ChannelDataset(samples)

    assert len(dataset) == 100
    assert dataset.input_dim == 2
    assert isinstance(dataset[0], torch.Tensor)
    assert dataset[0].shape == (2,)
    assert dataset[0].dtype == torch.float32


def test_channel_dataset_from_tensor() -> None:
    samples = torch.randn(100, 2)
    dataset = ChannelDataset(samples)

    assert len(dataset) == 100
    assert dataset.input_dim == 2
    assert dataset[0].shape == (2,)


def test_channel_dataset_rejects_invalid_shape() -> None:
    samples = np.random.randn(100).astype(np.float32)

    with pytest.raises(ValueError, match="samples must have shape"):
        ChannelDataset(samples)


def test_channel_dataset_rejects_empty_samples() -> None:
    samples = np.empty((0, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="at least one sample"):
        ChannelDataset(samples)


def test_compute_split_sizes() -> None:
    split_sizes = compute_split_sizes(
        num_samples=1000,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
    )

    assert split_sizes.train_size == 800
    assert split_sizes.val_size == 100
    assert split_sizes.test_size == 100


def test_compute_split_sizes_handles_rounding() -> None:
    split_sizes = compute_split_sizes(
        num_samples=1001,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
    )

    assert split_sizes.train_size == 800
    assert split_sizes.val_size == 100
    assert split_sizes.test_size == 101


def test_compute_split_sizes_rejects_invalid_sum() -> None:
    with pytest.raises(ValueError, match="must equal 1.0"):
        compute_split_sizes(
            num_samples=1000,
            train_split=0.7,
            val_split=0.2,
            test_split=0.2,
        )


def test_create_dataloaders_batch_shape() -> None:
    samples = generate_rayleigh_samples(
        num_samples=1000,
        sigma=1.0,
        seed=42,
    )

    loaders = create_dataloaders(
        samples=samples,
        batch_size=128,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        seed=42,
    )

    batch = next(iter(loaders.train))

    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (128, 2)
    assert batch.dtype == torch.float32


def test_create_dataloaders_split_lengths() -> None:
    samples = generate_rayleigh_samples(
        num_samples=1000,
        sigma=1.0,
        seed=42,
    )

    loaders = create_dataloaders(
        samples=samples,
        batch_size=128,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        seed=42,
    )

    assert len(loaders.train.dataset) == 800
    assert len(loaders.val.dataset) == 100
    assert len(loaders.test.dataset) == 100


def test_create_dataloaders_rejects_invalid_batch_size() -> None:
    samples = generate_rayleigh_samples(
        num_samples=100,
        sigma=1.0,
        seed=42,
    )

    with pytest.raises(ValueError, match="batch_size must be"):
        create_dataloaders(samples=samples, batch_size=0)