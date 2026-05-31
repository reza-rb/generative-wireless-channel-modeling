"""PyTorch datasets and dataloaders for wireless channel samples.

This module wraps real-valued channel samples into PyTorch Dataset objects.

A complex wireless channel coefficient:

    h = h_real + j * h_imag

is represented as:

    x = [Re(h), Im(h)]

Therefore, a dataset of N scalar complex channel coefficients has shape:

    (N, 2)

For later MIMO extensions, the input dimension may become larger:

    (N, D)

where D = 2 * num_receive_antennas * num_transmit_antennas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class ChannelDataset(Dataset[torch.Tensor]):
    """PyTorch Dataset for real-valued wireless channel samples.

    Args:
        samples:
            Channel samples with shape (num_samples, input_dim). For scalar
            complex channel coefficients, input_dim = 2 and columns represent
            [Re(h), Im(h)].
        dtype:
            Tensor dtype used for storing samples.

    Raises:
        ValueError:
            If samples do not have shape (num_samples, input_dim).
    """

    def __init__(
        self,
        samples: np.ndarray | torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if isinstance(samples, np.ndarray):
            tensor_samples = torch.as_tensor(samples, dtype=dtype)
        elif isinstance(samples, torch.Tensor):
            tensor_samples = samples.to(dtype=dtype)
        else:
            raise TypeError(
                "samples must be either a NumPy array or a PyTorch tensor."
            )

        if tensor_samples.ndim != 2:
            raise ValueError(
                "samples must have shape (num_samples, input_dim)."
            )

        if tensor_samples.shape[0] <= 0:
            raise ValueError("samples must contain at least one sample.")

        if tensor_samples.shape[1] <= 0:
            raise ValueError("input_dim must be positive.")

        self.samples = tensor_samples

    def __len__(self) -> int:
        """Return the number of channel samples."""
        return self.samples.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        """Return one channel sample.

        Args:
            index:
                Sample index.

        Returns:
            Tensor with shape (input_dim,).
        """
        return self.samples[index]

    @property
    def input_dim(self) -> int:
        """Return the dimensionality of each channel sample."""
        return self.samples.shape[1]


@dataclass(frozen=True)
class ChannelDataLoaders:
    """Container for train, validation, and test dataloaders."""

    train: DataLoader[torch.Tensor]
    val: DataLoader[torch.Tensor]
    test: DataLoader[torch.Tensor]


@dataclass(frozen=True)
class DatasetSplits:
    """Container for split sizes."""

    train_size: int
    val_size: int
    test_size: int


def compute_split_sizes(
    num_samples: int,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
) -> DatasetSplits:
    """Compute train, validation, and test split sizes.

    Args:
        num_samples:
            Total number of samples.
        train_split:
            Fraction of samples used for training.
        val_split:
            Fraction of samples used for validation.
        test_split:
            Fraction of samples used for testing.

    Returns:
        DatasetSplits containing integer split sizes.

    Raises:
        ValueError:
            If split values are invalid.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    splits = train_split + val_split + test_split

    if not np.isclose(splits, 1.0):
        raise ValueError(
            "train_split + val_split + test_split must equal 1.0."
        )

    if train_split <= 0 or val_split <= 0 or test_split <= 0:
        raise ValueError("All split fractions must be positive.")

    train_size = int(num_samples * train_split)
    val_size = int(num_samples * val_split)
    test_size = num_samples - train_size - val_size

    if train_size <= 0 or val_size <= 0 or test_size <= 0:
        raise ValueError(
            "Each split must contain at least one sample. "
            "Increase num_samples or adjust split ratios."
        )

    return DatasetSplits(
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
    )


def create_dataloaders(
    samples: np.ndarray | torch.Tensor,
    batch_size: int = 256,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 42,
    shuffle_train: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> ChannelDataLoaders:
    """Create train, validation, and test dataloaders.

    Args:
        samples:
            Channel samples with shape (num_samples, input_dim).
        batch_size:
            Number of samples per batch.
        train_split:
            Fraction of data used for training.
        val_split:
            Fraction of data used for validation.
        test_split:
            Fraction of data used for testing.
        seed:
            Random seed used for reproducible dataset splitting.
        shuffle_train:
            If True, shuffle training batches.
        num_workers:
            Number of subprocesses used for data loading.
        pin_memory:
            If True, DataLoader copies tensors into CUDA pinned memory.
            Usually useful when training on GPU.

    Returns:
        ChannelDataLoaders object containing train, val, and test loaders.

    Raises:
        ValueError:
            If batch_size is invalid.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    dataset = ChannelDataset(samples)

    split_sizes = compute_split_sizes(
        num_samples=len(dataset),
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
    )

    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        lengths=[
            split_sizes.train_size,
            split_sizes.val_size,
            split_sizes.test_size,
        ],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return ChannelDataLoaders(
        train=train_loader,
        val=val_loader,
        test=test_loader,
    )