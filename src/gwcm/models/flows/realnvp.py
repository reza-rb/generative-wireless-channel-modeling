"""RealNVP normalizing flow implementation.

This module implements RealNVP for density modeling of wireless channel
coefficients represented as real-valued vectors.

For scalar complex channels:

    h = h_real + j * h_imag

we use:

    x = [Re(h), Im(h)]

with shape:

    (batch_size, 2)

The RealNVP model learns an invertible mapping between channel samples x and
latent Gaussian variables z.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MLP(nn.Module):
    """Simple multilayer perceptron used inside coupling layers.

    Args:
        input_dim:
            Input feature dimension.
        output_dim:
            Output feature dimension.
        hidden_dim:
            Hidden layer width.
        num_hidden_layers:
            Number of hidden layers.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        num_hidden_layers: int = 2,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        if num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive.")

        layers: list[nn.Module] = []

        current_dim = input_dim

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the MLP.

        Args:
            x:
                Input tensor with shape (batch_size, input_dim).

        Returns:
            Output tensor with shape (batch_size, output_dim).
        """
        return self.network(x)


class AffineCouplingLayer(nn.Module):
    """Affine coupling layer for RealNVP.

    A binary mask selects which dimensions remain unchanged and which
    dimensions are transformed.

    Forward transformation:

        y_masked = x_masked
        y_transformed = x_transformed * exp(s(x_masked)) + t(x_masked)

    Inverse transformation:

        x_transformed = (y_transformed - t(y_masked)) * exp(-s(y_masked))

    Args:
        input_dim:
            Number of input features.
        mask:
            Binary mask tensor with shape (input_dim,).
        hidden_dim:
            Hidden layer width for scale and translation networks.
        num_hidden_layers:
            Number of hidden layers in scale and translation networks.
        scale_clamp:
            Clamp value for scale output to improve numerical stability.
    """

    def __init__(
        self,
        input_dim: int,
        mask: Tensor,
        hidden_dim: int = 128,
        num_hidden_layers: int = 2,
        scale_clamp: float = 2.0,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        if mask.shape != (input_dim,):
            raise ValueError("mask must have shape (input_dim,).")

        if scale_clamp <= 0:
            raise ValueError("scale_clamp must be positive.")

        self.input_dim = input_dim
        self.scale_clamp = scale_clamp

        self.register_buffer("mask", mask.float())

        self.scale_net = MLP(
            input_dim=input_dim,
            output_dim=input_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
        )

        self.translate_net = MLP(
            input_dim=input_dim,
            output_dim=input_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Transform data x into y.

        Args:
            x:
                Input tensor with shape (batch_size, input_dim).

        Returns:
            Tuple containing:
                y:
                    Transformed tensor with shape (batch_size, input_dim).
                log_det:
                    Log absolute determinant of the Jacobian with shape
                    (batch_size,).
        """
        self._validate_input(x)

        x_masked = x * self.mask

        scale = self.scale_net(x_masked)
        translate = self.translate_net(x_masked)

        scale = torch.tanh(scale) * self.scale_clamp  #Clamp scale for numerical stability.

        transform_mask = 1.0 - self.mask

        y = x_masked + transform_mask * (
            x * torch.exp(scale) + translate
        )

        log_det = torch.sum(transform_mask * scale, dim=1)

        return y, log_det

    def inverse(self, y: Tensor) -> tuple[Tensor, Tensor]:
        """Transform y back into x.

        Args:
            y:
                Tensor with shape (batch_size, input_dim).

        Returns:
            Tuple containing:
                x:
                    Inverted tensor with shape (batch_size, input_dim).
                log_det:
                    Log absolute determinant of the inverse Jacobian with
                    shape (batch_size,).
        """
        self._validate_input(y)

        y_masked = y * self.mask

        scale = self.scale_net(y_masked)
        translate = self.translate_net(y_masked)

        scale = torch.tanh(scale) * self.scale_clamp   

        transform_mask = 1.0 - self.mask

        x = y_masked + transform_mask * (
            (y - translate) * torch.exp(-scale)
        )

        log_det = -torch.sum(transform_mask * scale, dim=1)

        return x, log_det

    def _validate_input(self, x: Tensor) -> None:
        """Validate input shape.

        Args:
            x:
                Input tensor.

        Raises:
            ValueError:
                If input tensor shape is invalid.
        """
        if x.ndim != 2:
            raise ValueError("input must have shape (batch_size, input_dim).")

        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"expected input_dim={self.input_dim}, "
                f"but received {x.shape[1]}."
            )


class RealNVP(nn.Module):
    """RealNVP normalizing flow model.

    The model maps data samples x to latent Gaussian variables z and computes
    exact log-likelihoods using the change-of-variables formula.

    Args:
        input_dim:
            Input feature dimension. For scalar complex channels, this is 2.
        num_coupling_layers:
            Number of affine coupling layers.
        hidden_dim:
            Hidden layer width for coupling networks.
        num_hidden_layers:
            Number of hidden layers in each coupling network.
        scale_clamp:
            Clamp value for coupling-layer scale output.
    """

    def __init__(
        self,
        input_dim: int = 2,
        num_coupling_layers: int = 6,
        hidden_dim: int = 128,
        num_hidden_layers: int = 2,
        scale_clamp: float = 2.0,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        if num_coupling_layers <= 0:
            raise ValueError("num_coupling_layers must be positive.")

        self.input_dim = input_dim

        masks = self._create_alternating_masks(
            input_dim=input_dim,
            num_masks=num_coupling_layers,
        )

        self.coupling_layers = nn.ModuleList(
            [
                AffineCouplingLayer(
                    input_dim=input_dim,
                    mask=mask,
                    hidden_dim=hidden_dim,
                    num_hidden_layers=num_hidden_layers,
                    scale_clamp=scale_clamp,
                )
                for mask in masks
            ]
        )

        self.base_distribution = torch.distributions.MultivariateNormal(
            loc=torch.zeros(input_dim),
            covariance_matrix=torch.eye(input_dim),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Map data samples x to latent variables z.

        This is the density-estimation direction:

            x -> z

        Args:
            x:
                Data tensor with shape (batch_size, input_dim).

        Returns:
            Tuple containing:
                z:
                    Latent tensor with shape (batch_size, input_dim).
                log_det:
                    Total log absolute determinant with shape (batch_size,).
        """
        self._validate_input(x)

        z = x  # Initialize z(z0) as x and apply coupling layers sequentially.
        total_log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        for layer in self.coupling_layers:
            z, log_det = layer.inverse(z)
            total_log_det = total_log_det + log_det

        return z, total_log_det

    def inverse(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """Map latent variables z to generated samples x.

        This is the sampling direction:

            z -> x

        Args:
            z:
                Latent tensor with shape (batch_size, input_dim).

        Returns:
            Tuple containing:
                x:
                    Generated data tensor with shape (batch_size, input_dim).
                log_det:
                    Total log absolute determinant with shape (batch_size,).
        """
        self._validate_input(z)

        x = z
        total_log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)

        for layer in reversed(self.coupling_layers):
            x, log_det = layer.forward(x)
            total_log_det = total_log_det + log_det

        return x, total_log_det

    def log_prob(self, x: Tensor) -> Tensor:
        """Compute log probability of data samples.

        Args:
            x:
                Data tensor with shape (batch_size, input_dim).

        Returns:
            Log probability tensor with shape (batch_size,).
        """
        z, log_det = self.forward(x)

        base_distribution = self._base_distribution_on_device(
            device=x.device,
            dtype=x.dtype,
        )

        base_log_prob = base_distribution.log_prob(z)  

        return base_log_prob + log_det

    def sample(self, num_samples: int, device: torch.device | str | None = None) -> Tensor:
        """Generate samples from the RealNVP model.

        Args:
            num_samples:
                Number of samples to generate.
            device:
                Optional device for sampling.

        Returns:
            Generated samples with shape (num_samples, input_dim).

        Raises:
            ValueError:
                If num_samples is not positive.
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")

        if device is None:
            device = next(self.parameters()).device

        base_distribution = self._base_distribution_on_device(
            device=torch.device(device),
            dtype=next(self.parameters()).dtype,
        )

        z = base_distribution.sample((num_samples,))
        x, _ = self.inverse(z)

        return x

    def negative_log_likelihood(self, x: Tensor) -> Tensor:
        """Compute mean negative log-likelihood.

        Args:
            x:
                Data tensor with shape (batch_size, input_dim).

        Returns:
            Scalar tensor containing mean negative log-likelihood.
        """
        return -self.log_prob(x).mean()

    @staticmethod
    def _create_alternating_masks(
        input_dim: int,
        num_masks: int,
    ) -> list[Tensor]:
        """Create alternating binary masks.

        Args:
            input_dim:
                Number of input dimensions.
            num_masks:
                Number of masks to create.

        Returns:
            List of binary mask tensors.
        """
        base_mask = torch.arange(input_dim) % 2
        inverse_mask = 1 - base_mask

        masks = []

        for index in range(num_masks):
            mask = base_mask if index % 2 == 0 else inverse_mask
            masks.append(mask.float())

        return masks

    def _base_distribution_on_device(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.distributions.MultivariateNormal:
        """Create base Gaussian distribution on the correct device and dtype.

        Args:
            device:
                Target device.
            dtype:
                Target dtype.

        Returns:
            Standard multivariate normal distribution.
        """
        loc = torch.zeros(self.input_dim, device=device, dtype=dtype)
        covariance_matrix = torch.eye(self.input_dim, device=device, dtype=dtype)

        return torch.distributions.MultivariateNormal(
            loc=loc,
            covariance_matrix=covariance_matrix,
        )

    def _validate_input(self, x: Tensor) -> None:
        """Validate input shape.

        Args:
            x:
                Input tensor.

        Raises:
            ValueError:
                If input tensor shape is invalid.
        """
        if x.ndim != 2:
            raise ValueError("input must have shape (batch_size, input_dim).")

        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"expected input_dim={self.input_dim}, "
                f"but received {x.shape[1]}."
            )