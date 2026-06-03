"""Tests for RealNVP normalizing flow."""

from __future__ import annotations

import torch
import pytest

from gwcm.models.flows.realnvp import AffineCouplingLayer, MLP, RealNVP


def test_mlp_output_shape() -> None:
    model = MLP(
        input_dim=2,
        output_dim=2,
        hidden_dim=16,
        num_hidden_layers=2,
    )

    x = torch.randn(32, 2)
    y = model(x)

    assert y.shape == (32, 2)


def test_affine_coupling_forward_inverse_consistency() -> None:
    input_dim = 2
    mask = torch.tensor([1.0, 0.0])

    layer = AffineCouplingLayer(
        input_dim=input_dim,
        mask=mask,
        hidden_dim=16,
        num_hidden_layers=2,
    )

    x = torch.randn(64, input_dim)

    y, forward_log_det = layer.forward(x)
    x_reconstructed, inverse_log_det = layer.inverse(y)

    assert y.shape == x.shape
    assert forward_log_det.shape == (64,)
    assert inverse_log_det.shape == (64,)

    torch.testing.assert_close(
        x_reconstructed,
        x,
        rtol=1e-5,
        atol=1e-5,
    )

    torch.testing.assert_close(
        forward_log_det + inverse_log_det,
        torch.zeros_like(forward_log_det),
        rtol=1e-5,
        atol=1e-5,
    )


def test_realnvp_forward_output_shapes() -> None:
    model = RealNVP(
        input_dim=2,
        num_coupling_layers=4,
        hidden_dim=16,
        num_hidden_layers=2,
    )

    x = torch.randn(128, 2)

    z, log_det = model.forward(x)

    assert z.shape == (128, 2)
    assert log_det.shape == (128,)


def test_realnvp_inverse_output_shapes() -> None:
    model = RealNVP(
        input_dim=2,
        num_coupling_layers=4,
        hidden_dim=16,
        num_hidden_layers=2,
    )

    z = torch.randn(128, 2)

    x, log_det = model.inverse(z)

    assert x.shape == (128, 2)
    assert log_det.shape == (128,)


def test_realnvp_forward_inverse_consistency() -> None:
    model = RealNVP(
        input_dim=2,
        num_coupling_layers=6,
        hidden_dim=32,
        num_hidden_layers=2,
    )

    x = torch.randn(64, 2)

    z, forward_log_det = model.forward(x)
    x_reconstructed, inverse_log_det = model.inverse(z)

    assert z.shape == x.shape
    assert x_reconstructed.shape == x.shape

    torch.testing.assert_close(
        x_reconstructed,
        x,
        rtol=1e-5,
        atol=1e-5,
    )

    torch.testing.assert_close(
        forward_log_det + inverse_log_det,
        torch.zeros_like(forward_log_det),
        rtol=1e-5,
        atol=1e-5,
    )


def test_realnvp_log_prob_shape() -> None:
    model = RealNVP(
        input_dim=2,
        num_coupling_layers=4,
        hidden_dim=16,
        num_hidden_layers=2,
    )

    x = torch.randn(32, 2)

    log_prob = model.log_prob(x)

    assert log_prob.shape == (32,)
    assert torch.isfinite(log_prob).all()


def test_realnvp_negative_log_likelihood_is_scalar() -> None:
    model = RealNVP(
        input_dim=2,
        num_coupling_layers=4,
        hidden_dim=16,
        num_hidden_layers=2,
    )

    x = torch.randn(32, 2)

    loss = model.negative_log_likelihood(x)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_realnvp_sample_shape() -> None:
    model = RealNVP(
        input_dim=2,
        num_coupling_layers=4,
        hidden_dim=16,
        num_hidden_layers=2,
    )

    samples = model.sample(num_samples=100)

    assert samples.shape == (100, 2)
    assert torch.isfinite(samples).all()


def test_realnvp_rejects_invalid_input_shape() -> None:
    model = RealNVP(input_dim=2)

    x = torch.randn(32, 3)

    with pytest.raises(ValueError, match="expected input_dim"):
        model.log_prob(x)


def test_realnvp_rejects_invalid_sample_count() -> None:
    model = RealNVP(input_dim=2)

    with pytest.raises(ValueError, match="num_samples must be"):
        model.sample(num_samples=0)