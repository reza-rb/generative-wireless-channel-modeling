# Wireless Channel Modeling with Normalizing Flows

This repository implements a modular PyTorch project for learning probability
distributions of wireless channel coefficients using Normalizing Flows.

## Objective

The main density modeling problem is:

\[
p(h)
\]

where \(h\) is a complex-valued wireless channel coefficient:

\[
h = h_\text{real} + j h_\text{imag}
\]

For neural network processing, each complex sample is represented as a real-valued vector:

\[
x = [\operatorname{Re}(h), \operatorname{Im}(h)]
\]

The project learns:

\[
p(x)
\]

using flow-based generative models.

## Channel Models

The repository will include reusable simulators for:

1. Rayleigh fading channels
2. Rician fading channels
3. Multipath fading channels
4. Simple MIMO channel samples as a later extension

## Target Flow Models

1. RealNVP
2. Masked Autoregressive Flow, MAF
3. Neural Spline Flow, NSF
4. Optional: Glow-style invertible 1x1 convolution

## Repository Structure

```text
configs/        YAML experiment configurations
docs/           mathematical and wireless communication notes
notebooks/      exploratory notebooks
scripts/        training and evaluation entry points
src/gwcm/       main Python package
tests/          unit tests