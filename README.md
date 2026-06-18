# Wireless Channel Modeling with Normalizing Flows

This repository implements a modular PyTorch project for learning probability
distributions of wireless channel coefficients using Normalizing Flows.

## Objective

The main density modeling problem is:

$$
p(h)
$$

where $h$ is a complex-valued wireless channel coefficient:

$$
h = h_{\text{real}} + j h_{\text{imag}}
$$

For neural network processing, each complex sample is represented as a real-valued vector:

$$
x = [\mathrm{Re}(h), \mathrm{Im}(h)]
$$

The project learns:

$$
p(x)
$$

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
```

## Baseline Result: RealNVP on Rayleigh Fading

The first experiment trains RealNVP on scalar complex Rayleigh fading channel
coefficients represented as:

$$
x = [\mathrm{Re}(h), \mathrm{Im}(h)]
$$

For Rayleigh fading with $\sigma = 1.0$, the target distribution is:

$$
x \sim \mathcal{N}(0,I)
$$

The theoretical negative log-likelihood for a 2D standard Gaussian is:

$$
1 + \log(2\pi) \approx 2.8379
$$

Run the experiment:

```bash
python3 scripts/train_realnvp_rayleigh.py
```

The script saves:

```text
results/metrics/realnvp_rayleigh_metrics.json
results/figures/realnvp_rayleigh_comparison.png
results/checkpoints/realnvp_rayleigh_best.pt
```

Detailed results are documented in:

```text
docs/results/realnvp_rayleigh_results.md
```