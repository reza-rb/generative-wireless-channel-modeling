# RealNVP on Rayleigh Fading Channels

## 1. Experiment Overview

This experiment trains a RealNVP normalizing flow to learn the probability
distribution of scalar complex Rayleigh fading channel coefficients.

The target density is:

$$
p(h)
$$

where:

$$
h = h_\text{real} + jh_\text{imag}
$$

For neural network processing, the complex-valued channel coefficient is
represented as a real-valued vector:

$$
x =
\begin{bmatrix}
\operatorname{Re}(h) \\
\operatorname{Im}(h)
\end{bmatrix}
\in \mathbb{R}^2
$$

Therefore, the actual density learned by RealNVP is:

$$
p(x)
$$

---

## 2. Wireless Channel Model

Rayleigh fading is used to model non-line-of-sight wireless propagation, where
the received signal is the sum of many scattered components and no dominant
line-of-sight path exists.

The simulator generates:

$$
h_\text{real} \sim \mathcal{N}(0, \sigma^2)
$$

$$
h_\text{imag} \sim \mathcal{N}(0, \sigma^2)
$$

and:

$$
h = h_\text{real} + jh_\text{imag}
$$

In this experiment:

$$
\sigma = 1.0
$$

Therefore:

$$
x =
[\operatorname{Re}(h), \operatorname{Im}(h)]
\sim
\mathcal{N}(0, I)
$$

The theoretical average channel power is:

$$
\mathbb{E}[|h|^2]
=
\mathbb{E}[h_\text{real}^2 + h_\text{imag}^2]
=
2\sigma^2
=
2
$$

---

## 3. Model

The model is RealNVP, a flow-based generative model using affine coupling
layers.

The model learns an invertible transformation between data samples $x$ and
latent variables $z$:

$$
z = f^{-1}(x)
$$

$$
x = f(z)
$$

The base distribution is a standard Gaussian:

$$
z \sim \mathcal{N}(0, I)
$$

The likelihood is computed using the change-of-variables formula:

$$
\log p_X(x)
=
\log p_Z(z)
+
\log
\left|
\det
\frac{\partial f^{-1}(x)}{\partial x}
\right|
$$

---

## 4. Training Objective

The model is trained by minimizing negative log-likelihood:

$$
\mathcal{L}(\theta)
=
-\frac{1}{B}
\sum_{i=1}^{B}
\log p_\theta(x_i)
$$

where:

- $B$ is the batch size.
- $x_i$ is one channel sample.
- $p_\theta(x_i)$ is the learned density assigned by RealNVP.

---

## 5. Experiment Configuration

| Category | Value |
|---|---|
| Channel model | Rayleigh fading |
| Number of samples | 10,000 |
| Input representation | $[\operatorname{Re}(h), \operatorname{Im}(h)]$ |
| Input dimension | 2 |
| $\sigma$ | 1.0 |
| Train/validation/test split | 80% / 10% / 10% |
| Batch size | 256 |
| Model | RealNVP |
| Coupling layers | 6 |
| Hidden dimension | 128 |
| Hidden layers per network | 2 |
| Optimizer | Adam |
| Learning rate | 0.0005 |
| Weight decay | 0.0 |
| Epochs | 100 |
| Gradient clipping | 5.0 |
| Seed | 42 |

---

## 6. Results


```bash
python3 scripts/train_realnvp_rayleigh.py
```

| Metric | Value |
|---|---:|
| Final training NLL | 2.8341 |
| Final validation NLL | 2.8951 |
| Best validation NLL | 2.8853 |
| Test NLL | 2.8879 |
| Test mean log-likelihood | -2.8879 |
| Reference average power | 2.0189 |
| Generated average power | 1.9414 |

---

## 7. Theoretical NLL Reference

For a 2D standard Gaussian:

$$
x \sim \mathcal{N}(0,I)
$$

the theoretical negative log-likelihood is:

$$
\text{NLL}_\text{theory}
=
\frac{d}{2}
(1 + \log(2\pi))
$$

For:

$$
d = 2
$$

we get:

$$
\text{NLL}_\text{theory}
=
1 + \log(2\pi)
\approx 2.8379
$$

Because Rayleigh fading with $\sigma = 1.0$ produces:

$$
[\operatorname{Re}(h), \operatorname{Im}(h)] \sim \mathcal{N}(0,I)
$$

a well-trained RealNVP model should achieve a test NLL close to this value.

---

## 8. Visual Results

After running the experiment script, the following plots are generated:

```text
results/figures/rayleigh_reference_scatter.png
results/figures/realnvp_rayleigh_generated_scatter.png
results/figures/realnvp_rayleigh_comparison.png
```

Because the `results/` directory is ignored by Git, selected figures should be copied into the documentation assets directory:

```bash
mkdir -p docs/assets
cp results/figures/realnvp_rayleigh_comparison.png docs/assets/realnvp_rayleigh_comparison.png
```

The comparison plot is shown below:

![RealNVP Rayleigh Comparison](../assets/realnvp_rayleigh_comparison.png)

### Expected Visual Interpretation

The reference Rayleigh samples should form a circular cloud centered around zero in the complex plane.

The generated RealNVP samples should also form a similar circular cloud after training.

This means the model has learned the joint density of the real and imaginary channel components:

$$
[\operatorname{Re}(h), \operatorname{Im}(h)]
$$

rather than only the magnitude distribution:

$$
|h|
$$

---

## 9. Interpretation

The RealNVP model successfully learns the Rayleigh fading distribution if:

1. The training and validation negative log-likelihood decrease and stabilize.
2. The test negative log-likelihood is close to the theoretical Gaussian reference value.
3. The generated samples have mean close to zero.
4. The generated average power is close to the reference average power.
5. The generated scatter plot visually matches the Rayleigh reference scatter plot.

This experiment is a baseline. Since Rayleigh fading with $\sigma = 1.0$ produces a two-dimensional standard Gaussian distribution,

$$
[\operatorname{Re}(h), \operatorname{Im}(h)] \sim \mathcal{N}(0,I),
$$

it is expected to be relatively easy for RealNVP.

The purpose of this experiment is to verify that the full pipeline works correctly before moving to more structured wireless channel models such as Rician and multipath fading.

---

## 10. Limitations

This experiment is intentionally simple.

Current limitations:

1. The channel is scalar, not MIMO.
2. The Rayleigh distribution is analytically simple.
3. The model is trained only on synthetic data.
4. The experiment does not yet include distribution-distance metrics.
5. The experiment does not yet compare RealNVP with MAF or NSF.
6. The channel is flat fading, not frequency-selective.

---

## 11. Next Experiments

The next experiments should extend the same pipeline to:

1. Rician fading with different K-factors.
2. Multipath fading with different numbers of paths.
3. RealNVP comparison across Rayleigh, Rician, and multipath channels.
4. MAF and NSF implementation.
5. Comparison between RealNVP, MAF, and NSF.
6. Simple MIMO channel samples.
7. Additional evaluation metrics such as MMD or Wasserstein distance.

---

## 12. Reproducibility

Run the experiment from the repository root:

```bash
python3 scripts/train_realnvp_rayleigh.py
```

Expected generated artifacts:

```text
results/checkpoints/realnvp_rayleigh_best.pt
results/metrics/realnvp_rayleigh_metrics.json
results/figures/rayleigh_reference_scatter.png
results/figures/realnvp_rayleigh_generated_scatter.png
results/figures/realnvp_rayleigh_comparison.png
```

The experiment uses seed `42` for reproducibility.