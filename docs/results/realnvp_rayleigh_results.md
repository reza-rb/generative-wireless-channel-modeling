cat > docs/results/realnvp_rayleigh_results.md <<'EOF'
# RealNVP on Rayleigh Fading Channels

## 1. Experiment Overview

This experiment trains a RealNVP normalizing flow to learn the probability distribution of scalar complex Rayleigh fading channel coefficients.

The target density is:

$$
p(h)
$$

where the complex wireless channel coefficient is:

$$
h = h_{\mathrm{real}} + jh_{\mathrm{imag}}
$$

For neural network processing, each complex-valued channel sample is represented as a real-valued vector:

$$
x = [\mathrm{Re}(h), \mathrm{Im}(h)]
$$

Therefore, the model learns the density:

$$
p(x)
$$

where \(x\) contains the real and imaginary parts of the channel coefficient.

---

## 2. Wireless Channel Model

Rayleigh fading is commonly used to model non-line-of-sight wireless propagation. In this case, the received signal is assumed to be formed by many scattered components, without a dominant line-of-sight path.

The simulator generates:

$$
h_{\mathrm{real}} \sim \mathcal{N}(0, \sigma^2)
$$

$$
h_{\mathrm{imag}} \sim \mathcal{N}(0, \sigma^2)
$$

and combines them as:

$$
h = h_{\mathrm{real}} + jh_{\mathrm{imag}}
$$

In this experiment:

$$
\sigma = 1.0
$$

Therefore, the real-valued model input follows:

$$
x = [\mathrm{Re}(h), \mathrm{Im}(h)] \sim \mathcal{N}(0, I)
$$

The theoretical average channel power is:

$$
E[|h|^2] = E[h_{\mathrm{real}}^2 + h_{\mathrm{imag}}^2] = 2\sigma^2 = 2
$$

This means that, for \(\sigma = 1.0\), the average channel power should be close to 2.

---

## 3. Model

The model is RealNVP, a flow-based generative model based on affine coupling layers.

RealNVP learns an invertible transformation between data samples \(x\) and latent variables \(z\):

$$
z = f^{-1}(x)
$$

and:

$$
x = f(z)
$$

The latent variable follows a standard Gaussian distribution:

$$
z \sim \mathcal{N}(0, I)
$$

The model computes exact likelihoods using the change-of-variables formula:

$$
\log p_X(x) = \log p_Z(z) + \log |\det J|
$$

where:

- \(x\) is the channel sample in real-valued form.
- \(z\) is the latent representation.
- \(J\) is the Jacobian of the inverse transformation.
- \(\log |\det J|\) is the log-determinant correction.

---

## 4. Training Objective

The model is trained by minimizing negative log-likelihood:

$$
L(\theta) = -\frac{1}{B} \sum_{i=1}^{B} \log p_{\theta}(x_i)
$$

where:

- \(B\) is the batch size.
- \(x_i\) is one wireless channel sample.
- \(p_{\theta}(x_i)\) is the density assigned by the RealNVP model.
- \(\theta\) represents the trainable model parameters.

The objective is to assign high likelihood to realistic Rayleigh fading channel samples.

---

## 5. Experiment Configuration

| Category | Value |
|---|---:|
| Channel model | Rayleigh fading |
| Number of samples | 10,000 |
| Input representation | \([\mathrm{Re}(h), \mathrm{Im}(h)]\) |
| Input dimension | 2 |
| \(\sigma\) | 1.0 |
| Train split | 80% |
| Validation split | 10% |
| Test split | 10% |
| Batch size | 256 |
| Model | RealNVP |
| Coupling layers | 6 |
| Hidden dimension | 128 |
| Hidden layers per coupling network | 2 |
| Optimizer | Adam |
| Learning rate | 0.0005 |
| Weight decay | 0.0 |
| Epochs | 100 |
| Gradient clipping | 5.0 |
| Seed | 42 |

---

## 6. Results

Replace the placeholder values below after running the experiment:

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


The values should be copied from:

```text
results/metrics/realnvp_rayleigh_metrics.json
```

---

## 7. Theoretical NLL Reference

For a two-dimensional standard Gaussian distribution:

$$
x \sim \mathcal{N}(0, I)
$$

the theoretical negative log-likelihood is:

$$
NLL_{\mathrm{theory}} = 1 + \log(2\pi)
$$

Numerically:

$$
NLL_{\mathrm{theory}} \approx 2.8379
$$

Because Rayleigh fading with \(\sigma = 1.0\) produces:

$$
[\mathrm{Re}(h), \mathrm{Im}(h)] \sim \mathcal{N}(0, I)
$$

a well-trained RealNVP model should achieve a test NLL close to 2.8379.

Small deviations are normal because of finite sample size, optimization noise, and model initialization.

---

## 8. Visual Results

After running the experiment script, the following plots are generated:

```text
results/figures/rayleigh_reference_scatter.png
results/figures/realnvp_rayleigh_generated_scatter.png
results/figures/realnvp_rayleigh_comparison.png
```

Because the root-level `results/` directory is ignored by Git, selected figures should be copied into the documentation assets directory:

```bash
mkdir -p docs/assets
cp results/figures/realnvp_rayleigh_comparison.png docs/assets/realnvp_rayleigh_comparison.png
```

The comparison plot is shown below:

![RealNVP Rayleigh Comparison](..docs/assets/realnvp_rayleigh_comparison.png)

### Expected Visual Interpretation

The reference Rayleigh samples should form a circular cloud centered around zero in the complex plane.

The generated RealNVP samples should also form a similar circular cloud after training.

This means the model has learned the joint density of the real and imaginary channel components:

$$
[\mathrm{Re}(h), \mathrm{Im}(h)]
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
4. The generated standard deviations are close to the reference standard deviations.
5. The generated average power is close to the reference average power.
6. The generated scatter plot visually matches the Rayleigh reference scatter plot.

This experiment is a baseline. Since Rayleigh fading with \(\sigma = 1.0\) produces a two-dimensional standard Gaussian distribution, it is expected to be relatively easy for RealNVP.

The purpose of this experiment is to verify that the complete pipeline works correctly before moving to more structured wireless channel models such as Rician and multipath fading.

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
7. The model currently learns only \(p(h)\), not conditional distributions such as \(p(h|y)\).

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

---

## 13. Summary

This experiment confirms whether the implemented RealNVP model can learn a simple but important wireless channel distribution.

A successful result should show:

- test NLL close to the theoretical Rayleigh/Gaussian reference,
- generated samples visually similar to Rayleigh reference samples,
- generated average power close to the theoretical and empirical reference power,
- reproducible training and evaluation artifacts.

This provides the first validated baseline for the repository.
EOF