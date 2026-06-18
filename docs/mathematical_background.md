cat > docs/mathematical_background.md <<'EOF'
# Mathematical Background

This document summarizes the mathematical ideas used in the project.

The project studies density modeling of wireless channel coefficients using normalizing flows.

---

## 1. Complex Channel Representation

A scalar wireless channel coefficient is complex-valued:

$$
h = h_{\mathrm{real}} + jh_{\mathrm{imag}}
$$

where:

- \(h\) is the complex channel coefficient.
- \(h_{\mathrm{real}}\) is the real part.
- \(h_{\mathrm{imag}}\) is the imaginary part.
- \(j\) is the imaginary unit.

Most PyTorch neural networks use real-valued tensors. Therefore, each complex channel coefficient is represented as a two-dimensional real vector:

$$
x = [\mathrm{Re}(h), \mathrm{Im}(h)]
$$

So the original density modeling problem:

$$
p(h)
$$

is implemented as:

$$
p(x)
$$

where \(x\) contains the real and imaginary parts of the channel coefficient.

For one scalar complex channel coefficient:

$$
x \in R^2
$$

For a dataset with \(N\) channel samples:

$$
X \in R^{N \times 2}
$$

The first column contains the real parts and the second column contains the imaginary parts.

---

## 2. Density Modeling Objective

The objective is to learn the probability distribution of wireless channel samples.

Given training samples:

$$
x_1, x_2, ..., x_N
$$

we want to learn a parametric density model:

$$
p_{\theta}(x)
$$

where \(\theta\) represents the trainable parameters of the model.

A good model should assign high probability to realistic channel samples and low probability to unlikely channel samples.

---

## 3. Normalizing Flows

A normalizing flow is an invertible neural network that maps data samples \(x\) to latent variables \(z\).

The data-to-latent direction is:

$$
z = f^{-1}(x)
$$

The latent-to-data direction is:

$$
x = f(z)
$$

The latent variable usually follows a simple base distribution:

$$
z \sim \mathcal{N}(0, I)
$$

where \(I\) is the identity covariance matrix.

---

## 4. Change of Variables Formula

Normalizing flows compute exact likelihoods using the change-of-variables formula.

The density of \(x\) is:

$$
p_X(x) = p_Z(z) |\det J|
$$

where:

- \(x\) is the data sample.
- \(z = f^{-1}(x)\) is the latent representation.
- \(p_Z(z)\) is the base density.
- \(J\) is the Jacobian of the inverse transformation.
- \(|\det J|\) is the absolute determinant of the Jacobian.

Taking the logarithm gives:

$$
\log p_X(x) = \log p_Z(z) + \log |\det J|
$$

This expression is used directly during training.

---

## 5. Negative Log-Likelihood

The model is trained by maximizing log-likelihood:

$$
\max_{\theta} \frac{1}{N} \sum_{i=1}^{N} \log p_{\theta}(x_i)
$$

In PyTorch, optimization is usually written as minimization. Therefore, we minimize the negative log-likelihood:

$$
L(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \log p_{\theta}(x_i)
$$

For a mini-batch with batch size \(B\):

$$
L_B(\theta) = -\frac{1}{B} \sum_{i=1}^{B} \log p_{\theta}(x_i)
$$

Lower negative log-likelihood means the model assigns higher probability to the observed channel samples.

---

## 6. RealNVP Coupling Layer

RealNVP uses affine coupling layers.

The input vector \(x\) is split into two parts:

$$
x = [x_a, x_b]
$$

One part remains unchanged:

$$
y_a = x_a
$$

The other part is transformed using scale and translation networks:

$$
y_b = x_b \exp(s(x_a)) + t(x_a)
$$

where:

- \(s(x_a)\) is the scale network output.
- \(t(x_a)\) is the translation network output.
- \(\exp\) makes the scaling positive.

The inverse transformation is:

$$
x_b = (y_b - t(y_a)) \exp(-s(y_a))
$$

This makes the transformation invertible and efficient.

---

## 7. Log-Determinant in RealNVP

For the affine transformation:

$$
y_b = x_b \exp(s) + t
$$

the log-determinant is:

$$
\log |\det J| = \sum s
$$

This is computationally efficient because the full Jacobian determinant does not need to be calculated directly.

---

## 8. Rayleigh Fading as a Baseline

For Rayleigh fading:

$$
h_{\mathrm{real}} \sim \mathcal{N}(0, \sigma^2)
$$

$$
h_{\mathrm{imag}} \sim \mathcal{N}(0, \sigma^2)
$$

Therefore:

$$
x = [\mathrm{Re}(h), \mathrm{Im}(h)]
$$

follows a two-dimensional Gaussian distribution.

For \(\sigma = 1.0\):

$$
x \sim \mathcal{N}(0, I)
$$

The theoretical negative log-likelihood for a two-dimensional standard Gaussian is:

$$
NLL_{\mathrm{theory}} = 1 + \log(2\pi)
$$

Numerically:

$$
NLL_{\mathrm{theory}} \approx 2.8379
$$

This value is useful for checking whether the RealNVP implementation is working correctly.

---

## 9. Why This Matters for Wireless Channels

Wireless channels are random because of reflection, scattering, diffraction, mobility, and environmental changes.

A generative density model can learn:

- the shape of the channel distribution,
- the likelihood of observed channel samples,
- how generated samples compare to simulated channel models,
- uncertainty in wireless propagation.

This project starts with scalar complex channels and can later be extended to MIMO channels, conditional channel estimation, and neural receivers.
EOF