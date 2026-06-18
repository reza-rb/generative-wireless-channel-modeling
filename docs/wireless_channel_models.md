cat > docs/wireless_channel_models.md <<'EOF'
# Wireless Channel Models

This document summarizes the wireless channel models used in the project.

The project focuses on scalar complex channel coefficients first. Each complex coefficient is represented as:

$$
h = h_{\mathrm{real}} + jh_{\mathrm{imag}}
$$

For neural network models, this is converted to:

$$
x = [\mathrm{Re}(h), \mathrm{Im}(h)]
$$

The density modeling problem becomes:

$$
p(x)
$$

instead of directly modeling complex-valued $p(h)$.

---

## 1. Rayleigh Fading

Rayleigh fading is used when there is no dominant line-of-sight path between transmitter and receiver.

This situation is common in environments with strong scattering, reflection, and diffraction.

Examples include:

- urban propagation,
- indoor propagation,
- non-line-of-sight links,
- mobile communication with many reflected paths.

A scalar Rayleigh fading channel is modeled as:

$$
h = h_{\mathrm{real}} + jh_{\mathrm{imag}}
$$

where:

$$
h_{\mathrm{real}} \sim \mathcal{N}(0, \sigma^2)
$$

and:

$$
h_{\mathrm{imag}} \sim \mathcal{N}(0, \sigma^2)
$$

The real and imaginary parts are independent Gaussian random variables.

The magnitude is:

$$
|h| = \sqrt{h_{\mathrm{real}}^2 + h_{\mathrm{imag}}^2}
$$

The magnitude follows a Rayleigh distribution.

The average channel power is:

$$
E[|h|^2] = 2\sigma^2
$$

For $\sigma = 1.0$:

$$
E[|h|^2] = 2
$$

### Machine Learning Representation

For machine learning, each channel coefficient is represented as:

$$
x = [\mathrm{Re}(h), \mathrm{Im}(h)]
$$

For $N$ samples, the dataset has shape:

$$
X \in R^{N \times 2}
$$

Rayleigh fading is the first baseline because its real-valued representation is a simple two-dimensional Gaussian distribution.

---

## 2. Rician Fading

Rician fading is used when the received signal contains:

- one dominant line-of-sight component,
- plus several scattered non-line-of-sight components.

The channel is modeled as:

$$
h = h_{\mathrm{LOS}} + h_{\mathrm{scatter}}
$$

where:

- $h_{\mathrm{LOS}}$ is the deterministic line-of-sight component.
- $h_{\mathrm{scatter}}$ is the random scattered component.

The strength of the line-of-sight component is controlled by the Rician K-factor:

$$
K = \frac{P_{\mathrm{LOS}}}{P_{\mathrm{scatter}}}
$$

where:

- $P_{\mathrm{LOS}}$ is the power of the line-of-sight component.
- $P_{\mathrm{scatter}}$ is the power of the scattered component.

The K-factor in decibels is:

$$
K_{\mathrm{dB}} = 10 \log_{10}(K)
$$

The conversion from decibels to linear scale is:

$$
K = 10^{K_{\mathrm{dB}}/10}
$$

A normalized Rician channel can be written as:

$$
h = \sqrt{\frac{K}{K+1}} h_{\mathrm{LOS}} + \sqrt{\frac{1}{K+1}} h_{\mathrm{NLOS}}
$$

where:

- $h_{\mathrm{LOS}}$ is usually a deterministic complex number.
- $h_{\mathrm{NLOS}}$ is a Rayleigh scattering component.
- $K$ is the linear Rician K-factor.

A simple choice for the line-of-sight component is:

$$
h_{\mathrm{LOS}} = e^{j\phi}
$$

where $\phi$ is the line-of-sight phase.

If $\phi = 0$, then:

$$
h_{\mathrm{LOS}} = 1
$$

### Interpretation of K-Factor

| K-factor | Interpretation |
|---:|---|
| $K = 0$ | No dominant line-of-sight path. Similar to Rayleigh fading. |
| Small $K$ | Weak line-of-sight component. |
| Large $K$ | Strong line-of-sight component. |
| Very large $K$ | Channel becomes concentrated around the deterministic component. |

### Machine Learning Representation

The model input is still:

$$
x = [\mathrm{Re}(h), \mathrm{Im}(h)]
$$

Compared with Rayleigh fading, Rician fading usually creates a shifted distribution in the complex plane.

This makes it more interesting for density modeling.

---

## 3. Multipath Fading

In real wireless propagation, the transmitted signal can reach the receiver through multiple paths.

Each path may have a different:

- amplitude,
- phase,
- delay,
- attenuation.

A general multipath impulse response can be written as:

$$
h(\tau) = \sum_{\ell=1}^{L} \alpha_{\ell} \delta(\tau - \tau_{\ell})
$$

where:

- $L$ is the number of paths.
- $\alpha_{\ell}$ is the complex gain of path $\ell$.
- $\tau_{\ell}$ is the delay of path $\ell$.
- $\delta(\tau - \tau_{\ell})$ represents a delayed impulse.

For the first version of this project, we use a flat-fading equivalent channel coefficient:

$$
h = \sum_{\ell=1}^{L} \alpha_{\ell}
$$

Each path gain is:

$$
\alpha_{\ell} = a_{\ell} e^{j\phi_{\ell}}
$$

where:

- $a_{\ell}$ is the path amplitude.
- $\phi_{\ell}$ is the path phase.

The phase is usually sampled randomly:

$$
\phi_{\ell} \sim U(0, 2\pi)
$$

The path power can follow an exponential decay profile:

$$
P_{\ell} = e^{-\ell / \lambda}
$$

where:

- $P_{\ell}$ is the power of path $\ell$.
- $\lambda$ controls how quickly later paths become weaker.

The path amplitude is:

$$
a_{\ell} = \sqrt{P_{\ell}}
$$

because power is amplitude squared.

### Machine Learning Representation

After summing the paths, the resulting complex coefficient is represented as:

$$
x = [\mathrm{Re}(h), \mathrm{Im}(h)]
$$

For $N$ generated channel samples:

$$
X \in R^{N \times 2}
$$

Multipath fading can produce richer distributions than basic Rayleigh fading, especially when the number of paths is small or when a deterministic line-of-sight component is added.

---

## 4. Flat Fading vs Frequency-Selective Fading

This project starts with flat fading.

In flat fading, the channel is represented by one complex coefficient:

$$
h
$$

This is a simplification, but it is useful for starting the density modeling problem.

A more advanced frequency-selective channel depends on frequency or subcarrier index. For an OFDM-like model, the channel on subcarrier $k$ can be written as:

$$
h[k] = \sum_{\ell=1}^{L} \alpha_{\ell} e^{-j2\pi k \Delta f \tau_{\ell}}
$$

where:

- $k$ is the subcarrier index.
- $\Delta f$ is the subcarrier spacing.
- $\tau_{\ell}$ is the delay of path $\ell$.

This extension can be added later.

---

## 5. MIMO Channel Extension

A MIMO channel has multiple transmit and receive antennas.

The channel is represented as a complex matrix:

$$
H \in C^{N_r \times N_t}
$$

where:

- $N_r$ is the number of receive antennas.
- $N_t$ is the number of transmit antennas.

For neural networks, the complex matrix can be flattened and converted into a real-valued vector:

$$
x = [\mathrm{Re}(H), \mathrm{Im}(H)]
$$

The input dimension becomes:

$$
D = 2N_rN_t
$$

So the density modeling problem becomes:

$$
p(x)
$$

where:

$$
x \in R^D
$$

This is a natural extension after scalar Rayleigh, Rician, and multipath channel modeling.

---

## 6. Summary

| Channel model | Main assumption | Distribution behavior |
|---|---|---|
| Rayleigh | No dominant line-of-sight path | Circular cloud centered around zero |
| Rician | Line-of-sight plus scattering | Shifted cloud toward LOS component |
| Multipath | Several explicit propagation paths | Shape depends on number of paths, phases, and path powers |
| MIMO | Multiple transmit and receive antennas | Higher-dimensional complex channel distribution |

The first implementation models scalar complex coefficients. Later versions can extend the same framework to frequency-selective channels and MIMO channel matrices.
EOF