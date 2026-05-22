# Wireless Channel Models

## Rayleigh Fading

Rayleigh fading models non-line-of-sight wireless propagation.

A complex Rayleigh channel coefficient can be generated as:

\[
h = h_\text{real} + j h_\text{imag}
\]

where:

\[
h_\text{real} \sim \mathcal{N}(0, \sigma^2)
\]

\[
h_\text{imag} \sim \mathcal{N}(0, \sigma^2)
\]

The magnitude:

\[
|h|
\]

follows a Rayleigh distribution.

## Rician Fading

Rician fading includes a line-of-sight component plus scattered components.

It is controlled by the Rician K-factor:

\[
K = \frac{\text{power of line-of-sight component}}
{\text{power of scattered component}}
\]

## Multipath Fading

In multipath fading, the received signal is affected by several delayed paths:

\[
h(t) = \sum_{\ell=1}^{L} \alpha_\ell \delta(t - \tau_\ell)
\]

where:

- \(L\) is the number of paths
- \(\alpha_\ell\) is the complex gain of path \(\ell\)
- \(\tau_\ell\) is the delay of path \(\ell\)