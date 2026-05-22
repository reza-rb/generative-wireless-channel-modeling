# Mathematical Background

## Complex Channel Representation

A wireless channel coefficient is written as:

\[
h = h_\text{real} + j h_\text{imag}
\]

To use this value in a real-valued neural network, we represent it as:

\[
x =
\begin{bmatrix}
\operatorname{Re}(h) \\
\operatorname{Im}(h)
\end{bmatrix}
\]

Therefore, density modeling of \(p(h)\) becomes density modeling of \(p(x)\).

## Normalizing Flow Objective

A normalizing flow defines an invertible transformation between data \(x\)
and latent variable \(z\):

\[
z = f^{-1}(x)
\]

The log-likelihood is computed as:

\[
\log p_X(x)
=
\log p_Z(z)
+
\log \left| \det \frac{\partial f^{-1}(x)}{\partial x} \right|
\]

The training objective is negative log-likelihood:

\[
\mathcal{L}
=
-\frac{1}{N}
\sum_{i=1}^{N}
\log p_X(x_i)
\]