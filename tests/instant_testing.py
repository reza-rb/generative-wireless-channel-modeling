from gwcm.channels.rayleigh import (
    generate_rayleigh_samples,
    average_channel_power,
    rayleigh_magnitude,
)

samples = generate_rayleigh_samples(
    num_samples=10_000,
    sigma=1.0,
    seed=42, 
)
print (f"the outputs for rayleigh samples are: ")
print(samples.shape)
print(samples[:5])
print(average_channel_power(samples))
print(rayleigh_magnitude(samples).shape)

print(samples[:, 0].mean())
print(samples[:, 1].mean()) 

print(samples[:, 0].std())
print(samples[:, 1].std())


from gwcm.channels.rician import generate_rician_samples, average_channel_power

samples_k0 = generate_rician_samples(
    num_samples=10_000,
    k_factor=0.0,
    seed=42,
)

samples_k10 = generate_rician_samples(
    num_samples=10_000,
    k_factor=10.0,
    seed=42,
)

samples_k100 = generate_rician_samples(
    num_samples=10_000,
    k_factor=100.0,
    seed=42,
)

print (f"the outputs for rician samples with k=0, 10, 100 are: ")
print(samples_k0.mean(axis=0))
print(samples_k10.mean(axis=0))
print(samples_k100.mean(axis=0))

print(average_channel_power(samples_k10))

from gwcm.channels.multipath import (
    generate_multipath_samples,
    average_channel_power,
)

samples = generate_multipath_samples(
    num_samples=10_000,
    num_paths=5,
    decay_factor=1.0,
    seed=42,
)
print (f"the outputs for multipath samples are: ")
print(samples.shape)
print(samples[:5])
print(samples.mean(axis=0))
print(average_channel_power(samples))


samples_los = generate_multipath_samples(
    num_samples=10_000,
    num_paths=5,
    decay_factor=1.0,
    los_component=1.0 + 0.0j,
    normalize_power=False,
    seed=42,
)
print (f"the outputs for multipath samples with LOS component are: ")
print(samples_los.mean(axis=0))


from gwcm.channels.rayleigh import generate_rayleigh_samples
from gwcm.data.channel_dataset import create_dataloaders

samples = generate_rayleigh_samples(
    num_samples=10_000,
    sigma=1.0,
    seed=42,
)

loaders = create_dataloaders(
    samples=samples,
    batch_size=256,
    train_split=0.8,
    val_split=0.1,
    test_split=0.1,
    seed=42,
)

batch = next(iter(loaders.train))
print (f"the outputs for dataloader batch are: ")
print(batch.shape)
print(batch.dtype)
print(batch.mean(dim=0))
print(batch.std(dim=0))

import matplotlib.pyplot as plt
from gwcm.visualization.distributions import(
    plot_complex_scatter,
    plot_real_imag_histograms,
    plot_magnitude_histogram,
)

rayleigh = generate_rayleigh_samples(
    num_samples=10_000,
    sigma=1.0,
    seed=42,
)

rician = generate_rician_samples(
    num_samples=10_000,
    k_factor=10.0,  
    seed=42,
)

multipath = generate_multipath_samples(
    num_samples=10_000,
    num_paths=5,
    decay_factor=1.0,
    seed=42,
)

plot_complex_scatter(rayleigh, title="Rayleigh Samples",
  save_path="results/figures/rayleigh_scatter.png"                   
)
plot_complex_scatter(rician, title="Rician Samples (k=10)",
  save_path="results/figures/rician_scatter.png"                   
)
plot_complex_scatter(multipath, title="Multipath Samples",
  save_path="results/figures/multipath_scatter.png"                   
)


plot_real_imag_histograms(rayleigh,
    title="Rayleigh Real and Imaginary Components",
    save_path="results/figures/rayleigh_real_imag_hist.png",
)

plot_magnitude_histogram(rayleigh,
    title="Rayleigh Magnitude Distribution",
    save_path="results/figures/rayleigh_magnitude_hist.png",
)

plt.show()





import torch

from gwcm.models.flows.realnvp import RealNVP

model = RealNVP(
    input_dim=2,
    num_coupling_layers=6,
    hidden_dim=128,
    num_hidden_layers=2,
)

x = torch.randn(256, 2)

z, log_det = model.forward(x)
x_reconstructed, inverse_log_det = model.inverse(z)
log_prob = model.log_prob(x)
loss = model.negative_log_likelihood(x)
samples = model.sample(1000)

print (f"the outputs for realnvp model are: ")

print("x:", x.shape)
print("z:", z.shape)
print("log_det:", log_det.shape)
print("log_prob:", log_prob.shape)
print("loss:", loss.item())
print("samples:", samples.shape)
print("reconstruction error:", torch.max(torch.abs(x - x_reconstructed)).item())