import torch

from gwcm.channels.rayleigh import generate_rayleigh_samples 
from gwcm.data.channel_dataset import create_dataloaders
from gwcm.models.flows.realnvp import RealNVP
from gwcm.training.trainer import train_model

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

model = RealNVP(
    input_dim=2,
    num_coupling_layers=6,
    hidden_dim=128,
    num_hidden_layers=2,
    scale_clamp=2.0,
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=5e-4,
    weight_decay=0.0,
)

history = train_model(
    model=model,
    train_loader=loaders.train,
    val_loader=loaders.val,
    optimizer=optimizer,
    device="mps" if torch.backends.mps.is_available() else "cpu",
    epochs=50,
    grad_clip_norm=5.0,
    checkpoint_path="results/checkpoints/realnvp_rayleigh_best.pt",
)

print("Final train loss:", history.train_losses[-1])
print("Final val loss:", history.val_losses[-1])
print("Best val loss:", history.best_val_loss)
print("mps is available:", torch.backends.mps.is_available())