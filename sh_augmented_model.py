"""
Simple equivariant model using spherical harmonics (no e3nn Gate issues).
We use e3nn's spherical harmonics as features, then a standard MLP.
This is not fully equivariant but is rotation-sensitive.
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Load data
PROJECT_ROOT = Path(r"F:\Deep Riemannian Diffusion Experiment")
results = np.load("full_diffusion_results_final.npz")
vertices_ds = results["vertices_ds"]
field_true = results["field_ds"]

# Use only a subset for training
n_train = 5000
sample_idx = np.random.choice(len(vertices_ds), n_train, replace=False)
X_train = vertices_ds[sample_idx].astype(np.float32)
y_train = field_true[sample_idx].astype(np.float32).reshape(-1,1)

# Use e3nn for spherical harmonics
try:
    import e3nn
    from e3nn import o3
    max_degree = 4
    irreps_sh = o3.Irreps.spherical_harmonics(max_degree)
    sh_func = o3.SphericalHarmonics(irreps_sh, normalize=True, normalization='component')
    print("Using e3nn spherical harmonics.")
except ImportError:
    print("e3nn not installed; using simple random features.")
    sh_func = None

if sh_func is not None:
    # Compute SH for all vertices (on CPU, move to GPU later)
    sh_all = sh_func(torch.tensor(vertices_ds, dtype=torch.float32)).numpy()
    X_train_sh = np.concatenate([X_train, sh_all[sample_idx]], axis=1)
    input_dim = 3 + irreps_sh.dim
else:
    X_train_sh = X_train
    input_dim = 3
    print("Falling back to plain coordinates.")

# Simple MLP
class EquivariantMLP(nn.Module):
    def __init__(self, input_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x)

model = EquivariantMLP(input_dim, hidden=128)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Train
dataset = torch.utils.data.TensorDataset(torch.tensor(X_train_sh), torch.tensor(y_train))
loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True)

print("Training equivariant model...")
for epoch in tqdm(range(50)):
    for bx, by in loader:
        optimizer.zero_grad()
        pred = model(bx)
        loss = criterion(pred, by)
        loss.backward()
        optimizer.step()

# Evaluate on all vertices
if sh_func is not None:
    all_inputs = np.concatenate([vertices_ds, sh_all], axis=1)
else:
    all_inputs = vertices_ds
with torch.no_grad():
    pred_all = model(torch.tensor(all_inputs, dtype=torch.float32)).numpy().flatten()

mse = np.mean((field_true - pred_all)**2)
print(f"Equivariant model MSE: {mse:.6f}")

# Compare with SIREN from earlier (load from saved)
# (assuming you have the SIREN field stored)
siren_field = results["generated_ds"]
mse_siren = np.mean((field_true - siren_field)**2)
print(f"SIREN MSE: {mse_siren:.6f}")

# Plot comparison
fig, axes = plt.subplots(1, 3, figsize=(15,5), subplot_kw={'projection':'3d'})
step = max(1, len(vertices_ds)//3000)
for ax, field, title in zip(axes, [field_true, siren_field, pred_all],
                             ["True", "SIREN", "Equivariant (SH+MLP)"]):
    ax.scatter(vertices_ds[::step,0], vertices_ds[::step,1], vertices_ds[::step,2],
               c=field[::step], cmap='viridis', s=2)
    ax.set_title(title)
plt.savefig(PROJECT_ROOT / "figures" / "equivariant_comparison.png", dpi=300)
print("Saved figure: figures/equivariant_comparison.png")