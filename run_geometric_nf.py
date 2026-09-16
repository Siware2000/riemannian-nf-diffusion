"""
run_geometric_nf.py (DirectML / Intel iGPU version)
Trains SIREN Neural Fields on Intel iGPU via DirectML.
"""
import numpy as np
import torch
import torch.nn as nn
import trimesh
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh
import time

# ============================================================
# Device selection: try DirectML (Intel iGPU) -> CUDA -> CPU
# ============================================================
DEVICE = None
try:
    import torch_directml
    DEVICE = torch_directml.device()
    print(f"✅ Using DirectML (Intel iGPU): {DEVICE}")
except ImportError:
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print(f"✅ Using CUDA: {DEVICE}")
    else:
        DEVICE = torch.device("cpu")
        print(f"⚠️  DirectML not installed. Using CPU (slower).")
        print(f"    Install with: pip install torch-directml")

# ============================================================
# Load data
# ============================================================
print("\nLoading data...")
data = np.load("full_diffusion_results_sh_augmented.npz")
vertices = data["vertices_ds"].astype(np.float32)
faces    = data["faces_ds"].astype(np.int64)
target   = data["field_ds"].astype(np.float32)
print(f"Vertices: {vertices.shape}, Faces: {faces.shape}, Target: {target.shape}")

# ============================================================
# Geometric features
# ============================================================
def compute_normals(vertices, faces):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh.vertex_normals.astype(np.float32)

def compute_curvatures(vertices, faces):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    gauss = mesh.vertex_defects.astype(np.float32)
    mean = np.zeros(len(vertices), dtype=np.float32)
    return mean, gauss

print("\nComputing geometric features...")
t0 = time.time()
normals = compute_normals(vertices, faces)
print(f"  Normals: {normals.shape}  ({time.time()-t0:.1f}s)")

t0 = time.time()
mean_c, gauss_c = compute_curvatures(vertices, faces)
print(f"  Curvatures: {mean_c.shape}  ({time.time()-t0:.1f}s)")

# Use precomputed eigenvectors from your .npz
eigenfuncs = data["evecs"].astype(np.float32)
print(f"  Using precomputed eigenvectors: {eigenfuncs.shape}")

features_xyz = vertices
features_full = np.concatenate([
    vertices, normals, mean_c[:, None], gauss_c[:, None], eigenfuncs,
], axis=1).astype(np.float32)

print(f"\nFeature matrix (xyz only):  {features_xyz.shape}")
print(f"Feature matrix (geometric): {features_full.shape}")

# ============================================================
# SIREN Neural Field
# ============================================================
class SineLayer(nn.Module):
    def __init__(self, in_f, out_f, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_f, out_f)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1/in_f, 1/in_f)
            else:
                bound = np.sqrt(6/in_f) / omega_0
                self.linear.weight.uniform_(-bound, bound)
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class SIRENField(nn.Module):
    def __init__(self, in_dim, hidden=128, layers=4, omega_0=30.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SineLayer(in_dim, hidden, is_first=True, omega_0=omega_0))
        for _ in range(layers - 2):
            self.layers.append(SineLayer(hidden, hidden, omega_0=omega_0))
        self.final = nn.Linear(hidden, 1)
        with torch.no_grad():
            bound = np.sqrt(6/hidden) / omega_0
            self.final.weight.uniform_(-bound, bound)
    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return self.final(x).squeeze(-1)

def train_nf(features, target, train_mask, val_mask,
             hidden=128, layers=4, epochs=1000, lr=1e-3, batch=2048):
    # Convert to tensors on DirectML device
    X = torch.tensor(features, dtype=torch.float32).to(DEVICE)
    y = torch.tensor(target, dtype=torch.float32).to(DEVICE)
    tr = torch.tensor(train_mask).to(DEVICE)
    va = torch.tensor(val_mask).to(DEVICE)

    model = SIRENField(features.shape[1], hidden, layers).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    Xtr, ytr = X[tr], y[tr]
    Xva, yva = X[va], y[va]

    best = float("inf")
    best_state = None
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), device="cpu").to(DEVICE)
        for i in range(0, len(Xtr), batch):
            idx = perm[i:i+batch]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            vmse = nn.functional.mse_loss(model(Xva), yva).item()
        if vmse < best:
            best = vmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep % 200 == 0 or ep == epochs - 1:
            print(f"    Epoch {ep:4d}: val MSE = {vmse:.6e}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        full_pred = model(X).cpu().numpy()
    return model, best, full_pred, time.time() - t0

# ============================================================
# Train/test split
# ============================================================
np.random.seed(42)
n = len(vertices)
perm = np.random.permutation(n)
train_idx = perm[:int(0.8*n)]
val_idx   = perm[int(0.8*n):int(0.9*n)]
test_idx  = perm[int(0.9*n):]

train_mask = np.zeros(n, dtype=bool); train_mask[train_idx] = True
val_mask   = np.zeros(n, dtype=bool); val_mask[val_idx]     = True
test_mask  = np.zeros(n, dtype=bool); test_mask[test_idx]   = True

# ============================================================
# Train both models
# ============================================================
print("\n" + "=" * 60)
print("TRAINING 1/2: SIREN with xyz only")
print("=" * 60)
model_xyz, val_xyz, pred_xyz, t_xyz = train_nf(
    features_xyz, target, train_mask, val_mask, epochs=1000
)
test_mse_xyz = float(np.mean((pred_xyz[test_mask] - target[test_mask])**2))
print(f"✅ Test MSE (xyz): {test_mse_xyz:.6e}   ({t_xyz:.1f}s)")

print("\n" + "=" * 60)
print("TRAINING 2/2: Geometric SIREN (xyz + normals + curvatures + eigenfuncs)")
print("=" * 60)
model_geo, val_geo, pred_geo, t_geo = train_nf(
    features_full, target, train_mask, val_mask, epochs=1000
)
test_mse_geo = float(np.mean((pred_geo[test_mask] - target[test_mask])**2))
print(f"✅ Test MSE (geometric): {test_mse_geo:.6e}   ({t_geo:.1f}s)")

# ============================================================
# Save results
# ============================================================
out = {
    "test_mse_xyz": test_mse_xyz,
    "test_mse_geo": test_mse_geo,
    "pred_xyz": pred_xyz.astype(np.float32),
    "pred_geo": pred_geo.astype(np.float32),
    "train_mask": train_mask,
    "val_mask": val_mask,
    "test_mask": test_mask,
    "improvement_pct": 100 * (test_mse_xyz - test_mse_geo) / test_mse_xyz,
}
np.savez("geometric_nf_results.npz", **out)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"SIREN (xyz only)         test MSE: {test_mse_xyz:.6e}")
print(f"Geometric SIREN (ours)   test MSE: {test_mse_geo:.6e}")
print(f"Improvement: {out['improvement_pct']:.2f}%")
print(f"\n✅ Saved: geometric_nf_results.npz")