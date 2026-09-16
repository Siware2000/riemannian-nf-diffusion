"""
Hybrid Pipeline – SIREN with spherical harmonics features.
Uses e3nn only to compute SH, then standard SIREN MLP.
Runs on DirectML (iGPU).
"""
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mne
from mne_bids import BIDSPath, read_raw_bids
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ==============================================
# OPTIONAL: TRY TO IMPORT E3NN FOR SPHERICAL HARMONICS
# ==============================================
try:
    import e3nn
    from e3nn import o3
    HAS_E3NN = True
    print("✅ e3nn available – using spherical harmonics features.")
except ImportError:
    HAS_E3NN = False
    print("⚠️ e3nn not installed – falling back to plain coordinates.")

try:
    import pyvista as pv
    HAS_PV = True
except ImportError:
    HAS_PV = False
    print("⚠️ pyvista not installed. Install: pip install pyvista")

# ==============================================
# 0. DIRECTML DEVICE
# ==============================================
try:
    import torch_directml
    device = torch_directml.device()
    print(f"✅ Using DirectML device: {device}")
    x = torch.ones(10, 10).to(device)
    print(f"✅ Tensor test successful – tensor is on {x.device}")
except ImportError:
    print("⚠️ torch-directml not installed. Falling back to CPU.")
    device = torch.device('cpu')
except Exception as e:
    print(f"⚠️ Could not initialise DirectML: {e}. Using CPU.")
    device = torch.device('cpu')
print(f"🔧 Final device for training: {device}")

# ==============================================
# 1. CONFIGURATION
# ==============================================
PROJECT_ROOT = Path(r"F:\Deep Riemannian Diffusion Experiment")
SURFACE_DIR = PROJECT_ROOT / "templates" / "onavg"
BIDS_ROOT = PROJECT_ROOT / "Data Set" / "20231231_cog_exp1_bids_ecog_126b5c55-2a56-4fa6-aa98-f20f8e6ed9ec" / "mnt" / "beegfs" / "workspace" / "2023-0385-Cogitatedatarelease" / "CURATE" / "COG_ECOG_EXP1_BIDS"
HEMI = "lh"
SUBJECT = "CG105"
SESSION = "1"

TARGET_VERTICES = 10000
K_MODES = 30
HIDDEN_DIM = 128
NUM_LAYERS = 3
EPOCHS = 50
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3

# ==============================================
# 2. LOAD SURFACE AND MAPPING
# ==============================================
surface_file = SURFACE_DIR / f"{HEMI}.pial"
vertices_full, faces_full = nib.freesurfer.read_geometry(str(surface_file))
n_vertices_full = len(vertices_full)
print(f"✅ Full surface: {n_vertices_full} vertices, {len(faces_full)} faces")

mapping = np.load("electrode_mapping.npz")
elec_coords = mapping["electrode_coords"]
n_elec = len(elec_coords)
print(f"✅ Mapping: {n_elec} electrodes")

# ==============================================
# 3. LOAD iEEG DATA (mean field)
# ==============================================
print("📂 Loading iEEG data...")
try:
    bids_path = BIDSPath(
        subject=SUBJECT, session=SESSION, task="Dur", suffix="ieeg",
        extension=".edf", root=BIDS_ROOT, datatype="ieeg"
    )
    if not bids_path.fpath.exists():
        bids_path = bids_path.copy().update(extension=".vhdr")
    raw = read_raw_bids(bids_path, verbose=False)
    sfreq = raw.info['sfreq']
    data, times = raw[:, :]
    signal_data = data[:n_elec, :]
    electrode_mean = signal_data.mean(axis=1)
    print(f"✅ Loaded iEEG: {signal_data.shape[0]} channels, mean computed")
except Exception as e:
    print(f"⚠️ Could not load iEEG: {e}. Using synthetic.")
    t = np.linspace(0, 10, 1000)
    signal_data = np.sin(2 * np.pi * 0.5 * t) * 5 + np.random.randn(n_elec, 1000) * 0.5
    electrode_mean = signal_data.mean(axis=1)
    sfreq = 100
    print("✅ Using synthetic data.")

# ==============================================
# 4. DOWNSAMPLE SURFACE
# ==============================================
if not HAS_PV:
    raise ImportError("pyvista required. Install: pip install pyvista")
faces_pv = np.hstack([np.full((len(faces_full), 1), 3), faces_full]).astype(np.int64)
mesh = pv.PolyData(vertices_full, faces_pv)
reduction = 1.0 - TARGET_VERTICES / n_vertices_full
decimated = mesh.decimate(target_reduction=reduction, inplace=False)
vertices_ds = decimated.points.astype(np.float64)
faces_ds = decimated.faces.reshape(-1, 4)[:, 1:].astype(np.int64)
print(f"✅ Downsampled surface: {len(vertices_ds)} vertices")

# ==============================================
# 5. PROJECT ELECTRODES TO DOWNSAMPLED SURFACE
# ==============================================
from scipy.spatial import cKDTree
tree_ds = cKDTree(vertices_ds)
_, idx_ds = tree_ds.query(elec_coords, k=1)
field_ds = np.zeros(len(vertices_ds))
for i, idx in enumerate(idx_ds):
    field_ds[idx] += electrode_mean[i]
unique_idx, counts = np.unique(idx_ds, return_counts=True)
for idx, cnt in zip(unique_idx, counts):
    field_ds[idx] /= cnt
print(f"✅ Interpolated mean field (shape: {field_ds.shape})")

# ==============================================
# 6. COTANGENT LAPLACIAN (CPU)
# ==============================================
def compute_cotangent_laplacian(vertices, faces):
    nv = len(vertices)
    rows, cols, vals = [], [], []
    for face in tqdm(faces, desc="Building cotangent Laplacian"):
        i, j, k = face
        vi, vj, vk = vertices[i], vertices[j], vertices[k]
        e1 = vi - vk; e2 = vj - vk
        cross = np.cross(e1, e2)
        area = 0.5 * np.linalg.norm(cross)
        if area < 1e-12: continue
        cot_angle = np.dot(e1, e2) / (2.0 * area + 1e-12)
        rows.extend([i, j]); cols.extend([j, i]); vals.extend([-cot_angle, -cot_angle])
        rows.extend([i, j, k]); cols.extend([i, j, k]); vals.extend([cot_angle, cot_angle, cot_angle])
    L = coo_matrix((vals, (rows, cols)), shape=(nv, nv)).tocsr()
    L = (L + L.T) / 2
    return L

print("🔍 Computing cotangent Laplacian (CPU)...")
L_ds = compute_cotangent_laplacian(vertices_ds, faces_ds)
print(f"✅ Laplacian: {L_ds.nnz} non-zero entries")

# ==============================================
# 7. EIGENMODES ON GPU (Dense Tensor)
# ==============================================
print(f"🔍 Computing first {K_MODES} eigenmodes on GPU via dense tensor...")
L_dense_np = L_ds.toarray().astype(np.float32)
L_torch = torch.tensor(L_dense_np, device=device)

try:
    evals, evecs = torch.linalg.eigh(L_torch)
    idx_sort = torch.argsort(evals)[:K_MODES]
    evals = evals[idx_sort].cpu().numpy()
    evecs = evecs[:, idx_sort].cpu().numpy()
    print(f"✅ Eigenmodes computed (GPU): λ[0]={evals[0]:.4f}, λ[-1]={evals[-1]:.4f}")
except Exception as e:
    print(f"⚠️ GPU eigenmode failed: {e}. Falling back to SciPy (CPU).")
    from scipy.sparse.linalg import eigsh
    evals, evecs = eigsh(L_ds, k=K_MODES, which='SM', sigma=0.0)
    idx_sort = np.argsort(evals)
    evals = evals[idx_sort]
    evecs = evecs[:, idx_sort]

# ==============================================
# 8. PROJECT FIELD ONTO EIGENBASIS
# ==============================================
coeffs = evecs.T @ field_ds
print(f"✅ Coefficients shape: {coeffs.shape}")

# ==============================================
# 9. HYBRID NEURAL FIELD: SIREN + SPHERICAL HARMONICS
# ==============================================
if HAS_E3NN:
    # Precompute spherical harmonics function (on the CPU, will be moved to GPU during training)
    max_degree = 4
    irreps_sh = o3.Irreps.spherical_harmonics(max_degree)
    sh_func = o3.SphericalHarmonics(irreps_sh, normalize=True, normalization='component')
    sh_dim = irreps_sh.dim
    input_dim = 3 + sh_dim   # x,y,z + SH features
    print(f"✅ Using spherical harmonics features (degree {max_degree}, dim {sh_dim})")
else:
    input_dim = 3
    print("⚠️ e3nn not available – using plain coordinates.")

# SIREN model with variable input dimension
class SirenLayer(nn.Module):
    def __init__(self, in_dim, out_dim, omega=30.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.omega = omega
        nn.init.uniform_(self.linear.weight, -1/out_dim, 1/out_dim)
        nn.init.uniform_(self.linear.bias, -1/out_dim, 1/out_dim)
    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))

class HybridField(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3):
        super().__init__()
        layers = [SirenLayer(input_dim, hidden_dim, omega=30.0)]
        for _ in range(num_layers - 2):
            layers.append(SirenLayer(hidden_dim, hidden_dim, omega=30.0))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# Build model
model = HybridField(input_dim=input_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# Prepare training data
if HAS_E3NN:
    # Compute SH for all vertices (on CPU, then move to GPU in batches)
    sh_all = sh_func(torch.tensor(vertices_ds, dtype=torch.float32))
    sh_all = sh_all.numpy()
else:
    sh_all = None

n_train = min(len(vertices_ds), 5000)
sample_idx = np.random.choice(len(vertices_ds), n_train, replace=False)

if HAS_E3NN:
    X_train = np.concatenate([vertices_ds[sample_idx], sh_all[sample_idx]], axis=1).astype(np.float32)
else:
    X_train = vertices_ds[sample_idx].astype(np.float32)
y_train = field_ds[sample_idx].astype(np.float32).reshape(-1, 1)

X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
dataset = TensorDataset(X_train_t, y_train_t)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"🏋️ Training Hybrid Neural Field on {device}...")
loss_history = []
for epoch in tqdm(range(EPOCHS), desc="Training"):
    epoch_loss = 0.0
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    loss_history.append(epoch_loss / len(dataloader))
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss = {loss_history[-1]:.6f}")

# Generate field for all vertices
with torch.no_grad():
    if HAS_E3NN:
        all_inputs = np.concatenate([vertices_ds, sh_all], axis=1).astype(np.float32)
    else:
        all_inputs = vertices_ds.astype(np.float32)
    all_coords = torch.tensor(all_inputs, dtype=torch.float32).to(device)
    generated_ds = model(all_coords).cpu().numpy().flatten()

# ==============================================
# 10. SCORE NETWORK & DIFFUSION (iGPU) – same as before
# ==============================================
beta_min, beta_max = 0.1, 20.0
def get_beta(t):
    return beta_min + t * (beta_max - beta_min)

class ScoreNet(nn.Module):
    def __init__(self, in_dim=K_MODES, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + 1, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, in_dim)
        )
    def forward(self, x, t):
        t_tensor = torch.full((x.size(0), 1), t, dtype=torch.float32, device=x.device)
        inp = torch.cat([x, t_tensor], dim=1)
        return self.net(inp)

score_model = ScoreNet().to(device)
optimizer_score = optim.Adam(score_model.parameters(), lr=1e-3)
criterion_score = nn.MSELoss()
coeffs_torch = torch.tensor(coeffs, dtype=torch.float32).unsqueeze(0).to(device)

loss_score_history = []
print(f"🏋️ Training Score Network on {device}...")
for epoch in tqdm(range(200), desc="Score training"):
    t_val = np.random.uniform(0, 1)
    n_samples = 128
    clean = coeffs_torch.repeat(n_samples, 1)
    noise = torch.randn_like(clean)
    beta = get_beta(t_val)
    mean = clean * np.exp(-0.5 * beta * t_val)
    std = np.sqrt(1 - np.exp(-beta * t_val))
    noisy = mean + std * noise
    pred_noise = score_model(noisy, t_val)
    loss = criterion_score(pred_noise, noise)
    optimizer_score.zero_grad()
    loss.backward()
    optimizer_score.step()
    loss_score_history.append(loss.item())
    if epoch % 50 == 0:
        print(f"Score epoch {epoch}: loss = {loss.item():.6f}")

def reverse_diffusion(score_model, num_steps=10, return_states=False):
    dt = 1.0 / num_steps
    x = torch.randn(1, K_MODES, device=device, dtype=torch.float32) * np.sqrt(1 - np.exp(-beta_max))
    states = [x.cpu().numpy().flatten()]
    for i in range(num_steps):
        t = 1.0 - i * dt
        beta = get_beta(t)
        with torch.no_grad():
            score = score_model(x, t)
            x = x + 0.5 * beta * x * dt + beta * score * dt + np.sqrt(beta * dt) * torch.randn_like(x) * 0.5
            states.append(x.cpu().numpy().flatten())
    if return_states:
        return x, states
    else:
        return x

sample_coeffs_tensor, diffusion_states = reverse_diffusion(score_model, num_steps=10, return_states=True)
sample_coeffs = sample_coeffs_tensor.cpu().numpy().flatten()
generated_diffusion = evecs @ sample_coeffs
diffusion_states_np = np.array(diffusion_states)
print(f"✅ Diffusion states collected: {diffusion_states_np.shape}")

# ==============================================
# 11. PLOTS
# ==============================================
fig = plt.figure(figsize=(18, 6))
step = max(1, len(vertices_ds) // 3000)
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(vertices_ds[::step,0], vertices_ds[::step,1], vertices_ds[::step,2],
            c=field_ds[::step], cmap='viridis', s=2)
ax1.scatter(elec_coords[:,0], elec_coords[:,1], elec_coords[:,2], c='red', s=40)
ax1.set_title("Mean Field (iEEG)")

ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(vertices_ds[::step,0], vertices_ds[::step,1], vertices_ds[::step,2],
            c=generated_ds[::step], cmap='plasma', s=2)
ax2.scatter(elec_coords[:,0], elec_coords[:,1], elec_coords[:,2], c='red', s=40)
ax2.set_title("Hybrid Field (SH + SIREN)")

ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(vertices_ds[::step,0], vertices_ds[::step,1], vertices_ds[::step,2],
            c=generated_diffusion[::step], cmap='magma', s=2)
ax3.set_title("Diffusion Sample")
plt.tight_layout()
plt.show()

# ==============================================
# 12. SAVE
# ==============================================
np.savez("full_diffusion_results_hybrid.npz",
         vertices_ds=vertices_ds, faces_ds=faces_ds,
         field_ds=field_ds, generated_ds=generated_ds,
         generated_diffusion=generated_diffusion,
         evals=evals, evecs=evecs, coeffs=coeffs,
         sample_coeffs=sample_coeffs,
         diffusion_states=diffusion_states_np,
         loss_history=loss_history, loss_score=loss_score_history,
         sfreq=sfreq)
print("💾 Saved full_diffusion_results_hybrid.npz")
print("🎉 ALL DONE!")