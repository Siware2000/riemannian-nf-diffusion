"""
Held‑out electrode comparison: IDW vs Neural Field.
Trains IDW on a subset of electrodes, tests on held‑out electrodes.
"""
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from pathlib import Path
import mne
from mne_bids import BIDSPath, read_raw_bids
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(r"F:\Deep Riemannian Diffusion Experiment")
BIDS_ROOT = PROJECT_ROOT / "Data Set" / "20231231_cog_exp1_bids_ecog_126b5c55-2a56-4fa6-aa98-f20f8e6ed9ec" / "mnt" / "beegfs" / "workspace" / "2023-0385-Cogitatedatarelease" / "CURATE" / "COG_ECOG_EXP1_BIDS"
SUBJECT = "CG105"
SESSION = "1"

# Load surface and ground truth
results = np.load("full_diffusion_results_final.npz")
vertices_ds = results["vertices_ds"]
field_true = results["field_ds"]

# Load electrode coordinates and values
mapping = np.load("electrode_mapping.npz")
elec_coords = mapping["electrode_coords"]
n_elec = len(elec_coords)

bids_path = BIDSPath(subject=SUBJECT, session=SESSION, task="Dur", suffix="ieeg",
                     extension=".edf", root=BIDS_ROOT, datatype="ieeg")
if not bids_path.fpath.exists():
    bids_path = bids_path.copy().update(extension=".vhdr")
raw = read_raw_bids(bids_path, verbose=False)
data, times = raw[:, :]
signal = data[:n_elec, :]
signal = (signal - signal.mean(axis=1, keepdims=True)) / (signal.std(axis=1, keepdims=True) + 1e-8)
electrode_mean = signal.mean(axis=1)

# Randomly hold out 20% of electrodes
np.random.seed(42)
held_out_mask = np.random.rand(n_elec) < 0.2
kept_idx = np.where(~held_out_mask)[0]
held_idx = np.where(held_out_mask)[0]

# IDW on kept electrodes
def idw_interpolate(vertices, elec_pos, elec_vals, power=2, n_neighbors=5):
    tree = cKDTree(elec_pos)
    distances, indices = tree.query(vertices, k=n_neighbors)
    distances = np.maximum(distances, 1e-12)
    weights = 1.0 / (distances ** power)
    weights = weights / weights.sum(axis=1, keepdims=True)
    return np.sum(weights * elec_vals[indices], axis=1)

kept_coords = elec_coords[kept_idx]
kept_vals = electrode_mean[kept_idx]
idw_field = idw_interpolate(vertices_ds, kept_coords, kept_vals, power=2, n_neighbors=5)

# Evaluate on held‑out electrodes
# Find nearest surface vertex for each held‑out electrode
tree = cKDTree(vertices_ds)
_, held_vtx = tree.query(elec_coords[held_idx], k=1)
true_held = field_true[held_vtx]
idw_held = idw_field[held_vtx]
nf_held = results["generated_ds"][held_vtx]

mse_idw_held = np.mean((true_held - idw_held)**2)
mse_nf_held = np.mean((true_held - nf_held)**2)

print(f"IDW MSE (held‑out electrodes): {mse_idw_held:.6f}")
print(f"Neural Field MSE (held‑out electrodes): {mse_nf_held:.6f}")
print(f"Improvement: {(mse_idw_held - mse_nf_held)/mse_idw_held*100:.2f}%")

# Scatter plot: true vs predicted for held‑out electrodes
plt.figure(figsize=(8,6))
plt.scatter(true_held, idw_held, label='IDW', alpha=0.7, s=50)
plt.scatter(true_held, nf_held, label='Neural Field', alpha=0.7, s=50)
plt.plot([min(true_held), max(true_held)], [min(true_held), max(true_held)], 'k--')
plt.xlabel("True Value")
plt.ylabel("Predicted Value")
plt.title(f"Held‑out Electrodes (n={len(held_idx)})")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(PROJECT_ROOT / "figures" / "held_out_idw_comparison.png", dpi=300)
print("Saved figure: figures/held_out_idw_comparison.png")
plt.show()