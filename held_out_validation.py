"""
Validate the Neural Field by leaving out a random subset of electrodes.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(r"F:\Deep Riemannian Diffusion Experiment")
RESULTS_FILE = PROJECT_ROOT / "full_diffusion_results_gpu_dense.npz"
MAPPING_FILE = PROJECT_ROOT / "electrode_mapping.npz"

# Load results
data = np.load(RESULTS_FILE, allow_pickle=True)
vertices = data["vertices_ds"]
field_nf = data["generated_ds"]
field_mean = data["field_ds"]

# Load original electrode coordinates
mapping = np.load(MAPPING_FILE)
elec_coords = mapping["electrode_coords"]
n_elec = len(elec_coords)

# Find nearest downsampled vertex for each electrode
tree = cKDTree(vertices)
_, idx_ds = tree.query(elec_coords, k=1)

# Extract the true field at those vertices (from the mean field)
true_values = field_mean[idx_ds]

# Extract the Neural Field predictions at those vertices
pred_values = field_nf[idx_ds]

# Randomly hold out 20% of electrodes
np.random.seed(42)
held_out_mask = np.random.rand(n_elec) < 0.2
held_idx = np.where(held_out_mask)[0]
kept_idx = np.where(~held_out_mask)[0]

# Compute errors
mse_all = np.mean((true_values - pred_values)**2)
mse_held = np.mean((true_values[held_idx] - pred_values[held_idx])**2)
mse_kept = np.mean((true_values[kept_idx] - pred_values[kept_idx])**2)

print("="*60)
print("📊 HELD-OUT ELECTRODE VALIDATION")
print("="*60)
print(f"Total electrodes: {n_elec}")
print(f"Held out: {len(held_idx)} ({len(held_idx)/n_elec*100:.1f}%)")
print(f"MSE (all electrodes): {mse_all:.6f}")
print(f"MSE (held out): {mse_held:.6f}")
print(f"MSE (kept): {mse_kept:.6f}")
print(f"ΔMSE (held - kept): {mse_held - mse_kept:.6f}")

# Plot: predicted vs true for held-out electrodes
plt.figure(figsize=(8, 6))
plt.scatter(true_values[kept_idx], pred_values[kept_idx], c='blue', alpha=0.5, label='Kept')
plt.scatter(true_values[held_idx], pred_values[held_idx], c='red', s=60, label='Held Out')
plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], 'k--')
plt.xlabel("True Mean Field")
plt.ylabel("Neural Field Prediction")
plt.title(f"Held-Out Validation (20% held out)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(PROJECT_ROOT / "figures" / "held_out_validation.png", dpi=300)
plt.show()
print("💾 Saved figure: figures/held_out_validation.png")