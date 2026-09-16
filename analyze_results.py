"""
Post‑processing script for the Riemannian Diffusion Neural Field results.
Loads the .npz file, computes errors, and saves high‑quality figures.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore")

# ==============================================
# 1. CONFIGURATION
# ==============================================
PROJECT_ROOT = Path(r"F:\Deep Riemannian Diffusion Experiment")
RESULTS_FILE = PROJECT_ROOT / "full_diffusion_results_gpu_dense.npz"
OUTPUT_FIG_DIR = PROJECT_ROOT / "figures"
OUTPUT_FIG_DIR.mkdir(exist_ok=True)

# ==============================================
# 2. LOAD RESULTS
# ==============================================
if not RESULTS_FILE.exists():
    raise FileNotFoundError(f"Results file not found: {RESULTS_FILE}")

data = np.load(RESULTS_FILE, allow_pickle=True)
vertices = data["vertices_ds"]          # (N, 3)
field_mean = data["field_ds"]           # (N,) ground truth mean
field_nf = data["generated_ds"]        # (N,) Neural Field output
field_diffusion = data["generated_diffusion"]  # (N,) diffusion sample
evals = data["evals"]
print(f"✅ Loaded results from {RESULTS_FILE}")
print(f"   Vertices: {len(vertices)}")
print(f"   Eigenvalues: {len(evals)} modes")
print(f"   MSE (Neural Field vs Mean): {np.mean((field_mean - field_nf)**2):.6f}")
print(f"   MAE (Neural Field vs Mean): {np.mean(np.abs(field_mean - field_nf)):.6f}")

# ==============================================
# 3. COMPUTE ERRORS
# ==============================================
abs_error = np.abs(field_mean - field_nf)
max_abs = np.max(np.abs(field_mean))
rel_error = abs_error / (max_abs + 1e-8)

print(f"   Max absolute error: {np.max(abs_error):.4f}")
print(f"   Mean relative error: {np.mean(rel_error)*100:.2f}%")

# ==============================================
# 4. CREATE PUBLICATION‑READY FIGURES (3D)
# ==============================================
def plot_surface(ax, vertices, values, title, cmap="viridis", vmin=None, vmax=None):
    step = max(1, len(vertices) // 3000)
    sc = ax.scatter(
        vertices[::step, 0],
        vertices[::step, 1],
        vertices[::step, 2],
        c=values[::step],
        cmap=cmap,
        s=2,
        alpha=0.9,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")
    return sc

# Figure 1: Four subplots
fig = plt.figure(figsize=(20, 6))

# Common color scale for the first two plots
vmin = min(field_mean.min(), field_nf.min())
vmax = max(field_mean.max(), field_nf.max())

# Mean Field
ax1 = fig.add_subplot(141, projection="3d")
sc1 = plot_surface(ax1, vertices, field_mean, "Mean Field (iEEG)", "viridis", vmin, vmax)
plt.colorbar(sc1, ax=ax1, fraction=0.04, pad=0.1)

# Neural Field Reconstruction
ax2 = fig.add_subplot(142, projection="3d")
sc2 = plot_surface(ax2, vertices, field_nf, "Neural Field", "plasma", vmin, vmax)
plt.colorbar(sc2, ax=ax2, fraction=0.04, pad=0.1)

# Absolute Error
ax3 = fig.add_subplot(143, projection="3d")
vmin_err = 0
vmax_err = np.percentile(abs_error, 95)  # avoid outliers
sc3 = plot_surface(ax3, vertices, abs_error, f"Absolute Error (max = {np.max(abs_error):.3f})", "hot", vmin_err, vmax_err)
plt.colorbar(sc3, ax=ax3, fraction=0.04, pad=0.1)

# Diffusion Sample
ax4 = fig.add_subplot(144, projection="3d")
vmin_d = field_diffusion.min()
vmax_d = field_diffusion.max()
sc4 = plot_surface(ax4, vertices, field_diffusion, "Diffusion Sample", "magma", vmin_d, vmax_d)
plt.colorbar(sc4, ax=ax4, fraction=0.04, pad=0.1)

plt.tight_layout()
save_path = OUTPUT_FIG_DIR / "comparison_3d.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"💾 Saved 3D comparison figure: {save_path}")
# plt.show()  # uncomment if you want to see the figure interactively

# ==============================================
# 5. SIDE‑BY‑SIDE 2D PROJECTIONS (optional, good for papers)
# ==============================================
# Use a simple orthographic projection (x‑y plane) for a clean view
fig2, axes = plt.subplots(1, 4, figsize=(20, 5))
titles = ["Mean Field", "Neural Field", "Absolute Error", "Diffusion Sample"]
fields = [field_mean, field_nf, abs_error, field_diffusion]
cmaps = ["viridis", "plasma", "hot", "magma"]

for ax, f, title, cmap in zip(axes, fields, titles, cmaps):
    sc = ax.scatter(
        vertices[:, 0],
        vertices[:, 1],
        c=f,
        cmap=cmap,
        s=1,
        alpha=0.8,
        vmin=vmin if title == "Mean Field" or title == "Neural Field" else None,
        vmax=vmax if title == "Mean Field" or title == "Neural Field" else None,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")
    plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.05)

plt.tight_layout()
save_path2 = OUTPUT_FIG_DIR / "comparison_2d.png"
plt.savefig(save_path2, dpi=300, bbox_inches="tight")
print(f"💾 Saved 2D comparison figure: {save_path2}")

# ==============================================
# 6. PRINT STATISTICS FOR THE PAPER
# ==============================================
print("\n" + "="*60)
print("📊 STATISTICS FOR YOUR PAPER")
print("="*60)
print(f"Number of surface vertices: {len(vertices):,}")
print(f"Number of eigenmodes used: {len(evals)}")
print(f"Eigenvalue range: [{evals.min():.4f}, {evals.max():.4f}]")
print(f"MSE (Neural Field vs iEEG mean): {np.mean((field_mean - field_nf)**2):.6f}")
print(f"MAE (Neural Field vs iEEG mean): {np.mean(np.abs(field_mean - field_nf)):.6f}")
print(f"Maximum absolute error: {np.max(abs_error):.6f}")
print(f"Mean relative error: {np.mean(rel_error)*100:.2f}%")

print("\n📁 Figures saved in:")
print(f"   {OUTPUT_FIG_DIR}")
print("🎉 Analysis complete!")