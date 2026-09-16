import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path

PROJECT_ROOT = Path(r"F:\Deep Riemannian Diffusion Experiment")
RESULTS_FILE = PROJECT_ROOT / "full_diffusion_results_gpu_dense.npz"
OUTPUT_FILE = PROJECT_ROOT / "diffusion_movie.gif"   # changed to .gif

data = np.load(RESULTS_FILE, allow_pickle=True)
vertices = data["vertices_ds"]
evecs = data["evecs"]
states = data["diffusion_states"]
print(f"✅ Loaded {len(states)} diffusion states")

fields = [evecs @ state for state in states]
vmin = min(np.min(f) for f in fields)
vmax = max(np.max(f) for f in fields)

fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': '3d'})
step = max(1, len(vertices) // 3000)

def update(frame):
    ax.clear()
    sc = ax.scatter(vertices[::step,0], vertices[::step,1], vertices[::step,2],
                    c=fields[frame][::step], cmap='magma', s=2, alpha=0.9,
                    vmin=vmin, vmax=vmax)
    ax.set_title(f"Diffusion Step {frame}/{len(states)-1}", fontsize=14)
    ax.axis('off')
    return sc,

ani = FuncAnimation(fig, update, frames=len(states), interval=200, blit=False)
# Use pillow writer (supports GIF)
ani.save(OUTPUT_FILE, writer='pillow', fps=5, dpi=150)
print(f"💾 Movie saved as GIF: {OUTPUT_FILE}")