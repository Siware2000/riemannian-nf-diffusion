"""
Deep Riemannian Diffusion - Final Pipeline
Handles NaN electrodes, downloads full-resolution onavg surface.
"""

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.spatial import cKDTree
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore")

# ==============================================
# CONFIGURATION
# ==============================================
PROJECT_ROOT = Path(r"F:\Deep Riemannian Diffusion Experiment")
SURFACE_DIR = PROJECT_ROOT / "templates" / "onavg"
HEMI = "lh"

# Full dataset root (the one with the long UUID)
FULL_BIDS_ROOT = PROJECT_ROOT / "Data Set" / "20231231_cog_exp1_bids_ecog_126b5c55-2a56-4fa6-aa98-f20f8e6ed9ec" / "mnt" / "beegfs" / "workspace" / "2023-0385-Cogitatedatarelease" / "CURATE" / "COG_ECOG_EXP1_BIDS"

def find_bids_root(root):
    for path in root.rglob("participants.tsv"):
        return path.parent
    return None

def ensure_full_surface(surface_dir, hemi):
    """Try to get the full-resolution surface via templateflow."""
    # First, check if we already have a full-res file (size > 5 MB)
    pial_file = surface_dir / f"{hemi}.pial"
    if pial_file.exists() and pial_file.stat().st_size > 5 * 1024 * 1024:
        print(f"✅ Full-resolution {hemi}.pial already exists.")
        return pial_file

    # Try downloading via templateflow
    try:
        from templateflow import api as tf
        print(f"🔄 Attempting to download full-resolution {hemi} surface via templateflow...")
        # This returns a list of paths; take the first one
        downloaded = tf.get('onavg', hemi=hemi.upper(), suffix='pial', resolution=None)
        if downloaded:
            src = Path(downloaded[0])
            if src.exists() and src.stat().st_size > 5 * 1024 * 1024:
                # Copy to our surface_dir
                surface_dir.mkdir(parents=True, exist_ok=True)
                dest = surface_dir / f"{hemi}.pial"
                import shutil
                shutil.copy2(src, dest)
                print(f"✅ Downloaded and copied full-resolution {hemi}.pial")
                return dest
    except Exception as e:
        print(f"⚠️ Could not download full surface: {e}")

    # Fallback: use existing (maybe low-res) and warn
    if pial_file.exists():
        print(f"⚠️ Using existing {hemi}.pial (low resolution). This may affect accuracy.")
        return pial_file

    print(f"❌ No surface file found for {hemi}.")
    return None

# ==============================================
# 1. Find BIDS root
# ==============================================
bids_root = FULL_BIDS_ROOT if FULL_BIDS_ROOT.exists() else find_bids_root(PROJECT_ROOT)
if bids_root is None:
    print("❌ BIDS root not found.")
    exit(1)
print(f"✅ Using BIDS root: {bids_root}")

# ==============================================
# 2. Find first subject and session
# ==============================================
subjects = [d for d in bids_root.iterdir() if d.is_dir() and d.name.startswith("sub-")]
if not subjects:
    print("❌ No subjects found.")
    exit(1)
subject = subjects[0].name
sessions = [d for d in subjects[0].iterdir() if d.is_dir() and d.name.startswith("ses-")]
session = sessions[0].name if sessions else None
print(f"   Subject: {subject}")
if session:
    print(f"   Session: {session}")

# ==============================================
# 3. Load electrodes, drop NaN
# ==============================================
if session:
    elec_file = bids_root / subject / session / "ieeg" / f"{subject}_{session}_space-fsaverage_electrodes.tsv"
else:
    elec_file = bids_root / subject / "ieeg" / f"{subject}_space-fsaverage_electrodes.tsv"

if not elec_file.exists():
    print(f"❌ Electrodes file not found: {elec_file}")
    exit(1)

df = pd.read_csv(elec_file, sep="\t")
if not all(c in df.columns for c in ["x", "y", "z"]):
    print(f"❌ No x,y,z columns in {elec_file}")
    print(f"   Columns: {list(df.columns)}")
    exit(1)

# Drop rows with NaN in x,y,z
original_len = len(df)
df_clean = df.dropna(subset=["x", "y", "z"])
coords = df_clean[["x", "y", "z"]].values
print(f"✅ Loaded {len(coords)} electrodes (dropped {original_len - len(coords)} rows with NaN)")

# ==============================================
# 4. Ensure full-resolution surface
# ==============================================
surface_file = ensure_full_surface(SURFACE_DIR, HEMI)
if surface_file is None:
    print("❌ Cannot proceed without surface.")
    exit(1)

vertices, faces = nib.freesurfer.read_geometry(str(surface_file))
print(f"✅ Surface: {len(vertices):,} vertices, {len(faces):,} faces")
if len(vertices) < 10000:
    print("⚠️ WARNING: Low-resolution surface detected. Riemannian metrics may be inaccurate.")

# ==============================================
# 5. Project electrodes
# ==============================================
tree = cKDTree(vertices)
distances, vertex_indices = tree.query(coords, k=1)
projected = vertices[vertex_indices]
print(f"✅ Projection: mean distance {distances.mean():.2f} mm")

# ==============================================
# 6. Plot
# ==============================================
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')
step = max(1, len(vertices) // 3000)
ax.scatter(vertices[::step, 0], vertices[::step, 1], vertices[::step, 2],
           c='lightgrey', s=1, alpha=0.3)
ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
           c='red', s=60, label='Electrodes (original)')
ax.scatter(projected[:, 0], projected[:, 1], projected[:, 2],
           c='blue', s=30, label='Projected onto surface')
for i in range(len(coords)):
    ax.plot([coords[i,0], projected[i,0]],
            [coords[i,1], projected[i,1]],
            [coords[i,2], projected[i,2]],
            'g-', linewidth=0.5, alpha=0.3)
ax.set_title(f"{subject} - {HEMI} on onavg (full dataset)")
ax.legend()
plt.show()

# ==============================================
# 7. Save mapping
# ==============================================
np.savez("electrode_mapping.npz",
         electrode_coords=coords,
         vertex_indices=vertex_indices,
         projected_coords=projected,
         distances_mm=distances)
print("💾 Saved electrode_mapping.npz")
print("🎉 Done!")