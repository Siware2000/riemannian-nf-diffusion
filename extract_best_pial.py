"""
Extract the largest lh.pial and rh.pial from tpl-onavg_freesurfer.zip
"""
import zipfile
from pathlib import Path
import shutil

PROJECT_ROOT = Path(r"F:\Deep Riemannian Diffusion Experiment")
ZIP_PATH = PROJECT_ROOT / "tpl-onavg_freesurfer.zip"
OUTPUT_DIR = PROJECT_ROOT / "templates" / "onavg"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not ZIP_PATH.exists():
    print(f"❌ Zip not found: {ZIP_PATH}")
    exit(1)

print("📂 Scanning zip for .pial files...")
best_lh = None
best_rh = None
best_lh_size = 0
best_rh_size = 0

with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
    for member in zf.namelist():
        if member.endswith('lh.pial'):
            info = zf.getinfo(member)
            if info.file_size > best_lh_size:
                best_lh = member
                best_lh_size = info.file_size
        elif member.endswith('rh.pial'):
            info = zf.getinfo(member)
            if info.file_size > best_rh_size:
                best_rh = member
                best_rh_size = info.file_size

if best_lh is None or best_rh is None:
    print("❌ Could not find .pial files in zip.")
    exit(1)

print(f"✅ Found LH: {best_lh} ({best_lh_size / (1024*1024):.2f} MB)")
print(f"✅ Found RH: {best_rh} ({best_rh_size / (1024*1024):.2f} MB)")

# Extract them
with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
    for member in [best_lh, best_rh]:
        data = zf.read(member)
        filename = Path(member).name
        target = OUTPUT_DIR / filename
        with open(target, 'wb') as f:
            f.write(data)
        print(f"✅ Extracted {filename} to {target}")

print("🎉 Done! Now run the electrode pipeline:")
print("python run_electrode_final.py")