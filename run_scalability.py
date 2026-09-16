"""
Run the diffusion pipeline with different surface resolutions.
Saves results to separate files to avoid overwriting.
"""
import time
import subprocess
import sys
import numpy as np
import re
from pathlib import Path

PROJECT_ROOT = Path(r"F:\Deep Riemannian Diffusion Experiment")
PIPELINE_SCRIPT = PROJECT_ROOT / "diffusion_pipeline_final.py"
RESOLUTIONS = [5000, 10000, 20000]
results = []

for n_vertices in RESOLUTIONS:
    print(f"\n🚀 Running pipeline with TARGET_VERTICES = {n_vertices}")
    start = time.time()
    # Read and modify script
    with open(PIPELINE_SCRIPT, 'r', encoding='utf-8') as f:
        content = f.read()
    # Change TARGET_VERTICES and the output filename
    content = re.sub(r'TARGET_VERTICES\s*=\s*\d+', f'TARGET_VERTICES = {n_vertices}', content)
    # Change the save line to a unique filename
    content = re.sub(r'full_diffusion_results_final\.npz', f'full_diffusion_results_final_{n_vertices}.npz', content)
    temp_script = PROJECT_ROOT / f"temp_pipeline_{n_vertices}.py"
    temp_script.write_text(content, encoding='utf-8')
    # Run
    result = subprocess.run([sys.executable, str(temp_script)], capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - start
    # Load the saved results
    res_file = PROJECT_ROOT / f"full_diffusion_results_final_{n_vertices}.npz"
    if res_file.exists():
        res = np.load(res_file)
        field_true = res["field_ds"]
        field_pred = res["generated_ds"]
        mse = np.mean((field_true - field_pred)**2)
    else:
        mse = None
    results.append((n_vertices, mse, elapsed))
    temp_script.unlink()
    print(f"✅ Completed {n_vertices} vertices in {elapsed:.2f}s, MSE = {mse}")

print("\n📊 Scalability Results")
print("---------------------")
print("Vertices | MSE      | Time (s)")
for n, mse, t in results:
    if mse is not None:
        print(f"{n:8d} | {mse:.6f} | {t:.2f}")
    else:
        print(f"{n:8d} | N/A      | {t:.2f}")