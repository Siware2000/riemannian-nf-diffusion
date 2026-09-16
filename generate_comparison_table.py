import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(r"F:\Deep Riemannian Diffusion Experiment")

# Load results
res_final = np.load(PROJECT_ROOT / "full_diffusion_results_final.npz")
res_hybrid = np.load(PROJECT_ROOT / "full_diffusion_results_hybrid.npz")
res_equiv = np.load(PROJECT_ROOT / "full_diffusion_results_equivariant_full.npz")

mse_siren = np.mean((res_final["field_ds"] - res_final["generated_ds"])**2)
mse_hybrid = np.mean((res_hybrid["field_ds"] - res_hybrid["generated_ds"])**2)
mse_equiv = np.mean((res_equiv["field_ds"] - res_equiv["generated_ds"])**2)

# IDW held-out MSE
mse_idw = 0.1858   # from your run
# Neural Field on held-out electrodes (equivariant)
mse_nf_idw = 0.0489  # from your baseline script

# GCN MSE – replace with the value from the corrected gcn script
mse_gcn = 0.05  # <-- REPLACE THIS with actual GCN test MSE

data = {
    "Model": ["SIREN", "Hybrid (SH+SIREN)", "Equivariant (SH+MLP)", "IDW (held-out)", "GCN (held-out)"],
    "MSE": [mse_siren, mse_hybrid, mse_equiv, mse_idw, mse_gcn],
    "Note": ["Full surface", "Full surface", "Full surface", "Held-out electrodes", "Held-out electrodes"]
}

df = pd.DataFrame(data)
print(df.to_string(index=False))

# LaTeX table
latex = df.to_latex(index=False, column_format="lcc", caption="Model comparison", label="tab:comparison")
print("\nLaTeX Table:\n", latex)