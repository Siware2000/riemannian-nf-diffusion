"""
Generative sampling using LSTM: generate new coefficient sequences from random seeds.
Runs on CPU (LSTM is not supported on DirectML).
"""
import numpy as np
import matplotlib.pyplot as plt
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
# 0. DEVICE SETUP (LSTM on CPU)
# ==============================================
device = torch.device('cpu')
print(f"🔧 Using device: {device}")

PROJECT_ROOT = Path(r"F:\Deep Riemannian Diffusion Experiment")
BIDS_ROOT = PROJECT_ROOT / "Data Set" / "20231231_cog_exp1_bids_ecog_126b5c55-2a56-4fa6-aa98-f20f8e6ed9ec" / "mnt" / "beegfs" / "workspace" / "2023-0385-Cogitatedatarelease" / "CURATE" / "COG_ECOG_EXP1_BIDS"

# Choose a subject (use the first one)
SUBJECTS = ["CG105", "CE107", "CF102", "CE110", "CF104"]
SUBJECT = SUBJECTS[0]   # change if needed
SESSION = "1"

# Load eigenbasis and surface
results = np.load("full_diffusion_results_final.npz")
evecs = results["evecs"]          # (10001, 30)
vertices_ds = results["vertices_ds"]
n_modes = evecs.shape[1]

# LSTM hyperparameters
SEQ_LEN = 20
EPOCHS = 30
BATCH_SIZE = 64
HIDDEN_DIM = 64
NUM_LAYERS = 2
N_TIME = 1000   # number of time points to sample

# ==============================================
# 1. LOAD SUBJECT DATA AND PROJECT TO EIGENBASIS
# ==============================================
def load_subject_data(bids_root, subject, session, evecs, n_time):
    """Load iEEG, interpolate to surface, project to eigenbasis."""
    bids_path = BIDSPath(
        subject=subject, session=session, task="Dur", suffix="ieeg",
        extension=".edf", root=bids_root, datatype="ieeg"
    )
    if not bids_path.fpath.exists():
        bids_path = bids_path.copy().update(extension=".vhdr")
    raw = read_raw_bids(bids_path, verbose=False)
    sfreq = raw.info['sfreq']
    data, times = raw[:, :]
    n_elec = data.shape[0]
    # Normalise per channel
    signal = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)
    # Sample time points
    time_idx = np.linspace(0, signal.shape[1]-1, n_time, dtype=int)
    sampled_signal = signal[:, time_idx]
    # Interpolate to surface (simple assignment)
    mapping = np.load("electrode_mapping.npz")
    elec_coords = mapping["electrode_coords"]
    from scipy.spatial import cKDTree
    vertices_ds = np.load("full_diffusion_results_final.npz")["vertices_ds"]
    tree = cKDTree(vertices_ds)
    _, idx_ds = tree.query(elec_coords, k=1)
    field = np.zeros((len(vertices_ds), n_time))
    for t in range(n_time):
        f = np.zeros(len(vertices_ds))
        for e, v in enumerate(idx_ds):
            f[v] += sampled_signal[e, t]
        unique, counts = np.unique(idx_ds, return_counts=True)
        for u, c in zip(unique, counts):
            f[u] /= c
        field[:, t] = f
    # Project to eigenbasis
    coeffs = evecs.T @ field   # (30, n_time)
    coeffs = coeffs.T           # (n_time, 30)
    return coeffs, sfreq, n_elec

print(f"🔍 Loading data for subject {SUBJECT}...")
coeffs, sfreq, n_elec = load_subject_data(BIDS_ROOT, SUBJECT, SESSION, evecs, N_TIME)
print(f"✅ Coefficients shape: {coeffs.shape}")

# ==============================================
# 2. LSTM MODEL
# ==============================================
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=30, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

# Prepare LSTM data
X_seq, y_seq = [], []
for i in range(SEQ_LEN, len(coeffs)):
    X_seq.append(coeffs[i-SEQ_LEN:i])
    y_seq.append(coeffs[i])
X_seq = np.array(X_seq, dtype=np.float32)
y_seq = np.array(y_seq, dtype=np.float32)
print(f"✅ LSTM data: {X_seq.shape[0]} sequences")

# Train on all data (no validation split)
model = LSTMPredictor(input_dim=n_modes, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
dataset = TensorDataset(torch.tensor(X_seq), torch.tensor(y_seq))
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("🏋️ Training LSTM on CPU...")
for epoch in tqdm(range(EPOCHS), desc="Training"):
    model.train()
    for bx, by in loader:
        optimizer.zero_grad()
        pred = model(bx)
        loss = criterion(pred, by)
        loss.backward()
        optimizer.step()
print("✅ LSTM trained.")

# ==============================================
# 3. GENERATE NEW SEQUENCE FROM RANDOM SEED
# ==============================================
n_generations = 1
gen_length = 100   # number of steps to generate

# Start with random seed (SEQ_LEN random coefficient vectors)
seed = np.random.randn(SEQ_LEN, n_modes).astype(np.float32) * 0.1
generated_coeffs = []
model.eval()
with torch.no_grad():
    for _ in range(gen_length):
        inp = torch.tensor(seed[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0).to(device)
        pred = model(inp).squeeze(0).cpu().numpy()
        generated_coeffs.append(pred)
        seed = np.vstack([seed, pred])
generated_coeffs = np.array(generated_coeffs)   # (gen_length, n_modes)

# ==============================================
# 4. RECONSTRUCT SURFACE FIELD FOR THE LAST TIME STEP
# ==============================================
field_gen = evecs @ generated_coeffs[-1]   # (10001,)

# Plot the generated field on the surface
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
step = max(1, len(vertices_ds) // 3000)
ax.scatter(vertices_ds[::step,0], vertices_ds[::step,1], vertices_ds[::step,2],
           c=field_gen[::step], cmap='plasma', s=2)
ax.set_title("Generated Cortical Potential (from random seed)")
plt.savefig(PROJECT_ROOT / "figures" / "generated_sample.png", dpi=300)
plt.show()
print("💾 Saved generated_sample.png")

# ==============================================
# 5. OPTIONAL: Save the generated coefficients
# ==============================================
np.save(PROJECT_ROOT / "generated_coeffs.npy", generated_coeffs)
print("💾 Saved generated_coeffs.npy")
print("🎉 ALL DONE!")