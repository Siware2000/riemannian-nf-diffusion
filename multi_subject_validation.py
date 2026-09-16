"""
Multi‑subject LSTM validation on Cogitate iEEG.
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

# List of subjects (use all available, or a subset)
# You can auto-detect by scanning sub-* folders
SUBJECTS = ["CG105", "CE107", "CF102", "CE110", "CF104"]   # add more as needed
SESSION = "1"

# Load eigenbasis from first subject (assumes same surface for all)
results = np.load("full_diffusion_results_final.npz")
evecs = results["evecs"]          # (10001, 30)
n_modes = evecs.shape[1]

# LSTM hyperparameters
SEQ_LEN = 20
EPOCHS = 30
BATCH_SIZE = 64
HIDDEN_DIM = 64
NUM_LAYERS = 2
N_TIME = 1000   # number of time points to sample

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
    # We need electrode coordinates, load from mapping file
    mapping = np.load("electrode_mapping.npz")
    elec_coords = mapping["electrode_coords"]
    # Find nearest vertices
    from scipy.spatial import cKDTree
    vertices_ds = np.load("full_diffusion_results_final.npz")["vertices_ds"]
    tree = cKDTree(vertices_ds)
    _, idx_ds = tree.query(elec_coords, k=1)
    # Build field
    field = np.zeros((len(vertices_ds), n_time))
    for t in range(n_time):
        f = np.zeros(len(vertices_ds))
        for e, v in enumerate(idx_ds):
            f[v] += sampled_signal[e, t]
        # Normalise duplicates
        unique, counts = np.unique(idx_ds, return_counts=True)
        for u, c in zip(unique, counts):
            f[u] /= c
        field[:, t] = f
    # Project to eigenbasis
    coeffs = evecs.T @ field   # (30, n_time)
    coeffs = coeffs.T           # (n_time, 30)
    return coeffs, sfreq, n_elec

# Prepare LSTM model
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=30, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

# Evaluate each subject
all_mse = []
for subj in SUBJECTS:
    print(f"\n🔍 Processing subject {subj}...")
    try:
        coeffs, sfreq, n_elec = load_subject_data(BIDS_ROOT, subj, SESSION, evecs, N_TIME)
        # Prepare LSTM data
        X_seq, y_seq = [], []
        for i in range(SEQ_LEN, len(coeffs)):
            X_seq.append(coeffs[i-SEQ_LEN:i])
            y_seq.append(coeffs[i])
        X_seq = np.array(X_seq, dtype=np.float32)
        y_seq = np.array(y_seq, dtype=np.float32)
        split = int(0.8 * len(X_seq))
        X_train, y_train = X_seq[:split], y_seq[:split]
        X_val, y_val = X_seq[split:], y_seq[split:]
        X_train_t = torch.tensor(X_train).to(device)
        y_train_t = torch.tensor(y_train).to(device)
        X_val_t = torch.tensor(X_val).to(device)
        y_val_t = torch.tensor(y_val).to(device)

        model = LSTMPredictor(input_dim=n_modes, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
        # Train quickly
        for epoch in range(EPOCHS):
            model.train()
            for bx, by in train_loader:
                optimizer.zero_grad()
                pred = model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
        # Evaluate on held-out time segment (last 10%)
        model.eval()
        test_start = int(0.9 * len(X_val))
        X_test = X_val_t[test_start:]
        y_test = y_val_t[test_start:]
        with torch.no_grad():
            pred_test = model(X_test)
            mse = criterion(pred_test, y_test).item()
        all_mse.append(mse)
        print(f"   Subject {subj} MSE: {mse:.6f}")
    except Exception as e:
        print(f"   Subject {subj} failed: {e}")
        all_mse.append(np.nan)

print("\n📊 Summary")
print(f"Average MSE across subjects: {np.nanmean(all_mse):.6f}")
print(f"Number of subjects: {len(all_mse) - np.isnan(all_mse).sum()}")