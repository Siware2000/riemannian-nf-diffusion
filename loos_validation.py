"""
Leave-One-Subject-Out (LOSO) validation using LSTM on eigenmode coefficients.
Trains on all subjects except one, tests on the held-out subject.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import mne
from mne_bids import BIDSPath, read_raw_bids
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(r"F:\Deep Riemannian Diffusion Experiment")
BIDS_ROOT = PROJECT_ROOT / "Data Set" / "20231231_cog_exp1_bids_ecog_126b5c55-2a56-4fa6-aa98-f20f8e6ed9ec" / "mnt" / "beegfs" / "workspace" / "2023-0385-Cogitatedatarelease" / "CURATE" / "COG_ECOG_EXP1_BIDS"
ALL_SUBJECTS = ["CG105", "CE107", "CF102", "CE110", "CF104"]
SESSION = "1"

# Load eigenbasis
results = np.load("full_diffusion_results_final.npz")
evecs = results["evecs"]          # (10001, 30)
n_modes = evecs.shape[1]
vertices_ds = results["vertices_ds"]

# Electrode mapping
mapping = np.load("electrode_mapping.npz")
elec_coords = mapping["electrode_coords"]
from scipy.spatial import cKDTree
tree = cKDTree(vertices_ds)
_, idx_ds = tree.query(elec_coords, k=1)
n_elec = len(elec_coords)

# LSTM hyperparameters
SEQ_LEN = 20
EPOCHS = 30
BATCH_SIZE = 64
HIDDEN_DIM = 64
NUM_LAYERS = 2
N_TIME = 1000   # sample time points

def load_subject_coeffs(subject):
    """Load iEEG, interpolate, project to eigenbasis."""
    bids_path = BIDSPath(subject=subject, session=SESSION, task="Dur", suffix="ieeg",
                         extension=".edf", root=BIDS_ROOT, datatype="ieeg")
    if not bids_path.fpath.exists():
        bids_path = bids_path.copy().update(extension=".vhdr")
    raw = read_raw_bids(bids_path, verbose=False)
    data, _ = raw[:, :]
    signal = data[:n_elec, :]
    signal = (signal - signal.mean(axis=1, keepdims=True)) / (signal.std(axis=1, keepdims=True) + 1e-8)
    time_idx = np.linspace(0, signal.shape[1]-1, N_TIME, dtype=int)
    sampled_signal = signal[:, time_idx]
    # Interpolate to surface
    field = np.zeros((len(vertices_ds), N_TIME))
    for t in range(N_TIME):
        f = np.zeros(len(vertices_ds))
        for e, v in enumerate(idx_ds):
            f[v] += sampled_signal[e, t]
        unique, counts = np.unique(idx_ds, return_counts=True)
        for u, c in zip(unique, counts):
            f[u] /= c
        field[:, t] = f
    coeffs = evecs.T @ field
    coeffs = coeffs.T
    return coeffs

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=30, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

def train_lstm(subject_list):
    """Train LSTM on multiple subjects' data."""
    X_all, y_all = [], []
    for subj in subject_list:
        coeffs = load_subject_coeffs(subj)
        for i in range(SEQ_LEN, len(coeffs)):
            X_all.append(coeffs[i-SEQ_LEN:i])
            y_all.append(coeffs[i])
    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.float32)
    model = LSTMPredictor(n_modes, HIDDEN_DIM, NUM_LAYERS)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    dataset = TensorDataset(torch.tensor(X_all), torch.tensor(y_all))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    for epoch in range(EPOCHS):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
    return model

def evaluate_on_subject(model, subject):
    coeffs = load_subject_coeffs(subject)
    X_test, y_test = [], []
    for i in range(SEQ_LEN, len(coeffs)):
        X_test.append(coeffs[i-SEQ_LEN:i])
        y_test.append(coeffs[i])
    X_test = torch.tensor(np.array(X_test, dtype=np.float32))
    y_test = torch.tensor(np.array(y_test, dtype=np.float32))
    model.eval()
    with torch.no_grad():
        pred = model(X_test)
        mse = nn.MSELoss()(pred, y_test).item()
    return mse

# LOSO loop
all_mse = []
for held_out in ALL_SUBJECTS:
    train_subjs = [s for s in ALL_SUBJECTS if s != held_out]
    print(f"\nTraining on {train_subjs}, testing on {held_out}")
    model = train_lstm(train_subjs)
    mse = evaluate_on_subject(model, held_out)
    all_mse.append(mse)
    print(f"  Held-out MSE: {mse:.6f}")

print("\n=== LOSO Results ===")
print(f"Average MSE: {np.mean(all_mse):.6f} ± {np.std(all_mse):.6f}")
for subj, mse in zip(ALL_SUBJECTS, all_mse):
    print(f"  {subj}: {mse:.6f}")