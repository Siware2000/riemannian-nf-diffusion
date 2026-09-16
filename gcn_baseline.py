"""
Graph Neural Network (GCN) baseline with proper train/test split.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from pathlib import Path
import matplotlib.pyplot as plt

# Check torch_geometric
try:
    from torch_geometric.nn import GCNConv
    from torch_geometric.data import Data
    HAS_TG = True
except ImportError:
    HAS_TG = False
    print("⚠️ torch-geometric not installed. Install with: pip install torch-geometric")
    print("   Skipping GCN baseline.")
    exit()

PROJECT_ROOT = Path(r"F:\Deep Riemannian Diffusion Experiment")
results = np.load(PROJECT_ROOT / "full_diffusion_results_final.npz")
vertices_ds = results["vertices_ds"]
field_true = results["field_ds"]
faces_ds = results.get("faces_ds", None)

if faces_ds is None:
    from scipy.spatial import Delaunay
    print("Reconstructing faces via Delaunay...")
    tri = Delaunay(vertices_ds)
    faces_ds = tri.simplices

# Build graph edges
edges = np.concatenate([faces_ds[:, [0,1]], faces_ds[:, [1,2]], faces_ds[:, [0,2]]], axis=0)
edges = np.unique(np.sort(edges, axis=1), axis=0)
edge_index = torch.tensor(edges.T, dtype=torch.long)

# Load electrode coordinates and mean values
mapping = np.load(PROJECT_ROOT / "electrode_mapping.npz")
elec_coords = mapping["electrode_coords"]
n_elec = len(elec_coords)

# Load iEEG data to get electrode mean
import mne
from mne_bids import BIDSPath, read_raw_bids
BIDS_ROOT = PROJECT_ROOT / "Data Set" / "20231231_cog_exp1_bids_ecog_126b5c55-2a56-4fa6-aa98-f20f8e6ed9ec" / "mnt" / "beegfs" / "workspace" / "2023-0385-Cogitatedatarelease" / "CURATE" / "COG_ECOG_EXP1_BIDS"
SUBJECT = "CG105"
SESSION = "1"
bids_path = BIDSPath(subject=SUBJECT, session=SESSION, task="Dur", suffix="ieeg",
                     extension=".edf", root=BIDS_ROOT, datatype="ieeg")
if not bids_path.fpath.exists():
    bids_path = bids_path.copy().update(extension=".vhdr")
raw = read_raw_bids(bids_path, verbose=False)
signal = raw[:, :][0][:n_elec, :]
electrode_mean = signal.mean(axis=1)

# Map electrodes to vertices
tree = cKDTree(vertices_ds)
_, idx_ds = tree.query(elec_coords, k=1)

# Assign electrode mean to nearest vertex
x = np.zeros((len(vertices_ds), 1))
for i, v_idx in enumerate(idx_ds):
    x[v_idx, 0] = electrode_mean[i]
# Normalize
x = (x - x.mean()) / (x.std() + 1e-8)
x = torch.tensor(x, dtype=torch.float)
y = torch.tensor(field_true, dtype=torch.float).view(-1, 1)

# Split vertices randomly (80/20)
np.random.seed(42)
n_vertices = len(vertices_ds)
train_idx = np.random.choice(n_vertices, int(0.8*n_vertices), replace=False)
train_mask = torch.zeros(n_vertices, dtype=torch.bool)
train_mask[train_idx] = True
test_mask = ~train_mask

data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(in_channels=1, hidden_channels=64, out_channels=1).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

for epoch in range(200):
    loss = train()
    if epoch % 50 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.6f}')

model.eval()
with torch.no_grad():
    pred = model(data).cpu().numpy().flatten()
    test_vertices = data.test_mask.cpu().numpy()
    test_mse = np.mean((field_true[test_vertices] - pred[test_vertices])**2)
print(f"GCN Test MSE: {test_mse:.6f}")

# Neural Field on the same test vertices
nf_field = results["generated_ds"]
nf_mse = np.mean((field_true[test_vertices] - nf_field[test_vertices])**2)
print(f"Neural Field MSE (test vertices): {nf_mse:.6f}")
print(f"Improvement over GCN: {(test_mse - nf_mse)/test_mse*100:.2f}%")