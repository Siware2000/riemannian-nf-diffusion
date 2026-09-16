"""
Generate a movie of the evolving cortical field using the trained LSTM.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(r"F:\Deep Riemannian Diffusion Experiment")
results = np.load("full_diffusion_results_final.npz")
vertices_ds = results["vertices_ds"]
evecs = results["evecs"]

# Load coefficients (from the main pipeline – they are saved in the .npz)
coeffs = results["coeffs"]   # shape (N_time, 30) – if present
# If not, you may need to recompute; but it's there in final results.

# If coeffs not in this file, use the one from time_series_results.npz
try:
    coeffs = results["coeffs"]
except:
    ts = np.load("time_series_results.npz")
    coeffs = ts["coeffs"]

# Train a simple LSTM on these coefficients (or load pre-trained)
SEQ_LEN = 20
def build_lstm_data(coeffs):
    X, y = [], []
    for i in range(SEQ_LEN, len(coeffs)):
        X.append(coeffs[i-SEQ_LEN:i])
        y.append(coeffs[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X, y = build_lstm_data(coeffs)

# Define LSTM model
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=30, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

model = LSTMPredictor(input_dim=30, hidden_dim=64, num_layers=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()
dataset = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y))
loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
for epoch in range(30):
    for bx, by in loader:
        optimizer.zero_grad()
        pred = model(bx)
        loss = criterion(pred, by)
        loss.backward()
        optimizer.step()

# Generate a future sequence: start from last SEQ_LEN real coefficients
seed = coeffs[-SEQ_LEN:].copy()
gen_len = 100
generated = []
model.eval()
with torch.no_grad():
    for _ in range(gen_len):
        inp = torch.tensor(seed[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0)
        pred = model(inp).squeeze(0).numpy()
        generated.append(pred)
        seed = np.vstack([seed, pred])
generated = np.array(generated)  # (gen_len, 30)

# Reconstruct surface fields
fields = []
for i in range(gen_len):
    field = evecs @ generated[i]
    fields.append(field)
fields = np.array(fields)  # (gen_len, 10001)

# Create movie
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
step = max(1, len(vertices_ds)//3000)
vmin = fields.min()
vmax = fields.max()

def update(frame):
    ax.clear()
    ax.scatter(vertices_ds[::step,0], vertices_ds[::step,1], vertices_ds[::step,2],
               c=fields[frame][::step], cmap='plasma', s=2, vmin=vmin, vmax=vmax)
    ax.set_title(f"t = {frame}")
    ax.axis('off')
    return ax,

ani = FuncAnimation(fig, update, frames=gen_len, interval=100, blit=False)
ani.save(PROJECT_ROOT / "dynamics_movie.mp4", writer='ffmpeg', fps=10, dpi=150)
print("Saved dynamics_movie.mp4")