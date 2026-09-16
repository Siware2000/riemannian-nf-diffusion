"""
run_temporal_baselines.py
Compares LSTM against simpler temporal models on held-out electrode prediction.

Baselines:
  - Persistence (naive: predict previous value)
  - Linear Autoregressive (AR) on lagged values
  - Simple MLP on lagged values
  - Your LSTM (loaded from time_series_results.npz)
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import time

# ============================================================
# Load your existing LSTM results
# ============================================================
print("Loading time-series data...")
ts = np.load("time_series_results.npz")
truth = ts["held_truth"]       # (2000,) ground truth
lstm_preds = ts["held_preds"]  # (2000,) your LSTM predictions
sfreq = float(ts["sfreq"])
mse_lstm = float(ts["mse_held"])

print(f"Time-series length: {len(truth)}")
print(f"Sampling frequency: {sfreq} Hz")
print(f"Your LSTM MSE: {mse_lstm:.6e}")

# Normalize truth to match the input format
signal = truth.astype(np.float64)

# ============================================================
# Create lagged dataset for AR / MLP
# ============================================================
def make_lagged(signal, n_lags=20):
    X, y = [], []
    for t in range(n_lags, len(signal)):
        X.append(signal[t - n_lags:t])
        y.append(signal[t])
    return np.array(X), np.array(y)

N_LAGS = 20
TRAIN_FRAC = 0.8

X_all, y_all = make_lagged(signal, N_LAGS)
split = int(len(X_all) * TRAIN_FRAC)
X_train, X_test = X_all[:split], X_all[split:]
y_train, y_test = y_all[:split], y_all[split:]

print(f"\nTrain samples: {len(X_train)}")
print(f"Test samples:  {len(X_test)}")

# Align LSTM predictions to test window
# LSTM preds cover the same signal range; slice out the test portion
lstm_test_preds = lstm_preds[N_LAGS:][split:]
lstm_test_truth = signal[N_LAGS:][split:]

# ============================================================
# Baseline 1: Persistence (predict last value)
# ============================================================
print("\n" + "=" * 60)
print("BASELINE 1: Persistence (y_hat[t] = y[t-1])")
print("=" * 60)
persistence_pred = X_test[:, -1]  # last lagged value
mse_pers = mean_squared_error(y_test, persistence_pred)
print(f"Test MSE: {mse_pers:.6e}")

# ============================================================
# Baseline 2: Linear AR (Ridge regression)
# ============================================================
print("\n" + "=" * 60)
print("BASELINE 2: Linear AR (Ridge)")
print("=" * 60)
t0 = time.time()
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ar_pred = ridge.predict(X_test)
mse_ar = mean_squared_error(y_test, ar_pred)
t_ar = time.time() - t0
print(f"Test MSE: {mse_ar:.6e}   ({t_ar:.2f}s)")

# ============================================================
# Baseline 3: MLP on lagged features
# ============================================================
print("\n" + "=" * 60)
print("BASELINE 3: MLP (2 hidden layers)")
print("=" * 60)

DEVICE = torch.device("cpu")  # MLP is tiny, CPU is fine

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

Xtr_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
ytr_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
Xte_t = torch.tensor(X_test,  dtype=torch.float32, device=DEVICE)

mlp = MLP(X_train.shape[1], hidden=64).to(DEVICE)
opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

t0 = time.time()
EPOCHS = 500
for ep in range(EPOCHS):
    mlp.train()
    opt.zero_grad()
    loss = loss_fn(mlp(Xtr_t), ytr_t)
    loss.backward()
    opt.step()
t_mlp = time.time() - t0

mlp.eval()
with torch.no_grad():
    mlp_pred = mlp(Xte_t).cpu().numpy()
mse_mlp = mean_squared_error(y_test, mlp_pred)
print(f"Test MSE: {mse_mlp:.6e}   ({t_mlp:.2f}s)")

# ============================================================
# Your LSTM (recompute MSE on the same test window)
# ============================================================
print("\n" + "=" * 60)
print("YOUR LSTM (from time_series_results.npz)")
print("=" * 60)
mse_lstm_aligned = mean_squared_error(lstm_test_truth, lstm_test_preds)
print(f"Test MSE (aligned): {mse_lstm_aligned:.6e}")

# ============================================================
# Summary table
# ============================================================
print("\n" + "=" * 60)
print("TEMPORAL MODEL COMPARISON SUMMARY")
print("=" * 60)
print(f"{'Model':<25} {'Test MSE':>15}")
print("-" * 45)
print(f"{'Persistence':<25} {mse_pers:>15.6e}")
print(f"{'Linear AR (Ridge)':<25} {mse_ar:>15.6e}")
print(f"{'MLP (2 layers)':<25} {mse_mlp:>15.6e}")
print(f"{'LSTM (ours)':<25} {mse_lstm_aligned:>15.6e}")

# ============================================================
# Save results
# ============================================================
np.savez("temporal_baselines_results.npz",
         mse_persistence=mse_pers,
         mse_ar=mse_ar,
         mse_mlp=mse_mlp,
         mse_lstm=mse_lstm_aligned,
         n_lags=N_LAGS)

print(f"\n✅ Saved: temporal_baselines_results.npz")