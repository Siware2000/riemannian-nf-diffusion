"""
lstm_fixed.py
Correctly trains an LSTM to predict held-out iEEG electrode signal.

Key fixes vs. the broken version:
  1. Proper z-score normalization (zero mean, unit variance)
  2. Save scaler for un-normalization at prediction time
  3. Train longer with early stopping
  4. Larger hidden size, better regularization
  5. Reproducible seed
  6. Diagnostic output at every step
"""
import numpy as np
import torch
import torch.nn as nn
import time

torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# 1. LOAD SIGNAL
# ============================================================
print("Loading signal...")
d = np.load("time_series_results.npz")
signal = d["held_truth"].astype(np.float32)   # raw held-out electrode
print(f"Signal length: {len(signal)}")
print(f"Signal range:  [{signal.min():.3f}, {signal.max():.3f}]")
print(f"Signal std:    {signal.std():.4f}")

# ============================================================
# 2. Z-SCORE NORMALIZATION (critical fix!)
# ============================================================
mu = signal.mean()
sigma = signal.std()
signal_norm = (signal - mu) / (sigma + 1e-8)
print(f"\nAfter normalization:")
print(f"  mean = {signal_norm.mean():.6f} (should be ~0)")
print(f"  std  = {signal_norm.std():.6f} (should be ~1)")

# ============================================================
# 3. CREATE LAGGED DATASET
# ============================================================
N_LAGS = 30

def make_windows(x, n_lags):
    X, y = [], []
    for t in range(n_lags, len(x)):
        X.append(x[t - n_lags:t])
        y.append(x[t])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X, y = make_windows(signal_norm, N_LAGS)
print(f"\nDataset: X.shape={X.shape}, y.shape={y.shape}")

# Train/val/test split (chronological)
n = len(X)
n_train = int(0.7 * n)
n_val   = int(0.15 * n)

X_train, y_train = X[:n_train], y[:n_train]
X_val,   y_val   = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
X_test,  y_test  = X[n_train+n_val:], y[n_train+n_val:]

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

X_train_t = torch.tensor(X_train).unsqueeze(-1)
y_train_t = torch.tensor(y_train)
X_val_t   = torch.tensor(X_val).unsqueeze(-1)
y_val_t   = torch.tensor(y_val)
X_test_t  = torch.tensor(X_test).unsqueeze(-1)
y_test_t  = torch.tensor(y_test)

# ============================================================
# 4. LSTM MODEL
# ============================================================
class LSTMPredictor(nn.Module):
    def __init__(self, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]      # take last time step
        return self.head(last).squeeze(-1)

DEVICE = torch.device("cpu")  # tiny model, CPU is fine
model = LSTMPredictor(hidden=64, layers=2).to(DEVICE)

# ============================================================
# 5. TRAINING
# ============================================================
EPOCHS = 200
BATCH  = 64
LR     = 1e-3
PATIENCE = 20

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10
)
loss_fn = nn.MSELoss()

train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH, shuffle=True)

best_val = float("inf")
best_state = None
no_improve = 0
history = {"train": [], "val": []}

print("\nTraining LSTM...")
t0 = time.time()
for ep in range(EPOCHS):
    model.train()
    ep_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        ep_loss += loss.item() * len(xb)
    ep_loss /= len(X_train_t)

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t)
        val_loss = loss_fn(val_pred, y_val_t).item()
    scheduler.step(val_loss)

    history["train"].append(ep_loss)
    history["val"].append(val_loss)

    if val_loss < best_val - 1e-6:
        best_val = val_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1

    if ep % 10 == 0 or ep == EPOCHS - 1:
        print(f"  Epoch {ep:3d}: train={ep_loss:.6f}, val={val_loss:.6f}, "
              f"lr={optimizer.param_groups[0]['lr']:.1e}")

    if no_improve >= PATIENCE:
        print(f"  Early stop at epoch {ep} (no improvement for {PATIENCE} epochs)")
        break

print(f"Training time: {time.time() - t0:.1f}s")

# ============================================================
# 6. TEST EVALUATION (on normalized scale)
# ============================================================
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    test_pred_norm = model(X_test_t).numpy()
    val_pred_norm  = model(X_val_t).numpy()

mse_test_norm = float(np.mean((test_pred_norm - y_test)**2))
mse_val_norm  = float(np.mean((val_pred_norm  - y_val )**2))

# ============================================================
# 7. UN-NORMALIZE AND EVALUATE IN ORIGINAL UNITS
# ============================================================
test_pred_raw = test_pred_norm * sigma + mu
y_test_raw    = y_test         * sigma + mu
mse_test_raw  = float(np.mean((test_pred_raw - y_test_raw)**2))

# Baselines on test set
mse_persist = float(np.mean((X_test[:, -1] * sigma + mu - y_test_raw)**2))
mse_mean    = float(np.mean((y_test_raw.mean() - y_test_raw)**2))

# ============================================================
# 8. REPORT
# ============================================================
print("\n" + "=" * 60)
print("LSTM FIXED — RESULTS")
print("=" * 60)
print(f"\nNormalized scale (what the LSTM was trained on):")
print(f"  Val MSE:  {mse_val_norm:.6f}")
print(f"  Test MSE: {mse_test_norm:.6f}")

print(f"\nOriginal units (µV or whatever your signal is):")
print(f"  Test MSE:              {mse_test_raw:.6f}")
print(f"  Persistence baseline:  {mse_persist:.6f}")
print(f"  Mean baseline:         {mse_mean:.6f}")

print(f"\nDiagnostic checks:")
print(f"  Prediction std: {test_pred_norm.std():.4f} (truth std = 1.0)")
print(f"  Correlation:    {np.corrcoef(test_pred_norm, y_test)[0,1]:.4f}")

if mse_test_norm < 0.5 and np.corrcoef(test_pred_norm, y_test)[0,1] > 0.5:
    print(f"  ✅ LSTM LEARNED THE SIGNAL")
else:
    print(f"  ⚠️  LSTM still not learning well")

# ============================================================
# 9. SAVE
# ============================================================
np.savez("lstm_fixed_results.npz",
         test_pred_norm=test_pred_norm,
         test_truth_norm=y_test,
         test_pred_raw=test_pred_raw,
         test_truth_raw=y_test_raw,
         mse_test_norm=mse_test_norm,
         mse_test_raw=mse_test_raw,
         mse_persistence=mse_persist,
         mse_mean=mse_mean,
         mu=mu, sigma=sigma)

print(f"\n✅ Saved: lstm_fixed_results.npz")