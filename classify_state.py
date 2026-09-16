"""
Use eigenmode coefficients (or generated fields) to classify brain state.
We assume BIDS event files are available with condition labels.
We'll load events for the task and use the coefficient time-series as features.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import mne
from mne_bids import BIDSPath, read_raw_bids
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(r"F:\Deep Riemannian Diffusion Experiment")
BIDS_ROOT = PROJECT_ROOT / "Data Set" / "20231231_cog_exp1_bids_ecog_126b5c55-2a56-4fa6-aa98-f20f8e6ed9ec" / "mnt" / "beegfs" / "workspace" / "2023-0385-Cogitatedatarelease" / "CURATE" / "COG_ECOG_EXP1_BIDS"
SUBJECT = "CG105"
SESSION = "1"

# Load iEEG data (to get coefficients) and events
bids_path = BIDSPath(subject=SUBJECT, session=SESSION, task="Dur", suffix="ieeg",
                     extension=".edf", root=BIDS_ROOT, datatype="ieeg")
if not bids_path.fpath.exists():
    bids_path = bids_path.copy().update(extension=".vhdr")
raw = read_raw_bids(bids_path, verbose=False)

# Load events
events, event_id = mne.events_from_annotations(raw)
# The event_id dictionary maps condition names to numbers.
print("Event IDs:", event_id)

# We need to align the eigenmode coefficients with events.
# Assume we have coefficients from previous runs (coeffs) at the same sampling rate as raw.
# We'll downsample events to match the coefficient time points.
# For simplicity, we'll use the mean of coefficients over certain windows.
# A simpler approach: for each trial, extract the mean coefficient vector over the trial duration.

# Since we don't have the exact events structure, we'll provide a general framework.
# We'll assume you have a function to get trial labels and corresponding coefficient segments.

# Placeholder: Generate synthetic labels and coefficients for demonstration.
np.random.seed(42)
n_samples = 200
coeffs_demo = np.random.randn(n_samples, 30)   # 200 time points, 30 modes
labels = np.random.randint(0, 2, n_samples)   # binary classification

# Train a classifier
clf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(clf, coeffs_demo, labels, cv=5)
print(f"Cross-validation accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

# For a real implementation, you would:
# 1. Load events and extract trial onset times.
# 2. For each trial, average the coefficients over the trial duration.
# 3. Use the averaged coefficients as features.
# 4. Train a classifier to predict the condition (e.g., conscious vs unconscious).
# This script shows the framework; adapt it to your actual events.

print("Classification script ready – replace placeholder data with real events.")