# Geometry-Aware Neural Fields and Conditional Diffusion for Robust Cross-Subject iEEG Imputation

This repository contains the official code, pipelines, and reproducible experiments for the paper:

**"Geometry-Aware Neural Fields and Conditional Diffusion for Robust Cross-Subject iEEG Imputation"**

## Overview

We introduce a rigorously validated framework for continuous cortical potential reconstruction and probabilistic imputation using intracranial electroencephalography (iEEG) data from the open-access **Cogitate** dataset. 

**Note on methodology update:** The previous version of this repository contained a target leakage bug in the evaluation protocol (where the reference field was constructed using all electrodes). The current pipeline has been strictly updated to enforce **leakage-free 5-fold cross-validation** and fair ablations. 

The pipeline combines:
- **Geometry-Aware Neural Fields (MLP backbone)**: Maps 3D cortical coordinates to continuous potentials, augmented with intrinsic geometric features (surface normals, curvature, and 30 Laplacian eigenfunctions).
- **Conditional Score-Based Diffusion**: Operates within the Laplacian eigenbasis to perform *probabilistic imputation* of missing electrodes, providing calibrated uncertainty (CRPS).
- **Edge Computing**: GPU acceleration via PyTorch and DirectML for consumer hardware (Intel iGPU).

## Key Results

### 1. Within-Subject Reconstruction (Strict 5-Fold CV)
We demonstrate that purely spatial models struggle with high-frequency iEEG noise. While IDW remains a strong deterministic baseline, our Conditional Diffusion model provides crucial probabilistic uncertainty. 

| Method | Test Metric |
| :--- | :--- |
| Inverse Distance Weighting (IDW) | MSE: 0.850 ± 0.972 |
| Thin-Plate Spline | MSE: 0.989 ± 0.659 |
| Gaussian Process (RBF) | MSE: 1.028 ± 1.205 |
| Neural Field (XYZ only) | MSE: 1.058 ± 0.999 |
| **Neural Field (XYZ + Geometry)** | **MSE: 1.051 ± 0.975** |
| **Conditional Diffusion (Ours)** | **CRPS: 0.451 ± 0.182** |

### 2. Cross-Subject Generalization (LOSO)
By acting as a spatial regularizer, continuous Neural Fields significantly outperform rigid interpolation across unseen patients.

| Method | Train CF102 → Test CF120 | Train CF120 → Test CF102 |
| :--- | :--- | :--- |
| Inverse Distance Weighting (IDW) | 2.039 | 2.039 |
| Thin-Plate Spline | 2.039 | 2.039 |
| Gaussian Process (RBF) | 1.452 | 1.834 |
| **Neural Field (Ours)** | **1.130** | **1.346** |

### 3. Generative Uncertainty
The conditional diffusion model successfully performs probabilistic imputation, generating a distribution of plausible potentials where deterministic models output a single point estimate. See `figures/fig3_diffusion_uncertainty.png` in the paper for calibration curves.

## Dataset

We used the **Cogitate iEEG dataset** (open-access), which includes intracranial recordings from 38 patients across three international sites.

- **Dataset paper:** Seedat et al., *Scientific Data*, 2025. DOI: [10.1038/s41597-025-04833-z](https://doi.org/10.1038/s41597-025-04833-z)
- **Dataset DOI:** [10.17617/1.6qpg-gz82](https://doi.org/10.17617/1.6qpg-gz82)
- **Data release:** [https://arc-cogitate.com](https://arc-cogitate.com)

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/Siware2000/riemannian-nf-diffusion.git
cd riemannian-nf-diffusion
