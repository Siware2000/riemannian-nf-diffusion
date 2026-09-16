# Riemannian Diffusion Neural Fields for Continuous Bio-Potential Generation on Cortical Surfaces

This repository contains the code and pipelines for the paper:

**"Riemannian Score-Based Neural Field Diffusion for Continuous Bio-Potential Generation on Non-Euclidean Cortical Geometries"**

## Overview

We introduce a Riemannian diffusion framework that generates continuous 
bio-potentials on the cortical surface using iEEG data from the 
open-access **Cogitate** dataset. The pipeline combines:

- A **Neural Field** (SIREN) that maps any 3D coordinate on the cortical 
  surface to a continuous potential, augmented with intrinsic geometric 
  features (surface normals, curvatures, and Laplacian eigenfunctions).
- A **score-based diffusion model** on the Laplacian eigenbasis, enabling 
  generative sampling of cortical fields from noise.
- **Spherical harmonics** as geometric features to promote 
  rotation-sensitive representations.
- **GPU acceleration** on Intel iGPU via DirectML, making the pipeline 
  accessible on consumer hardware.

## Results

| Model | Task | MSE |
| :--- | :--- | :--- |
| IDW baseline | Held-out electrode | 0.1858 |
| SIREN (xyz only) | Mean field | 1.78e-5 |
| **SH-augmented (ours)** | **Mean field** | **6.55e-6** |
| **SH-augmented (ours)** | **Held-out electrode** | **0.0489 (73.7% better than IDW)** |

**Ablation:** Adding intrinsic geometric features (normals, curvatures, 
and 30 Laplacian eigenfunctions) improves reconstruction by **63.3%** 
over coordinate-only input.

**Negative result:** Temporal prediction of held-out electrode time-series 
at 1024 Hz did not outperform baselines (LSTM, AR, MLP all had 
MSE ≈ 1.2, similar to the "predict the mean" baseline). We report this 
honestly as a limitation. The pipeline is still useful for spatial 
reconstruction and generative sampling.

## Dataset

We used the **Cogitate iEEG dataset** (open-access), which includes 
intracranial recordings from 38 patients across three international sites.

- **Dataset paper:** Seedat et al., *Scientific Data*, 2025. 
  DOI: [10.1038/s41597-025-04833-z](https://doi.org/10.1038/s41597-025-04833-z)
- **Dataset DOI:** [10.17617/1.6qpg-gz82](https://doi.org/10.17617/1.6qpg-gz82)
- **Data release:** [https://arc-cogitate.com](https://arc-cogitate.com)

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd DeepRiemannianDiffusionExperiment