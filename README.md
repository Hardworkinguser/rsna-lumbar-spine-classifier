 Attention-Boosted ConvNeXt for L4/L5 Subarticular Stenosis Severity Classification

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Journal](https://img.shields.io/badge/Journal-Journal%20of%20Medical%20Imaging-blue)](https://www.spiedigitallibrary.org/journals/journal-of-medical-imaging)

<p align="center">
  <img src="figures/framework_overview.png" alt="Framework Overview" width="800"/>
  <br>
  <em>Figure 1: Overall workflow of the proposed attention-boosted ConvNeXt framework with label propagation for automated severity classification of subarticular stenosis (normal/mild, moderate, severe) on axial T2-weighted MRI at the L4/L5 level.</em>
</p>

 Overview

This repository contains the official PyTorch implementation accompanying the paper:

> An Attention-Boosted ConvNeXt Framework with Label Propagation for Automated Severity Classification of L4/L5 Subarticular Stenosis on Axial T2-Weighted MRI  
> (Submitted to Journal of Medical Imaging, 2025/2026)

The proposed pipeline:
- Uses ConvNeXt-Small backbone + custom spatial attention module
- Applies label propagation along the z-axis using DICOM spatial metadata
- Employs Focal Loss (class-balanced) to mitigate severe class imbalance
- Performs coordinate-based ROI cropping (64×64 patches around expert annotations)
- Trains with 5-fold stratified cross-validation, mixed precision (AMP), gradient accumulation, and cosine annealing LR

Key published results (RSNA 2024 dataset, Axial T2, L4/L5 level, held-out test set):
- Overall Accuracy: 89.54%
- Weighted F1-score: 0.8956
- Per-class AUC: 0.95 – 0.98
- Up to 18% F1-score improvement over recent published baselines

 Features

- Full end-to-end training + evaluation pipeline
- DICOM preprocessing (metadata extraction, normalization, 64×64 ROI cropping)
- Label propagation based on z-distance threshold (<15 mm)
- Custom spatial attention block integrated into ConvNeXt feature maps
- Class-balanced Focal Loss with inverse frequency weighting
- Stratified 5-fold CV + separate held-out test evaluation
- Comprehensive metrics: accuracy, macro/weighted F1, precision, recall, ROC-AUC, confusion matrix, per-class analysis
- Reproducible results via fixed random seed and detailed logging

 Requirements

```text
Python >= 3.8
torch >= 2.0
torchvision
pydicom
pandas
numpy
scikit-learn
matplotlib
seaborn
tqdm
Pillow
imbalanced-learn 


If you use this code, please cite:

Mina Hashemi, "RSNA Lumbar Spine Classifier", 2026.
