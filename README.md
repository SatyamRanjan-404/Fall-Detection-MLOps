# Fall Detection Systems: An MLOps approach to Time-Series Classification

## Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Dataset Description](#dataset-description)
4. [Preprocessing & Feature Engineering](#preprocessing--feature-engineering)
5. [Model Architectures](#model-architectures)
6. [Experiment Results](#experiment-results)
7. [Installation & Requirements](#installation--requirements)
8. [How to Run](#how-to-run)
9. [Key Design Decisions (The Engineering Journal)](#key-design-decisions-the-engineering-journal)
10. [Limitations & Future Work](#limitations--future-work)
11. [Author & Acknowledgements](#author--acknowledgements)

---

## Project Overview

This repository contains an end-to-end Machine Learning pipeline (MLOps) built to classify sequential human motion data—specifically detecting dangerous "Fall" events against a massive background of "Normal" human activity (walking, sitting, standing). 

The goal was to design multiple robust models (from standard 1D Convolutional Neural Networks to modern Selective State Space Models like Mamba) while strictly adhering to rigorous data science principles: addressing heavy class imbalances (a 5:1 ratio), avoiding simple up-sampling, mitigating data leakage, and designing a temporally-aware validation framework.

## Repository Structure

```
.
├── Balanced_DataSets/        # Pre-processed data handling techniques
├── DataSet/                  # Raw sensor streams from user subjects
├── images/                   # Contains loss curves, confusion matrices, and diagrams
├── logs/                     # Detailed reports for every evaluated experiment
├── models/                   # Saved best model checkpoints (.pth)
├── scripts/                  # Standalone execution Python scripts
├── utilities/                # Helper modules (dataset_router, window_extractor, etc.)
├── 1DCNN.ipynb               # Standard CNN experiments
├── data_analysis.ipynb       # Exploratory Data Analysis (EDA) and visualization
├── model.ipynb               # End-to-end orchestrator for modeling
├── using_transformer.ipynb   # Execution paths for Attention/Transformer/Conformer runs
├── work_on_balanced_data.ipynb # Balancing algorithms
├── work_on_imbalanced.ipynb  # Primary exploration notebook with native skewed data
├── requirements.txt          # Python dependencies
└── README.md                 # Project Documentation (You are here)
```

## Dataset Description

The core dataset is comprised of time-series CSV recordings taken from subjects wearing sensor devices.

- **Sampling Rate:** 100 Hz
- **Classes:** Binary Classification ( `0`: Normal, `1`: Fall )
- **Rows/Frames:** Over 1,000,000 Normal frames, and ~200,000 Fall frames.
- **Imbalance Ratio:** ~5:1. 

Each file contains the following critical columns:
- `TimeStamp(s)`, `FrameCounter`
- `AccX`, `AccY`, `AccZ` (3-axis Accelerometer)
- `GyrX`, `GyrY`, `GyrZ` (3-axis Gyroscope)
- `EulerX`, `EulerY`, `EulerZ` (3-axis Orientation)

**Validation Split Method:** 
The dataset is split using **Stratified Group K-Fold**. A "Group" is bounded to a subject/file. This strictly prevents identical timeline fractions (leakage) between Train and Validation sets. The Stratification ensures both sets still receive a balanced allocation of rare fall files. 

## Preprocessing & Feature Engineering

### 1. Signal Vector Magnitude (SVM) Feature
A smartwatch sits on a rotating wrist. A network fed raw axes must waste capacity learning to compensate for 3D rotation. We bypassed this by providing a calculated 10th channel: **Signal Vector Magnitude (SVM)**.

$$ \text{SVM} = \sqrt{A_x^2 + A_y^2 + A_z^2} $$

This delivers a single rotationally invariant channel, allowing the network (especially Attention mechanisms) to instantly anchor to impact magnitude rather than solving spatial orientations.

### 2. Variable-Stride Sliding Window (Temporal Oversampling)
To extract 50-sample sequence windows from the continuous files, while organically combating class imbalance, we implemented **Dynamic Stride**:
- **Normal Class:** Extracted using a strict 25-frame stride (50% overlap).
- **Fall Class ("Burst Mode"):** As soon as a fall is detected, extraction shifts to a tight **5-frame stride**. 

This shifts the model's perspective naturally, generating 5x more Fall windows without any artificial duplication or SMOTE up-sampling. Final labelling states a window is a 'Fall' if >40% of its frames denote a fall.

## Model Architectures

In our comparative analysis, we evaluated 4 broad modeling avenues, scaling from classic structures to cutting-edge Sequence models:
1. **1D CNN (Baseline & Tuned):** Utilizing Optuna-optimized architectures (`f1=32, f2=256, k=7, dropout=0.24`) to slide convolutional kernels directly onto temporal patches. 
2. **1D CNN + Focal Loss:** Modifying the loss function directly (`alpha=0.85, gamma=2.0`) to penalize confident misclassifications and handle native dataset skew organically.
3. **Optuna-Tuned Conformer:** Harnessing a hybrid Convolutional-Transformer model (`d_model=128, heads=8, L=2, drop=0.385`). Self-attention scales context while local convolutions isolate sudden impact spikes.
4. **Pure PyTorch Mamba (SSM):** Implementing a native Selective State Space Model (`d_model=64, drop=0.33`), exploring sub-quadratic recurrent mechanisms capable of endless context windows.

## Experiment Results

All blind tests are executed using a strictly rigid 50% overlap extraction (Stride = 25) to simulate a live 2Hz smartwatch inference buffer.

| Experiment Title             | Model Type          | Precision (Fall) | Recall (Fall) | F1-Score (Fall) | Macro F1 |
|------------------------------|---------------------|------------------|---------------|-----------------|----------|
| `Exp_002_Baseline_1DCNN`     | CNN (Basic)         | 0.90             | 0.92          | 0.91            | 0.95     |
| `Exp_002_Ultimate_1DCNN`     | CNN (Optuna Tuned)  | 0.91             | 0.90          | **0.91**        | 0.95     |
| `Exp_009_Ultimate_1DCNN_FL`  | CNN (Focal Loss)    | 0.85             | **0.94**      | 0.89            | 0.94     |
| `Exp_005_Tuned_Conformer`    | Transformer/CNN     | 0.91             | 0.91          | 0.91            | 0.95     |
| `Exp_006_Pure_Mamba`         | Mamba (SSM Native)  | 0.89             | 0.91          | 0.90            | 0.94     |

*Observation:* While standard tuning maximizes pure F1, Focal Loss modifications distinctly increased safe-guarded Recall (0.94). In biomedical and physical hazard contexts, missed falls (False Negatives) are mathematically far more expensive than False Alarms (False Positives).

## Installation & Requirements

Ensure your system incorporates CUDA-capable hardware for Deep Learning operations.

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd DM_PROJECT
   ```

2. **Create a Python Virtual Environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Unix/macOS
   # .venv\Scripts\activate   # On Windows
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: This repository explicitly relies on `torch>=2.1`, `tensorflow==2.21.0`, `optuna`, and a suite of `nvidia-cuda` bindings (cu12/cu13) for SSM compilation.*

## How to Run

Interaction mostly occurs via the provided Jupyter Notebooks:
- **`data_analysis.ipynb`**: Run this first to generate the necessary statistical profiling and visual understanding of the physical IMU structures.
- **`model.ipynb` / `1DCNN.ipynb`**: Use these to execute the baseline architectures. 
- **`using_transformer.ipynb`**: Handles the heavyweight multi-head attention and Conformer executions.

Alternatively, custom scripts inside `scripts/` provide discrete CLI handles to launch experiments without overhead.

## Key Design Decisions (The Engineering Journal)

- **Class Handling Constraints**: Project restrictions prevented explicit up-sampling or removal of minorities. The "Temporal Burst" approach utilizing window stride manipulation succeeded fully within the prompt's boundaries.
- **Stratified Group Split**: Implementing robust grouped splitting was pivotal. Initial naive splits generated an artificial >0.98 F1 illusion because partial movement sequences bled into validation sets.
- **Why SVM?**: We observed the models getting mathematically "confused" by users who fell with their arm twisted outwards relative to chest-first. SVM squashed orientation matrices into pure physical momentum arrays.

## Limitations & Future Work

- **Inference Speed**: The Conformer and Mamba architectures, while conceptually brilliant, carry larger overheads for deployment on low-wattage microcontrollers (e.g., standard RTOS smartwatches). Future work involves quantization and structured pruning of the Conformer layers.
- **False Alarms**: Vigorous physical activities (like rapid jumps or explosive sports) currently emulate severe IMU spikes. Sensor fusion (combining heart rate deltas with IMU) could provide the contextual key to filtering these out.

## Author & Acknowledgements

Created as part of an Advanced Machine Learning curriculum. Special thanks to the supervising Faculty for designing data constraints that forced engineering realities over synthetic scaling shortcuts.
Work of : Satyam Ranjan , Soham Ganguly , Guranash Singh , Vividh Yadav , Shivansh Goswami , Jigmit Wangldan
