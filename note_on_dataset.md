## Dataset & Preprocessing Pipeline

---

### 1. Dataset Structure
```
Dataset/
├── Sample_Training/
│   └── SAXX/               ← one folder per subject (XX = integer)
│       └── SAXXTYYRZZ.csv  ← task TYY, repetition RZZ
└── Sample_Test/
    └── (same structure as Sample_Training)
```

#### CSV Attributes

| Column | Description |
|--------|-------------|
| `TimeStamp(s)` | Recording timestamp in seconds |
| `FrameCounter` | Sequential frame index |
| `AccX/Y/Z` | Accelerometer readings (3 axes) |
| `GyrX/Y/Z` | Gyroscope readings (3 axes) |
| `EulerX/Y/Z` | Orientation angles (3 axes) |
| `FallCheck` | Binary label — 0 = Normal, 1 = Fall |

#### Class Distribution

| Label | Frames | Share |
|-------|--------|-------|
| 0 — Normal | ~1,000,000 | ~83% |
| 1 — Fall | ~200,000 | ~17% |
| **Imbalance ratio** | **5 : 1** | |

---

### 2. Train / Validation Split — Stratified Group Split

The dataset is split using a **Stratified Group K-Fold** strategy.

- **"Group"** — all rows belonging to a single file or subject are kept
  strictly together and cannot be separated across the split boundary.
- **"Stratified"** — the algorithm forces both the train and validation
  buckets to receive a proportional share of rare fall-containing files.

**Why it matters:**
- Prevents **data leakage** — a known danger in time-series data where
  adjacent windows from the same recording can bleed between splits.
- **Isolates subjects** — the model is validated on people it has never
  seen, proving generalisation rather than memorisation.

---

### 3. Window Extraction — Variable-Stride Sliding Window

A 50-frame window is slid across each recording using a
**class-aware dynamic stride**:

| Window content | Stride applied |
|----------------|----------------|
| Majority class (no fall) | Large stride — fewer windows extracted |
| Minority class (fall) | Small stride — many overlapping windows |

This is **temporal oversampling**. Instead of duplicating rows
(overfitting risk) or using synthetic data generation, it extracts
naturally occurring, time-shifted variations of the same physical fall
event. The model sees each fall from multiple temporal perspectives
without any artificial data.

---

### 4. Feature Engineering — Signal Vector Magnitude (SVM)

#### The Problem
A smartwatch sits on the wrist — a highly articulated joint. During a
fall, the impact spike may appear entirely on the X-axis, entirely on Y,
or split across all three depending on wrist orientation. A network fed
raw axes must waste capacity learning to compensate for 3D rotation.

#### The Solution
Compute a single **rotationally invariant** channel:

$$\text{SVM} = \sqrt{A_x^2 + A_y^2 + A_z^2}$$

Regardless of wrist orientation, the SVM channel always produces a
large, unambiguous spike at the moment of impact.

#### Tensor Transformation Pipeline

| Step | Operation | Shape |
|------|-----------|-------|
| A — Raw input | 9 physical sensor channels | `[B, 50, 9]` |
| B — SVM extraction | `np.sqrt(np.sum(accel² , axis=-1))` | `[B, 50, 1]` |
| C — Concatenation | Append SVM as 10th channel | `[B, 50, 10]` |
| D — PyTorch permute | Transpose for `nn.Conv1d` | `[B, 10, 50]` |

#### Architectural Impact

- **Saved network capacity** — the model receives the Euclidean norm
  pre-computed rather than learning Pythagoras implicitly across 50 frames.
- **Anchored attention** — the SVM spike acts as a high-magnitude anchor,
  allowing the Multi-Head Attention mechanism to instantly localise the
  moment of impact and contrast it against the preceding walking context.
  