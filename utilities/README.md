#  Framework: The Utilities Pipeline

To ensure absolute mathematical rigor, prevent data leakage, and standardize all model evaluations, the data processing and execution logic was abstracted into a custom, object-oriented MLOps framework located in the `utilities/` directory.

## 1. `dataset_router.py` (Data Organization & Leakage Prevention)

**Purpose:** Handles the physical routing of raw CSV files to guarantee strict isolation between training, validation, and testing environments.

| Function / Method | Description |
| :--- | :--- |
| `__init__(dataset_path, test_size, random_state)` | Initializes the router, targeting the base directory of the raw sensor CSV files and establishing the mathematical split ratios (e.g., 80/20). |
| `create_splits()` | Reads the directory structure, shuffles the physical files, and safely partitions them into `train_files` and `val_files` lists. By splitting at the file level rather than the row level, it mathematically guarantees zero temporal data leakage between the sets. |

## 2. `window_extractor.py` (Time-Series Slicing & Feature Engineering)

**Purpose:** Transforms continuous streams of physical sensor data into discrete, 3-dimensional mathematical tensors (`[Samples, Frames, Channels]`) suitable for neural networks.

| Function / Method | Description |
| :--- | :--- |
| `__init__(window_size, fall_threshold)` | Establishes the temporal physics of the pipeline. Sets the context window (e.g., $T=50$ frames, representing 1.0 seconds of real-time data). |
| `extract_dynamic(file_list, normal_stride, fall_stride)` | The Class Imbalance Fix. Sweeps across training files. Uses a standard step (e.g., 25 frames) for normal walking, but switches to "burst mode" (e.g., 5 frames) when a fall is detected, synthetically multiplying the minority class without creating fake data. |
| `extract_standard(file_list, stride)` | Sweeps across validation files using a fixed, realistic step size (e.g., 50% overlap) to simulate real-world smartwatch buffering. |
| `extract_strict_overlap(file_list, overlap_fraction)` | Safely slices the Blind Test data, strictly enforcing file boundaries to ensure testing windows never cross over two unrelated recording sessions. |
| `_process_window(window_data)` | The Feature Engineer. An internal private method that mathematically isolates the 3 Accelerometer axes, calculates the rotational-invariant Signal Vector Magnitude (SVM) using the Euclidean norm, and concatenates it as a 10th channel to the raw tensor. |

## 3. `experiment_trainer.py` (Automated Training & Optimization)

**Purpose:** Standardizes the PyTorch training loop across all architectures (Conformer, Mamba, 1D CNN) to ensure apples-to-apples performance comparisons.

| Function / Method | Description |
| :--- | :--- |
| `__init__(exp_name, description, model, criterion, optimizer)` | Initializes the experiment environment, linking the chosen neural architecture, the loss function (e.g., Focal Loss or BCE), and the optimizer. Prepares logging structures. |
| `train(X_train, y_train, X_val, y_val, epochs, batch_size)` | The core execution engine. Handles forward propagation, gradient descent, and validation checks. It tracks Training Loss, Validation Loss, F1-Score, and Recall. Crucially, it acts as a Model Checkpoint, automatically saving the `.pth` weights of the mathematically best epoch to the hard drive. |

## 4. `experiment_tester.py` (The Blind Test Auditor)

**Purpose:** A strictly locked-down inference engine that evaluates the saved models on data they have never seen, generating the final academic metrics.

| Function / Method | Description |
| :--- | :--- |
| `__init__(exp_name, model_architecture)` | Instantiates the empty PyTorch skeleton for the architecture being audited. |
| `run_blind_test(X_test, y_test, test_name, batch_size)` | Retrieves the optimal `.pth` weights saved by the ExperimentTrainer, locks the network's gradients (`torch.no_grad()`), and runs a pure forward pass on the test tensor. It outputs the final terminal classification report and the definitive Confusion Matrix (True Positives, False Positives, False Negatives, True Negatives). |
