# rfi_toolbox API Reference

This document provides essential import patterns and usage examples for integrating `rfi_toolbox` into your ML-based RFI detection pipeline.

---

## Core Modules

### 1. Measurement Set I/O (`io.ms_loader`)

**Load visibility data from CASA Measurement Sets:**

```python
from rfi_toolbox.io import MSLoader

# Load MS data
loader = MSLoader('observation.ms')
loader.load(num_antennas=10, mode='DATA')

# Access data
visibilities = loader.data  # Complex visibilities (baselines, pols, channels, times)
flags = loader.load_flags()  # Existing flags

# Save flags back to MS
loader.save_flags(predicted_flags, column='FLAG')
```

---

### 2. Preprocessing (`preprocessing.preprocessor`)

**Convert visibility data into ML-ready patches:**

```python
from rfi_toolbox.preprocessing import Preprocessor

# Initialize preprocessor
preprocessor = Preprocessor(
    data=loader.data,
    flags=loader.load_flags()
)

# Create dataset with patchification
dataset = preprocessor.create_dataset(
    patch_size=128,
    stretch='SQRT',  # Options: 'SQRT', 'LOG10', None
    enable_augmentation=False
)

# Access patches
data_patches = dataset['data']      # Shape: (N, C, H, W)
mask_patches = dataset['labels']    # Shape: (N, H, W)
```

---

### 3. Evaluation Metrics (`evaluation.metrics`)

**Compute segmentation metrics for RFI masks:**

```python
from rfi_toolbox.evaluation import (
    evaluate_segmentation,
    compute_iou,
    compute_precision,
    compute_recall,
    compute_f1,
    compute_dice
)

# All metrics at once
metrics = evaluate_segmentation(predicted_mask, ground_truth_mask)
# Returns: {'iou': 0.85, 'precision': 0.90, 'recall': 0.82, 'f1': 0.86, 'dice': 0.86}

# Individual metrics
iou = compute_iou(predicted_mask, ground_truth_mask)
precision = compute_precision(predicted_mask, ground_truth_mask)
recall = compute_recall(predicted_mask, ground_truth_mask)

# Accepts both numpy arrays and torch tensors
```

---

### 4. Statistical Evaluation (`evaluation.statistics`)

**Assess flagging quality on real data (no ground truth needed):**

```python
from rfi_toolbox.evaluation import (
    compute_ffi,
    compute_statistics,
    print_statistics_comparison
)

# Flagging Fidelity Index (FFI) - higher is better
ffi_metrics = compute_ffi(data, flags=predicted_mask)
# Returns: {'ffi': 0.65, 'mad_reduction': 0.45, 'std_reduction': 0.52, 'flagged_fraction': 0.28}

# Before/after statistics
stats = compute_statistics(data, flags=predicted_mask)
# Returns: {'mean': ..., 'median': ..., 'std': ..., 'mad': ..., 'count': ..., 'flagged_fraction': ...}

# Print formatted comparison
print_statistics_comparison(data, predicted_mask)
```

---

### 5. Synthetic Data Generation (`data_generation.synthetic_generator`)

**Generate training data with physics-based RFI simulation:**

```python
from rfi_toolbox.data_generation import SyntheticDataGenerator

# Initialize generator
generator = SyntheticDataGenerator(
    time_bins=1024,
    frequency_bins=1024,
    noise_level=1.0,
    rfi_power_min=100.0,
    rfi_power_max=1000.0
)

# Generate single sample
data, mask = generator.generate()

# Generate batch
batch = generator.generate_batch(num_samples=100)
```

---

### 6. PyTorch Datasets (`datasets`)

**Load pre-generated training data:**

```python
from rfi_toolbox.datasets import RFIMaskDataset
from torch.utils.data import DataLoader

# Load dataset from directory
train_dataset = RFIMaskDataset(
    data_dir='./rfi_dataset/train',
    normalization='global_min_max',  # Options: 'global_min_max', 'standardize', 'robust_scale', None
    transform=None
)

# PyTorch DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Iterate
for inputs, masks in train_loader:
    # inputs: (batch, 8, H, W) - 8 channels (4 pols × 2 real/imag)
    # masks: (batch, 1, H, W)
    predictions = your_model(inputs)
```

---

## Complete Workflow Examples

### Example 1: Load MS → Process → Evaluate Custom Model

```python
from rfi_toolbox.io import MSLoader
from rfi_toolbox.preprocessing import Preprocessor
from rfi_toolbox.evaluation import evaluate_segmentation

# 1. Load data
loader = MSLoader('observation.ms')
loader.load(num_antennas=10, mode='DATA')

# 2. Preprocess
preprocessor = Preprocessor(loader.data, flags=loader.load_flags())
dataset = preprocessor.create_dataset(patch_size=128, stretch='SQRT')

# 3. Run your custom model
def my_rfi_detector(data):
    # Your ML model here (PyTorch, JAX, TensorFlow, etc.)
    # Return binary mask with same shape as input
    return predicted_mask

predictions = my_rfi_detector(dataset['data'])

# 4. Evaluate
metrics = evaluate_segmentation(predictions, dataset['labels'])
print(f"IoU: {metrics['iou']:.3f}, F1: {metrics['f1']:.3f}")

# 5. Save flags back to MS
loader.save_flags(predictions, column='FLAG')
```

---

### Example 2: Generate Synthetic Training Data

```python
from rfi_toolbox.data_generation import SyntheticDataGenerator
import numpy as np

# Initialize generator
generator = SyntheticDataGenerator(
    time_bins=1024,
    frequency_bins=1024,
    noise_level=1.0,
    rfi_power_min=100.0,
    rfi_power_max=1000.0
)

# Generate training set
num_samples = 1000
for i in range(num_samples):
    data, mask = generator.generate()

    # Save to disk (your format)
    np.save(f'train/{i:04d}_data.npy', data)
    np.save(f'train/{i:04d}_mask.npy', mask)
```

---

### Example 3: Evaluate on Real Data (No Ground Truth)

```python
from rfi_toolbox.io import MSLoader
from rfi_toolbox.evaluation import compute_ffi, print_statistics_comparison

# Load MS
loader = MSLoader('observation.ms')
loader.load(num_antennas=5, mode='DATA')

# Your model predictions
predicted_flags = your_model.predict(loader.data)

# Assess quality with FFI
ffi_metrics = compute_ffi(loader.data, flags=predicted_flags)
print(f"FFI: {ffi_metrics['ffi']:.3f}")
print(f"MAD reduction: {ffi_metrics['mad_reduction']:.1%}")
print(f"Flagged fraction: {ffi_metrics['flagged_fraction']:.1%}")

# Detailed comparison
print_statistics_comparison(loader.data, predicted_flags)
```

---

## Import Patterns Summary

```python
# I/O
from rfi_toolbox.io import MSLoader

# Preprocessing
from rfi_toolbox.preprocessing import Preprocessor

# Evaluation
from rfi_toolbox.evaluation import (
    evaluate_segmentation,
    compute_iou,
    compute_ffi,
    compute_statistics
)

# Data Generation
from rfi_toolbox.data_generation import SyntheticDataGenerator

# Datasets
from rfi_toolbox.datasets import RFIMaskDataset, TorchDataset, BatchWriter

# Models (UNet reference implementation)
from rfi_toolbox.models.unet import UNet
```

---

## Configuration Loading

```python
from rfi_toolbox.config import ConfigLoader

# Load YAML config
config = ConfigLoader.load('config.yaml')

# Access nested values
batch_size = config.training.batch_size
patch_size = config.preprocessing.patch_size
```

---

## Tips for External Researchers

1. **Start minimal**: Use `MSLoader` + `Preprocessor` + `evaluate_segmentation` - that's 80% of use cases
2. **Normalize appropriately**: Synthetic data uses physical scales (no normalization), real data uses `stretch='SQRT'`
3. **Match patch sizes**: Inference must use same patch size as training
4. **Use FFI for real data**: When you don't have ground truth, FFI tells you if flagging improved data quality
5. **Check imports work**: `python -c "from rfi_toolbox.io import MSLoader; print('✓ Works')"`

---

## Common Patterns

### Pattern: Training Loop with rfi_toolbox

```python
from rfi_toolbox.datasets import RFIMaskDataset
from rfi_toolbox.evaluation import evaluate_segmentation
from torch.utils.data import DataLoader

# Load datasets
train_dataset = RFIMaskDataset('train/', normalization='global_min_max')
val_dataset = RFIMaskDataset('val/', normalization='global_min_max')

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Training loop
for epoch in range(num_epochs):
    for inputs, masks in train_loader:
        # Your training code
        loss = train_step(model, inputs, masks)

    # Validation
    all_preds, all_masks = [], []
    for inputs, masks in val_loader:
        preds = model(inputs)
        all_preds.append(preds)
        all_masks.append(masks)

    # Evaluate
    metrics = evaluate_segmentation(
        np.concatenate(all_preds),
        np.concatenate(all_masks)
    )
    print(f"Epoch {epoch}: IoU={metrics['iou']:.3f}")
```

---

## Need More Help?

- **Integration Guide**: See `docs/INTEGRATION_GUIDE.md` (if available)
- **Examples**: See `examples/` directory for standalone scripts
- **Issues**: Report at https://github.com/preshanth/rfi_toolbox/issues
