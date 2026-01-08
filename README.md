# RFI Toolbox


A Python package providing core infrastructure for Radio Frequency Interference (RFI) detection and evaluation in radio astronomy. This package serves as the foundation for machine learning-based RFI flagging systems, providing data I/O, preprocessing, synthetic data generation, and standardized evaluation metrics.

## Overview

RFI Toolbox is designed as a modular, framework-agnostic foundation for RFI detection research. It handles the domain-specific complexities of radio astronomy data (CASA measurement sets, complex visibilities, physical units) while remaining agnostic to the machine learning framework used (PyTorch, TensorFlow, JAX, etc.).

**Design Philosophy:**
- **Framework Independence**: Core functionality works without ML dependencies
- **Modular Architecture**: Use only the components you need
- **Physical Accuracy**: Preserves radio astronomy units and scales
- **Reproducible Evaluation**: Standardized metrics for method comparison

---

## Installation

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Optional: CASA tools (for measurement set operations)
- Optional: PyTorch (for dataset management)

### Basic Installation

```bash
# Clone repository
git clone https://github.com/preshanth/rfi_toolbox.git
cd rfi_toolbox

# Install core package
pip install .
```

### Installation with Optional Dependencies

```bash
# CASA support (measurement set I/O)
pip install .[casa]

# ML/Training tools (PyTorch, scikit-learn)
pip install .[training]

# Visualization (Bokeh, matplotlib)
pip install .[viz]

# All optional dependencies
pip install .[all]

# Development (testing and linting)
pip install .[dev]
```

**Dependency groups:**
- **Core (default)**: NumPy, SciPy, tqdm - provides evaluation metrics, synthetic generation, preprocessing
- **`[casa]`**: python-casacore (Linux) or casatools (macOS) - enables measurement set I/O
- **`[training]`**: PyTorch, albumentations - dataset management and augmentation
- **`[viz]`**: Bokeh, matplotlib - interactive visualization tools
- **`[dev]`**: pytest, black, ruff, pytest-cov - testing and code quality
- **`[all]`**: All of the above

---

## Quick Start

### For ML Researchers

RFI Toolbox provides data pipeline infrastructure independent of your ML framework choice:

```python
from rfi_toolbox.io import MSLoader
from rfi_toolbox.preprocessing import Preprocessor
from rfi_toolbox.evaluation import evaluate_segmentation

# 1. Load measurement set data
loader = MSLoader('observation.ms')
loader.load(num_antennas=10, mode='DATA')
# Output: Complex visibilities (baselines, pols, channels, times)

# 2. Preprocess into ML-ready format
preprocessor = Preprocessor(loader.data, flags=loader.load_flags())
dataset = preprocessor.create_dataset(
    patch_size=128,
    stretch='SQRT',
    normalize_before_stretch=True
)

# 3. Run your custom model (any framework)
def my_rfi_detector(data):
    # Your ML model here (PyTorch, TensorFlow, JAX, etc.)
    # Input: (N, 3, H, W) - 3-channel waterfall patches
    # Output: (N, H, W) - Binary mask predictions
    return predicted_mask

predictions = my_rfi_detector(dataset.get_batch())

# 4. Evaluate using standardized metrics
metrics = evaluate_segmentation(predictions, ground_truth)
print(f"IoU: {metrics['iou']:.3f}, F1: {metrics['f1']:.3f}")

# 5. Save flags back to measurement set
loader.save_flags(predictions)
loader.close()
```

### For Data Pipeline Development

Generate synthetic training data with exact ground truth:

```python
from rfi_toolbox.data_generation import SyntheticDataGenerator
from rfi_toolbox.evaluation import compute_ffi

# Generate synthetic RFI with known ground truth
generator = SyntheticDataGenerator(config_path='configs/synthetic.yaml')
waterfall, ground_truth, rfi_params = generator.generate_single_sample(
    num_channels=1024,
    num_times=1024,
    noise_level=1.0,      # 1 mJy noise
    rfi_power_min=1000.0, # 1000 Jy RFI
    rfi_power_max=10000.0
)

# Evaluate flagging quality
predicted_flags = run_flagging_algorithm(waterfall)
ffi_metrics = compute_ffi(waterfall, predicted_flags)
print(f"FFI: {ffi_metrics['ffi']:.3f}")
print(f"MAD Reduction: {ffi_metrics['mad_reduction']:.3f}")
```

---

## Core Modules

### I/O Operations (`rfi_toolbox.io`)

Load and manipulate CASA measurement sets:

```python
from rfi_toolbox.io import MSLoader, inject_synthetic_data

# Load measurement set
loader = MSLoader('observation.ms', field_id=0)
loader.load(num_antennas=5, mode='DATA')

# Access data
data = loader.data              # (baselines, pols, channels, times)
magnitude = loader.magnitude    # Magnitude of complex visibilities
flags = loader.load_flags()     # Existing FLAG column

# Save new flags
loader.save_flags(predicted_flags, column='FLAG')

# Inject synthetic data for validation
inject_synthetic_data(
    template_ms_path='template.ms',
    synthetic_data=waterfall,
    output_ms_path='synthetic.ms',
    baseline_map=[(0,1), (0,2), (1,2)]
)
```

**Available functions:**
- `MSLoader`: Load CASA measurement sets with spectral window combination
- `inject_synthetic_data()`: Replace DATA column with synthetic visibilities

### Preprocessing (`rfi_toolbox.preprocessing`)

Convert raw visibilities to ML-ready format:

```python
from rfi_toolbox.preprocessing import Preprocessor

preprocessor = Preprocessor(
    data,                          # Complex visibilities
    flags=existing_flags           # Optional FLAG column
)

dataset = preprocessor.create_dataset(
    patch_size=128,                # Patch dimensions
    stretch='SQRT',                # Amplitude stretch (SQRT/LOG10/None)
    normalize_before_stretch=True, # ImageNet normalization
    use_custom_flags=True,         # Use provided flags as ground truth
    flag_sigma=5,                  # MAD-based auto-flagging threshold
    enable_augmentation=True       # 4-way rotation augmentation
)

# Dataset structure:
# - dataset['data']: (N, 3, H, W) - 3-channel waterfall patches
# - dataset['labels']: (N, H, W) - Binary ground truth masks
# - dataset['metadata']: Patch coordinates and parameters
```

**3-Channel Feature Extraction:**
1. **Spatial Gradient**: Edge detection (Sobel filter on magnitude)
2. **Log Amplitude**: Intensity representation (scaled to [-3, 4])
3. **Phase**: Complex phase (mapped to [0, 1])

### Synthetic Data Generation (`rfi_toolbox.data_generation`)

Generate physically realistic RFI with exact ground truth:

```python
from rfi_toolbox.data_generation import SyntheticDataGenerator

generator = SyntheticDataGenerator(config_path='configs/synthetic.yaml')

# Generate single sample
waterfall, ground_truth, rfi_params = generator.generate_single_sample(
    num_channels=1024,
    num_times=1024,
    num_polarizations=4,
    noise_level=1.0,              # 1 mJy Gaussian noise
    rfi_power_min=1000.0,         # 1000 Jy RFI minimum
    rfi_power_max=10000.0,        # 10000 Jy RFI maximum
    enable_bandpass=True,         # 8th-order polynomial edge effects
    polarization_correlation=0.8  # Correlated RFI across XX/YY
)

# waterfall: (1, 4, 1024, 1024) complex128 - Visibility data
# ground_truth: (1, 4, 1024, 1024) bool - Exact RFI locations
# rfi_params: dict - RFI type, power, location metadata
```

**Supported RFI Types:**
1. **Narrowband Persistent** - Continuous narrowband signals (GPS, satellites)
2. **Broadband Persistent** - Wideband interference (power lines, harmonics)
3. **Narrowband Bursty** - Intermittent pulses (radar, transmitters)
4. **Broadband Bursty** - Transient events (lightning, arcing)
5. **Frequency Sweeps** - Linear and quadratic chirps (scanning radar)
6. **Narrowband Intermittent** - Duty-cycled narrowband (periodic radar)

**Physical Scales:**
- Noise: 1 mJy (typical system noise)
- RFI: 1000-10000 Jy (10⁶-10⁷ dynamic range)
- Bandpass: 8th-order polynomial edge rolloff
- Polarization: 0.8 correlation between XX/YY feeds

### Evaluation Metrics (`rfi_toolbox.evaluation`)

Standardized metrics for method comparison:

```python
from rfi_toolbox.evaluation import (
    evaluate_segmentation,  # All segmentation metrics
    compute_iou,           # Intersection over Union
    compute_f1,            # F1 score
    compute_precision,     # Precision (PPV)
    compute_recall,        # Recall (sensitivity)
    compute_dice,          # Dice coefficient
    compute_ffi,           # Flagging Fidelity Index
    compute_statistics,    # Data statistics
    compute_calcquality    # Calcquality metric
)

# Segmentation metrics (vs ground truth)
metrics = evaluate_segmentation(predicted_mask, ground_truth)
# Returns: {'iou': 0.85, 'precision': 0.90, 'recall': 0.82, 'f1': 0.86, 'dice': 0.86}

# Flagging quality metrics (on actual data)
ffi = compute_ffi(visibility_data, predicted_flags)
# Returns: {'ffi': 0.65, 'mad_reduction': 0.45, 'std_reduction': 0.52, 'flagged_fraction': 0.15}

# Statistical validation
stats = compute_statistics(visibility_data, flags=predicted_flags)
# Returns: {'mean': 1.2e-3, 'median': 8.5e-4, 'std': 3.4e-3, 'mad': 1.1e-3, 'flagged_fraction': 0.15}

# Calcquality metric (paper reference)
cq = compute_calcquality(visibility_data, predicted_flags)
# Returns: {'calcquality': 2.45, 'sensitivity': 0.12, 'mean_shift': 0.08, ...}
```

**Metric Definitions:**
- **IoU** (Intersection over Union): Jaccard index, [0,1], higher is better
- **Precision**: TP/(TP+FP), fraction of predicted RFI that is truly RFI
- **Recall**: TP/(TP+FN), fraction of true RFI that is detected
- **F1**: Harmonic mean of precision and recall
- **Dice**: Equivalent to F1 for binary segmentation
- **FFI** (Flagging Fidelity Index): Balances noise reduction vs over-flagging
- **Calcquality**: Combined metric from literature (lower is better)

### Dataset Management (`rfi_toolbox.datasets`)

Efficient dataset storage and loading (PyTorch-based):

```python
from rfi_toolbox.datasets import TorchDataset, BatchWriter

# Write batched dataset
writer = BatchWriter(output_dir='./datasets/train', batch_size=100)
for i in range(1000):
    waterfall, mask = generate_sample(i)
    writer.add_sample(waterfall, mask)
writer.finalize()

# Load batched dataset
dataset = TorchDataset.from_directory('./datasets/train')
print(f"Dataset size: {len(dataset)} samples")

# Use with DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)
for batch_data, batch_masks in loader:
    # Training loop
    pass
```

---

## Command-Line Tools

### Generate Synthetic Dataset

```bash
generate_rfi_dataset \
  --samples_training 5000 \
  --samples_validation 1000 \
  --output_dir my_rfi_data \
  --time_bins 1024 \
  --frequency_bins 1024
```

### Generate Dataset from Measurement Set

```bash
generate_rfi_dataset \
  --use_ms \
  --ms_name /path/to/observation.ms \
  --output_dir ms_data \
  --train_field 0 \
  --val_field 1
```

### Train UNet Model

```bash
train_rfi_model \
  --train_dir my_rfi_data/train \
  --val_dir my_rfi_data/val \
  --num_epochs 100 \
  --batch_size 8 \
  --lr 5e-5 \
  --device cuda
```

### Evaluate Model

```bash
evaluate_rfi_model \
  --model_path checkpoints/unet_rfi_latest.pt \
  --dataset_dir my_rfi_data/val \
  --batch_size 16 \
  --device cuda
```

### Visualize Data

```bash
visualize_rfi_data \
  --dataset_dir my_rfi_data/val \
  --model_path checkpoints/unet_rfi_best.pt \
  --device cuda \
  --num_samples 50
```

---

## Integration with ML Frameworks

### Example: Custom PyTorch Model

```python
import torch
import torch.nn as nn
from rfi_toolbox.io import MSLoader
from rfi_toolbox.preprocessing import Preprocessor
from rfi_toolbox.evaluation import evaluate_segmentation

# Define custom model
class MyRFIDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            # ... your architecture
        )
        self.decoder = nn.Sequential(
            # ... decoder layers
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.encoder(x)
        mask = self.decoder(features)
        return mask

# Load data
loader = MSLoader('observation.ms')
loader.load(num_antennas=5)

# Preprocess
preprocessor = Preprocessor(loader.data)
dataset = preprocessor.create_dataset(patch_size=128)

# Train model
model = MyRFIDetector().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# ... training loop ...

# Evaluate
predictions = model(torch.from_numpy(dataset['data']).cuda()).cpu().numpy()
metrics = evaluate_segmentation(predictions > 0.5, dataset['labels'])
print(metrics)
```

### Example: Framework-Agnostic Usage

```python
import jax
import jax.numpy as jnp
from rfi_toolbox.preprocessing import Preprocessor
from rfi_toolbox.evaluation import compute_f1

# Load data using rfi_toolbox
preprocessor = Preprocessor(visibility_data)
dataset = preprocessor.create_dataset(patch_size=128)

# Define JAX model
def rfi_detector_jax(params, x):
    # Your JAX model here
    return predictions

# Train using JAX
# ...

# Evaluate using standardized metrics
predictions = rfi_detector_jax(trained_params, dataset['data'])
f1_score = compute_f1(predictions, dataset['labels'])
```

---

## Configuration Files

Example configuration for synthetic data generation (`configs/synthetic.yaml`):

```yaml
synthetic:
  num_samples: 4000
  num_channels: 1024
  num_times: 1024
  num_polarizations: 4

  # Physical scales
  noise_mjy: 1.0                 # 1 mJy Gaussian noise
  rfi_power_min: 1000.0          # 1000 Jy RFI minimum
  rfi_power_max: 10000.0         # 10000 Jy RFI maximum

  # RFI type counts per sample
  rfi_type_counts:
    narrowband_persistent: 20
    broadband_persistent: 5
    frequency_sweep: 1
    narrowband_bursty: 20
    broadband_bursty: 5
    narrowband_intermittent: 5

  # Physical effects
  enable_bandpass_rolloff: true
  bandpass_polynomial_order: 8
  polarization_correlation: 0.8

processing:
  patch_size: 1024
  stretch: null                  # None/SQRT/LOG10
  normalize_before_stretch: false
  normalize_after_stretch: false
  enable_augmentation: true      # 4-way rotation
  flag_sigma: 5                  # MAD threshold for auto-flagging
```

---

## Project Structure

```
rfi_toolbox/
├── rfi_toolbox/
│   ├── io/                      # Measurement set I/O
│   │   ├── ms_loader.py         # MSLoader class
│   │   └── ms_injection.py      # inject_synthetic_data()
│   ├── preprocessing/           # Data preprocessing
│   │   └── preprocessor.py      # Preprocessor class
│   ├── data_generation/         # Synthetic RFI generation
│   │   └── synthetic_generator.py  # SyntheticDataGenerator class
│   ├── evaluation/              # Metrics and statistics
│   │   ├── metrics.py           # Segmentation metrics
│   │   └── statistics.py        # FFI, calcquality
│   ├── datasets/                # Dataset management
│   │   └── batched_dataset.py   # TorchDataset, BatchWriter
│   ├── core/                    # Core simulators
│   │   └── simulator.py         # RFISimulator (legacy)
│   ├── models/                  # Reference models
│   │   └── unet.py              # UNet architectures
│   ├── scripts/                 # CLI entry points
│   │   ├── generate_dataset.py
│   │   ├── train_model.py
│   │   └── evaluate_model.py
│   └── visualization/           # Visualization tools
│       └── visualize.py
├── tests/                       # Test suite
├── configs/                     # Example configurations
├── pyproject.toml               # Package definition
└── README.md                    # This file
```

---

## Development

### Running Tests

```bash
# Install development dependencies
pip install .[dev]

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=rfi_toolbox --cov-report=html
```

### Code Quality

```bash
# Format code
black rfi_toolbox/ tests/ --line-length 100

# Lint code
ruff check rfi_toolbox/ tests/ --fix

# Type checking (optional)
mypy rfi_toolbox/ --ignore-missing-imports
```

---

## Related Projects

- **SAM-RFI**: SAM2-based RFI detection system built on rfi_toolbox
  - Repository: https://github.com/preshanth/SAM-RFI
  - Uses rfi_toolbox for data I/O, preprocessing, and evaluation
  - Provides SAM2 training and inference capabilities

---

## Citation

If you use this software in your research, please cite:

```bibtex
@software{rfi_toolbox2025,
  title = {RFI Toolbox: Infrastructure for Radio Frequency Interference Detection},
  author = {Jagannathan, Preshanth and Sekhar, Srikrishna and Deal, Derod},
  year = {2025},
  url = {https://github.com/preshanth/rfi_toolbox}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Support

- **Issues**: https://github.com/preshanth/rfi_toolbox/issues
- **Contact**: pjaganna@nrao.edu

---

## Acknowledgments

- **NRAO** - National Radio Astronomy Observatory
- **NAC** - National Astronomy Consortium
- **CASA** - Common Astronomy Software Applications