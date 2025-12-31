# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RFI Toolbox is a Python package for generating synthetic Radio Frequency Interference (RFI) datasets and training machine learning models for RFI masking. The goal is to create a uniform API for datasets and enable easy integration with ML platforms like Hugging Face, while supporting both CNNs and segmentation models like SAM.

## Development Commands

### Installation
```bash
# Development installation (editable)
pip install -e .

# Standard installation
pip install .
```

### Dataset Generation
```bash
# Generate synthetic RFI dataset
generate_rfi_dataset --samples_training 5000 --samples_validation 1000 --output_dir my_rfi_data

# Generate from Measurement Set
generate_rfi_dataset --use_ms --ms_name /path/to/data.ms --output_dir ms_data --train_field 0 --val_field 1

# Normalize existing dataset
normalize_rfi_data --input_dir my_rfi_data --output_dir my_rfi_data_normalized
```

### Model Training and Evaluation
```bash
# Train UNet model
train_rfi_model --train_dir my_rfi_data/train --val_dir my_rfi_data/val --num_epochs 100

# Resume training from checkpoint
train_rfi_model --checkpoint_path checkpoints/model.pt --num_epochs 150

# Evaluate trained model
evaluate_rfi_model --model_path checkpoints/unet_rfi_latest.pt --dataset_dir my_rfi_data/val

# Visualize data and predictions
visualize_rfi_data --dataset_dir my_rfi_data/val --model_path checkpoints/unet_rfi_best.pt
```

## Architecture Overview

### Core Components

**RFI Simulator (`rfi_toolbox/core/simulator.py`)**
- `RFISimulator` class generates synthetic RFI data with various interference patterns
- Supports 4 polarizations (RR, RL, LR, LL) in complex format
- Generates broadband, narrowband, time-bursty, and sweeping RFI patterns
- Primary method: `generate_rfi()` returns tuple of (tf_plane_dict, mask_array)

**UNet Models (`rfi_toolbox/models/unet.py`)**
- Multiple UNet architectures: `UNet`, `UNetBigger`, `UNetOverfit`, `UNetDifferentActivation`
- Standard encoder-decoder with skip connections for segmentation
- Configurable input channels (default: 8) and features
- Designed for binary RFI mask prediction

**Dataset Management (`rfi_toolbox/scripts/generate_dataset.py`)**
- `RFIMaskDataset` class handles both synthetic and Measurement Set data
- Supports multiple normalization strategies: global_min_max, standardize, robust_scale
- Converts complex visibility data to 8-channel real format for training
- Field selection capability for MS data

### Data Pipeline

1. **Data Generation**: RFISimulator creates complex visibility data with RFI patterns
2. **Preprocessing**: Complex data converted to 8-channel format (real/imag for each polarization)
3. **Normalization**: Applied per-sample or globally depending on method
4. **Training**: UNet models trained on preprocessed data for mask prediction
5. **Evaluation**: Multiple metrics (Dice, IoU, precision, recall, F1) track performance

### Key Configuration

**SAM2 Integration (`rfi_toolbox/configs/training_rfi_sam2.yaml`)**
- Configuration for training SAM2 models on RFI data
- References external SAM-RFI repository for segmentation model retraining
- Supports normalized data pipeline with configurable image sizes

### Data Format Standards

- **Input**: Complex visibilities stored as 8-channel tensors (RR_real, RR_imag, RL_real, RL_imag, LR_real, LR_imag, LL_real, LL_imag)
- **Output**: Binary masks indicating RFI presence
- **Storage**: NumPy .npy files for data and masks
- **Naming**: Consistent sample_XXXXX.npy format for data/mask pairs

### Integration Points

- **Hugging Face**: Dataset format designed for easy upload to HF datasets
- **SAM Integration**: References `../SAM-RFI` for segmentation model workflows
- **CASA Tools**: Optional integration with casacore/casatools for MS data reading
- **Visualization**: Bokeh-based interactive dashboards for data exploration

## Development Notes

- No formal test suite currently implemented
- Dependencies managed via pyproject.toml (preferred) and setup.py (legacy)
- Platform-specific CASA tool handling (casacore for Linux, casatools for macOS)
- All console scripts defined as entry points for easy CLI access