"""
Command-line interface (CLI) scripts for RFI Toolbox.

Provides entry points for:
- generate_rfi_dataset: Generate synthetic training data
- train_rfi_model: Train UNet models
- evaluate_rfi_model: Evaluate trained models
- visualize_rfi_data: Interactive visualization
- normalize_rfi_data: Data normalization

Usage:
    $ generate_rfi_dataset --samples_training 1000 --output_dir ./data
    $ train_rfi_model --train_dir ./data/train --num_epochs 50
    $ evaluate_rfi_model --model_path model.pt --dataset_dir ./data/val
"""

__all__ = []