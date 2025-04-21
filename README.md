# RFI Toolbox

This Python package provides tools for generating synthetic Radio Frequency Interference (RFI) datasets for training and evaluating machine learning models, particularly for RFI masking. It also includes interactive visualization capabilities.

## Installation

To install the RFI Toolbox, navigate to the directory containing the `setup.py` file and run:

```bash
pip install .
```
or for development purposes:

```bash

pip install -e .
```

This will install the necessary dependencies (numpy, matplotlib, tqdm, torch, bokeh) and make the command-line script generate_rfi_dataset available.
This will install the necessary dependencies (numpy, matplotlib, tqdm, torch, bokeh) and make the command-line scripts generate_rfi_dataset, evaluate_rfi_model, and visualize_rfi_data available.
Generating the Dataset

The generate_rfi_dataset script is used to create synthetic RFI datasets as NumPy .npy files.

```bash

generate_rfi_dataset [options]
```
Options

    `--samples_training` <integer>: Number of training samples to generate (default: 1000).
    `--samples_validation` <integer>: Number of validation samples to generate (default: 200).
    `--output_dir` <path>: Output directory for the generated dataset (default: rfi_dataset).
    `--time_bins` <integer>: Number of time bins in the TF plane (default: 1024).
    `--frequency_bins` <integer>: Number of frequency bins in the TF plane (default: 1024).
    `--generate_mask`: Flag to enable the generation of RFI masks (default: True).
    `--no_generate_mask`: Flag to disable the generation of RFI masks.

Example
```bash

generate_rfi_dataset --samples_training 5000 --samples_validation 1000 --output_dir my_rfi_data --time_bins 512 --frequency_bins 512
```

Training the Model

The train_rfi_model script is used to train the UNet model for RFI masking.

```bash

train_rfi_model [options]
```
Options

    --train_dir <path>: Path to the training data directory (default: rfi_dataset/train).
    --val_dir <path>: Path to the validation data directory (default: rfi_dataset/val).
    --batch_size <int>: Batch size for training (default: 4).
    --num_epochs <int>: Number of training epochs (default: 50).
    --lr <float>: Learning rate (default: 1e-4).
    --device <str>: Device to use (cuda or cpu, default: cuda if available).
    --checkpoint_dir <path>: Directory to save model checkpoints (default: checkpoints).
    --in_channels <int>: Number of input channels to the UNet (default: 8).

Example
```bash

train_rfi_model --train_dir my_rfi_data/train --val_dir my_rfi_data/val --num_epochs 100 --batch_size 8 --lr 5e-5 --device cuda
```
Evaluating the Model

The evaluate_rfi_model script evaluates a trained model on a validation dataset.

```bash

evaluate_rfi_model --model_path <path_to_model_checkpoint> --dataset_dir <path_to_validation_data> [options]
```
Arguments

    `--model_path` <path_to_model_checkpoint>: (Required) The path to the saved .pt file of your trained model.
    `--dataset_dir` <path_to_validation_data>: (Required) The path to the directory containing your validation dataset.

Options

    `--batch_size` <int>: Batch size to use during evaluation (default: 8).
    `--device` <str>: The PyTorch device to use ('cuda' or 'cpu', default: cuda if available).
    `--in_channels` <int>: The number of input channels the model expects (default: 8).

Example
```bash

evaluate_rfi_model --model_path checkpoints/unet_rfi_latest.pt --dataset_dir my_rfi_data/val --batch_size 16 --device cuda
```
Interactive Visualization

The visualize_rfi_data script provides an interactive Bokeh dashboard to visualize a random subset of the validation dataset and, optionally, model predictions.

```bash

visualize_rfi_data --dataset_dir <path_to_validation_data> [options]
```
Arguments

    `--dataset_dir` <path_to_validation_data>: (Required) The path to the directory containing your validation dataset.

Options

    `--model_path` <path_to_model_checkpoint>: Path to the saved .pt file of your trained model (default: None).
    `--device` <str>: The PyTorch device to use for model inference ('cuda' or 'cpu', default: 'cpu').
    `--in_channels` <int>: The number of input channels the model expects (default: 8).
    `--num_samples` <int>: The number of random validation samples to visualize (default: 100).
    `--seed` <int>: Random seed for selecting the subset of samples (default: 42).

Example
```bash

visualize_rfi_data --dataset_dir my_rfi_data/val --model_path checkpoints/unet_rfi_best.pt --device cuda --num_samples 50
```
