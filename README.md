# RFI Toolbox

This Python package provides tools for generating synthetic Radio Frequency Interference (RFI) datasets for training and evaluating machine learning models, particularly for RFI masking.

## Installation

To install the RFI Toolbox, navigate to the directory containing the `setup.py` file and run:

```bash
pip install .
```
or for development purposes:

```bash

pip install -e .
```

This will install the necessary dependencies (numpy, matplotlib, tqdm, torch) and make the command-line script generate_rfi_dataset available.
Generating the Dataset

The generate_rfi_dataset script is used to create the synthetic RFI dataset. You can customize the number of training and validation samples, the output directory, and the dimensions of the time-frequency (TF) planes.
Usage

```bash

generate_rfi_dataset [options]

```
Options

    `--samples_training` <integer>: Number of training samples to generate (default: 1000).
    `--samples_validation` <integer>: Number of validation samples to generate (default: 200).
    `--output_dir` <path>: Output directory for the generated dataset (default: rfi_dataset).
    `--time_bin`s <integer>: Number of time bins in the TF plane (default: 1024).
    `--frequency_bins` <integer>: Number of frequency bins in the TF plane (default: 1024).
    `--generate_mask`: Flag to enable the generation of RFI masks (default: True - mask is always generated).
    `--no_generate_mask`: Flag to disable the generation of RFI masks. If this flag is present, only the RFI-affected time-frequency data will be saved.

Example

To generate 5000 training samples and 1000 validation samples with TF plane dimensions of 512x512 in a directory named my_rfi_data, run:
```bash

generate_rfi_dataset --samples_training 5000 --samples_validation 1000 --output_dir my_rfi_data --time_bins 512 --frequency_bins 512
```

To generate the same dataset but without the mask files:
```bash

generate_rfi_dataset --samples_training 5000 --samples_validation 1000 --output_dir my_rfi_data --time_bins 512 --frequency_bins 512 --no_mask
```

The generated data will be saved as `.npy` files within the specified output directory, organized into `train` and `val` subdirectories. For each sample, you will find `XXXX_input.npy` containing the stacked real and imaginary parts of the four polarizations (shape: 8 x time_bins x frequency_bins), and, XXXX_mask.npy containing the boolean RFI mask (shape: time_bins x frequency_bins) unless disabled with `--no_mask`.


## Evaluation Script (`rfi_toolbox/scripts/evaluate_model.py`)

This script evaluates a trained RFI masking model on a specified validation dataset and reports performance metrics such as Dice score, IoU, precision, recall, and F1 score.

### Usage

```bash
python -m rfi_toolbox.scripts.evaluate_model --model_path <path_to_model_checkpoint> --dataset_dir <path_to_validation_data> [options]
```

Arguments

    --model_path <path_to_model_checkpoint>: (Required) The path to the saved .pt file of your trained model.
    --dataset_dir <path_to_validation_data>: (Required) The path to the directory containing your validation dataset (the root directory that has the 'val' subdirectory).

Options

    --batch_size <int>: Batch size to use during evaluation (default: 8).
    --device <str>: The PyTorch device to use ('cuda' or 'cpu', defaults to CUDA if available).
    --in_channels <int>: The number of input channels the UNet model was trained with (default: 8).

Example

To evaluate a model located at checkpoints/unet_rfi_20250414_200000.pt on a validation dataset in rfi_dataset/val, run:
```bash

python -m rfi_toolbox.scripts.evaluate_model --model_path checkpoints/unet_rfi_20250414_200000.pt --dataset_dir rfi_dataset/val --batch_size 16 --device cuda
```

Output

The script will print the evaluation results to the console, including the average Dice score, IoU, precision, recall, and F1 score calculated on the validation dataset.
```
Evaluation Results:
dice: 0.8523
iou: 0.7432
precision: 0.8876
recall: 0.8211
f1: 0.8531
```

## Interactive Visualization (`rfi_toolbox/visualization/visualize.py`)

This script provides an interactive Bokeh dashboard to visualize a random subset of the validation dataset and, optionally, the predictions of a trained RFI masking model.

Usage
```bash

python -m rfi_toolbox.visualization.visualize --dataset_dir <path_to_validation_data> [options]
```
Arguments

    --dataset_dir <path_to_validation_data>: (Required) The path to the directory containing your validation dataset (the root directory with the 'val' subdirectory).

Options

    --model_path <path_to_model_checkpoint>: Path to the saved .pt file of your trained model. If provided, the model's predictions will also be displayed (default: None).
    --device <str>: The PyTorch device to use for model inference ('cuda' or 'cpu', default: 'cpu').
    --in_channels <int>: The number of input channels the UNet model expects (default: 8).
    --num_samples <int>: The number of random validation samples to visualize (default: 100).
    --seed <int>: Random seed for selecting the subset of samples (default: 42).

Example

To visualize 50 random samples from the validation dataset located at rfi_dataset/val:
```bash

python -m rfi_toolbox.visualization.visualize --dataset_dir rfi_dataset/val --num_samples 50
```
To visualize 100 random samples and the predictions of a model located at checkpoints/unet_rfi_latest.pt, using a CUDA-enabled GPU if available:
```bash

python -m rfi_toolbox.visualization.visualize --dataset_dir rfi_dataset/val --model_path checkpoints/unet_rfi_latest.pt --device cuda --num_samples 100
```
Interaction

Running the script will open an interactive dashboard in your web browser. You can use the slider at the top to navigate through the randomly selected validation samples. For each sample, you will see:

    Input RR Amp: Amplitude of the RR polarization.
    Input RL Amp: Amplitude of the RL polarization.
    Input LR Amp: Amplitude of the LR polarization.
    Input LL Amp: Amplitude of the LL polarization.
    Ground Truth Mask: The actual RFI mask for the sample.
    Model Prediction: If a model_path is provided, this shows the model's predicted RFI mask (after applying a sigmoid).

This interactive tool allows for quick visual inspection of the data and the model's performance on a subset of the validation set.