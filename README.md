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

The `generate_rfi_dataset` script is used to generate synthetic RFI datasets or create datasets from Measurement Sets (MS) as NumPy `.npy` files.

```bash

generate_rfi_dataset [options]
```
Options

- `--samples_training <integer>`: Number of training samples to generate (default: 1000).
- `--samples_validation <integer>`: Number of validation samples to generate (default: 200).
- `--output_dir <path>`: Output directory for the generated dataset (default: `rfi_dataset`).
- `--time_bins <integer>`: Number of time bins in the TF plane (default: 1024).
- `--frequency_bins <integer>`: Number of frequency bins in the TF plane (default: 1024).
- `--generate_mask`: Flag to enable the generation of RFI masks (default: `True`).
- `--no_generate_mask`: Flag to disable the generation of RFI masks.
- `--use_ms`: Flag to enable loading data from a Measurement Set.
- `--ms_name <path>`: Path to the Measurement Set. Required if `--use_ms` is set.
- `--train_field <integer>`: `FIELD_ID` to use for the training set when loading from an MS.
- `--val_field <integer>`: `FIELD_ID` to use for the validation set when loading from an MS.

### Generating Synthetic Data

```bash

generate_rfi_dataset --samples_training 5000 --samples_validation 1000 --output_dir my_rfi_data --time_bins 512 --frequency_bins 512
```

This command will generate 5000 training samples and 1000 validation samples, with a TF plane size of 512x512, and save the dataset to the my_rfi_data directory.

### Loading Data from a Measurement Set
```bash
generate_rfi_dataset --use_ms --ms_name /path/to/your/data.ms --output_dir ms_data --train_field 0 --val_field 1
```
This command will load data from the specified Measurement Set (`data.ms`).  It will create a dataset in the `ms_data` directory.  The data corresponding to `FIELD_ID=0` will be used for the training set, and `FIELD_ID=1` will be used for the validation set.

### Training the Model

The train_rfi_model script is used to train the UNet model for RFI masking.

```bash

train_rfi_model [options]
```
Options

    `--train_dir` <path>: Path to the training data directory (default: rfi_dataset/train).
    `--val_dir` <path>: Path to the validation data directory (default: rfi_dataset/val).
    `--batch_size` <int>: Batch size for training (default: 4).
    `--num_epochs` <int>: Number of training epochs (default: 50).
    `--lr` <float>: Learning rate (default: 1e-4).
    `--device` <str>: Device to use (cuda or cpu, default: cuda if available).
    `--checkpoint_dir` <path>: Directory to save model checkpoints (default: checkpoints).
    `--in_channels` <int>: Number of input channels to the UNet (default: 8).

Example
```bash

train_rfi_model --train_dir my_rfi_data/train --val_dir my_rfi_data/val --num_epochs 100 --batch_size 8 --lr 5e-5 --device cuda
```
### Evaluating the Model

The evaluate_rfi_model script evaluates a trained model on a validation dataset.

```bash

evaluate_rfi_model --model_path <path_to_model_checkpoint> --dataset_dir <path_to_validation_data> [options]
```
Arguments

    --model_path <path_to_model_checkpoint>: (Required) The path to the saved .pt file of your trained model.
    --dataset_dir <path_to_validation_data>: (Required) The path to the directory containing your validation dataset.

Options

    `--batch_size` <int>: Batch size to use during evaluation (default: 8).
    `--device` <str>: The PyTorch device to use ('cuda' or 'cpu', default: cuda if available).
    `--in_channels` <int>: The number of input channels the model expects (default: 8).

Example
```bash

evaluate_rfi_model --model_path checkpoints/unet_rfi_latest.pt --dataset_dir my_rfi_data/val --batch_size 16 --device cuda
```
### Interactive Visualization

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

Options for Resuming Training

To enable resuming training from a checkpoint, the following command-line arguments have been added to the `train_rfi_model` script:

* `--checkpoint_path`: Path to a `.pt` file containing a saved model checkpoint. If provided, the script will attempt to load the model's state and resume training from the epoch specified in the checkpoint.
* `--new_lr`: (Optional) A new learning rate to use when resuming training. If not provided, the learning rate from the checkpoint (if available) or the initial `--lr` will be used.

How to Resume Training

1.  **Locate your checkpoint file:** After a training run, the best model checkpoints are saved in the `checkpoints` directory. The filenames are typically timestamped (e.g., `unet_rfi_20250421_183849.pt`).

2.  **Run the training script with the `--checkpoint_path` argument:** Provide the path to the checkpoint file you want to resume from. You also need to specify the total number of epochs you want to train for using the `--num_epochs` argument.

    ```bash
    train_rfi_model --train_dir path/to/train_data --val_dir path/to/val_data --batch_size 8 --lr 5e-5 --device cuda --in_channels 8 --checkpoint_path checkpoints/your_last_checkpoint.pt --num_epochs 150
    ```

    Replace `path/to/train_data` and `path/to/val_data` with your data directories, and `checkpoints/your_last_checkpoint.pt` with the actual path to your checkpoint file. Adjust other arguments like `--batch_size`, `--lr`, and `--in_channels` as needed (they will be overridden by the checkpoint if its `args` were saved).

3.  **(Optional) Specify a new learning rate:** If you want to use a different learning rate after resuming, use the `--new_lr` argument:

    ```bash
    train_rfi_model --train_dir path/to/train_data --val_dir path/to/val_data --batch_size 8 --lr 5e-5 --device cuda --in_channels 8 --checkpoint_path checkpoints/your_last_checkpoint.pt --num_epochs 150 --new_lr 1e-6
    ```

By using these options, you can continue training your model from a specific point, which is useful for extending training runs or fine-tuning models.
