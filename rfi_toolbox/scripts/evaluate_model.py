"""
Evaluate a trained RFI masking model on a validation dataset.

Uses canonical metrics from rfi_toolbox.evaluation.metrics.
"""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from rfi_toolbox.evaluation import evaluate_segmentation
from rfi_toolbox.models.unet import UNet
from rfi_toolbox.datasets import RFIMaskDataset


def evaluate_model(model_path, dataset_dir, batch_size=4, device="cpu", in_channels=8):
    """
    Evaluates a trained RFI masking model on a given dataset.

    Args:
        model_path (str): Path to the saved model checkpoint.
        dataset_dir (str): Path to the validation dataset directory.
        batch_size (int): Batch size for evaluation.
        device (str): Device to use for evaluation (cuda or cpu).
        in_channels (int): Number of input channels the model expects.

    Returns:
        dict: Dictionary containing average metrics (iou, precision, recall, f1, dice)
    """
    val_dataset = RFIMaskDataset(dataset_dir)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model = UNet(in_channels=in_channels, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_metrics = []

    with torch.no_grad():
        for data, mask in val_loader:
            data, mask = data.to(device), mask.to(device)
            output = model(data)
            output_sigmoid = torch.sigmoid(output)

            # Apply threshold to get binary predictions
            pred_binary = (output_sigmoid > 0.5).float()

            # Compute metrics using canonical evaluation module
            metrics = evaluate_segmentation(pred_binary, mask)
            all_metrics.append(metrics)

    # Average metrics across all batches
    avg_metrics = {
        metric: np.mean([m[metric] for m in all_metrics]) for metric in all_metrics[0].keys()
    }

    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained RFI masking model.")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the saved model checkpoint"
    )
    parser.add_argument(
        "--dataset_dir", type=str, required=True, help="Path to the validation dataset directory"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for evaluation (cuda or cpu)",
    )
    parser.add_argument(
        "--in_channels", type=int, default=8, help="Number of input channels the model expects"
    )
    args = parser.parse_args()

    results = evaluate_model(
        args.model_path, args.dataset_dir, args.batch_size, args.device, args.in_channels
    )

    print("Evaluation Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
