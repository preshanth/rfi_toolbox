# rfi_toolbox/utils/evaluation.py
import torch
from torch.utils.data import DataLoader
from rfi_toolbox.scripts.generate_dataset import RFIMaskDataset  # Adjust import path as needed
from rfi_toolbox.models.unet import UNet  # Adjust import path as needed
import numpy as np

def dice_coefficient(pred, target, smooth=1.):
    pred_flat = (pred > 0.5).float().view(-1)
    target_flat = target.float().view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

def iou_coefficient(pred, target, smooth=1.):
    pred_flat = (pred > 0.5).float().view(-1)
    target_flat = target.float().view(-1)
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def precision_recall_f1(pred, target):
    pred_binary = (pred > 0.5).int().view(-1).cpu().numpy()
    target_binary = target.int().view(-1).cpu().numpy()

    tp = np.sum(pred_binary * target_binary)
    fp = np.sum(pred_binary * (1 - target_binary))
    fn = np.sum((1 - pred_binary) * target_binary)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return precision, recall, f1

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
        dict: A dictionary containing the average Dice score, IoU, precision, recall, and F1 score.
    """
    val_dataset = RFIMaskDataset(dataset_dir)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model = UNet(in_channels=in_channels, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dice_scores = []
    iou_scores = []
    precisions = []
    recalls = []
    f1_scores = []

    with torch.no_grad():
        for data, mask in val_loader:
            data, mask = data.to(device), mask.to(device)
            output = model(data)
            output_sigmoid = torch.sigmoid(output)

            dice = dice_coefficient(output_sigmoid, mask).item()
            iou = iou_coefficient(output_sigmoid, mask).item()
            precision, recall, f1 = precision_recall_f1(output_sigmoid, mask)

            dice_scores.append(dice)
            iou_scores.append(iou)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1_scores)

    return {
        "dice": avg_dice,
        "iou": avg_iou,
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1,
    }

if __name__ == "__main__":
    # Example usage:
    model_path = "checkpoints/unet_rfi_YYYYMMDD_HHMMSS.pt"  # Replace with your model path
    val_dir = "rfi_dataset/val"  # Replace with your validation data path
    results = evaluate_model(model_path, val_dir, batch_size=8, device="cuda")
    print("Evaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
