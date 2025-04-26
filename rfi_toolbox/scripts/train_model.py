# rfi_toolbox/scripts/train_model.py
import argparse
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.amp as amp
from datetime import datetime
from rfi_toolbox.models.unet import UNet,UNetBigger, UNetOverfit
import albumentations as A
from albumentations.pytorch import ToTensorV2

class TrainingRFIMaskDataset(Dataset):
    def __init__(self, data_dir, normalized_data_dir=None, transform=None, normalization=None, augment=True):
        self.data_dir = data_dir
        self.normalized_data_dir = normalized_data_dir
        self.transform = transform
        self.normalization = normalization
        self.augment = augment
        self.input_files = []
        self.mask_files = []

        input_base_dir = normalized_data_dir if normalized_data_dir else data_dir

        for root, _, files in os.walk(input_base_dir):
            for filename in files:
                if filename == 'input.npy':
                    self.input_files.append(os.path.join(root, filename))
                    relative_path = os.path.relpath(root, input_base_dir)
                    mask_path = os.path.join(data_dir, relative_path, 'rfi_mask.npy') # Assuming masks are in the original structure
                    if os.path.exists(mask_path):
                        self.mask_files.append(mask_path)
                    else:
                        print(f"Warning: Corresponding mask not found for {os.path.join(root, filename)}")

        # Ensure we have corresponding input and mask files
        self.samples = list(zip(self.input_files, self.mask_files))
        


        if self.augment:
            self.augmentation = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
                # A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
                # A.GaussNoise(var_limit=(0.001, 0.01), p=0.2),
                ToTensorV2(), # Convert to PyTorch tensors (Albumentations uses NumPy arrays)
            ])
        else:
            self.to_tensor = ToTensorV2()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_path, mask_path = self.samples[idx]
        input_np = np.load(input_path)
        mask = np.load(mask_path)

        #print(f"Input shape: {input_np.shape}, Mask shape: {mask.shape}, Mask dtype: {mask.dtype}")

        if mask.dtype == np.bool_:
            mask = mask.astype(np.uint8)

        if self.augment:
            # Transpose to (H, W, C) for Albumentations
            input_np_transposed = input_np.transpose(1, 2, 0)
            augmented = self.augmentation(image=input_np_transposed, mask=mask)
            input_tensor = augmented['image'].float() # ToTensorV2 will make it (C, H, W)
            mask_tensor = augmented['mask'].unsqueeze(0).float()
        else:
            input_tensor = torch.tensor(input_np, dtype=torch.float32).float() # Keep original (C, H, W)
            mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).float()

        return input_tensor, mask_tensor

def main():
    parser = argparse.ArgumentParser(description="Train a UNet model for RFI masking")
    parser.add_argument("--train_dir", type=str, default="rfi_dataset/train", help="Path to training data directory")
    parser.add_argument("--val_dir", type=str, default="rfi_dataset/val", help="Path to validation data directory")
    parser.add_argument("--normalized_data_dir", type=str, default=None, help="Path to the directory containing pre-normalized input data")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs (total if not resuming)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda or cpu)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--in_channels", type=int, default=8, help="Number of input channels to the UNet")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to a checkpoint file to resume training from")
    parser.add_argument("--new_lr", type=float, default=None, help="Optional new learning rate when resuming")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay (L2 regularization) strength")
    parser.add_argument("--normalization", type=str, default=None,
                        choices=['global_min_max', 'standardize', 'robust_scale', None],
                        help="Normalization scheme to use for input data (if --normalized_data_dir is not set)")
    parser.add_argument("--augment", action='store_true', help="Apply data augmentation during training")
    args = parser.parse_args()

    # Initialize datasets
    train_dataset = TrainingRFIMaskDataset(args.train_dir, normalized_data_dir=args.normalized_data_dir, normalization=args.normalization, augment=args.augment)
    val_dataset = TrainingRFIMaskDataset(args.val_dir, normalized_data_dir=args.normalized_data_dir, normalization=args.normalization) # should I add augmentation here ?
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = UNetOverfit(in_channels=args.in_channels).to(args.device)
    criterion = nn.BCEWithLogitsLoss()

    def dice_loss(pred, target, smooth=1.):
        pred = torch.sigmoid(pred)
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        return 1 - ((2. * intersection + smooth) /
                  (iflat.sum() + tflat.sum() + smooth))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = amp.GradScaler(enabled=torch.cuda.is_available())

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        tqdm_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}", leave=False)
        for data, mask in tqdm_bar:
            data, mask = data.to(args.device), mask.to(args.device)

            optimizer.zero_grad()

            with amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=torch.cuda.is_available()):
                output = model(data)
                loss = criterion(output, mask) + dice_loss(output, mask)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            tqdm_bar.set_postfix(loss=loss.item())

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, mask in val_loader:
                data, mask = data.to(args.device), mask.to(args.device)
                with amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=torch.cuda.is_available()):
                    output = model(data)
                    loss = criterion(output, mask) + dice_loss(output, mask)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{args.num_epochs}] - Train Loss: {total_loss/len(train_loader):.4f} - Val Loss: {avg_val_loss:.4f}")

        # Save checkpoint if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(args.checkpoint_dir, f"unet_rfi_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'args': args
            }, checkpoint_path)
            print(f"Model saved to {checkpoint_path}")

    print("Training finished.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    # Save final model
    final_model_path = os.path.join(args.checkpoint_dir, "unet_rfi_final.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': args
    }, final_model_path)
    print(f"Final model saved to {final_model_path}")

if __name__ == "__main__":
    from datetime import datetime
    main()
