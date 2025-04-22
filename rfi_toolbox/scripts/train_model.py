# rfi_toolbox/scripts/train_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
import numpy as np
from rfi_toolbox.models.unet import UNet
from rfi_toolbox.scripts.generate_dataset import RFIMaskDataset
import torch.cuda.amp as amp
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Train a UNet model for RFI masking")
    parser.add_argument("--train_dir", type=str, default="rfi_dataset/train", help="Path to training data directory")
    parser.add_argument("--val_dir", type=str, default="rfi_dataset/val", help="Path to validation data directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda or cpu)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--in_channels", type=int, default=8, help="Number of input channels to the UNet")

    args = parser.parse_args()

    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Load datasets
    train_dataset = RFIMaskDataset(args.train_dir)
    val_dataset = RFIMaskDataset(args.val_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = UNet(in_channels=args.in_channels).to(args.device)
    criterion = nn.BCEWithLogitsLoss()

    # Dice Loss (assuming you have this defined elsewhere or will define it)
    def dice_loss(pred, target, smooth=1.):
        pred = torch.sigmoid(pred)
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        return 1 - ((2. * intersection + smooth) /
                  (iflat.sum() + tflat.sum() + smooth))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5) # weight decay

    scaler = amp.GradScaler()

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        tqdm_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}", leave=False)
        for data, mask in tqdm_bar:
            data, mask = data.to(args.device), mask.to(args.device)

            optimizer.zero_grad()

            with amp.autocast():
                output = model(data)
                loss = criterion(output, mask) + dice_loss(output, mask)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            tqdm_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for val_data, val_mask in val_loader:
                val_data, val_mask = val_data.to(args.device), val_mask.to(args.device)
                with amp.autocast():
                    val_output = model(val_data)
                    val_loss = criterion(val_output, val_mask) + dice_loss(val_output, val_mask)
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{args.num_epochs}] - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        # Save checkpoint if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(args.checkpoint_dir, f"unet_rfi_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved to {checkpoint_path}")

if __name__ == "__main__":
    main()