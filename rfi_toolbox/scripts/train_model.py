# rfi_toolbox/scripts/train_model.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from datetime import datetime
from rfi_toolbox.models.unet import UNet
from rfi_toolbox.scripts.generate_dataset import RFIMaskDataset # Assuming dataset is in the same scripts dir for now

# Loss function
def dice_loss(pred, target, smooth=1.):
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

def main():
    parser = argparse.ArgumentParser(description="Train the RFI masking UNet model.")
    parser.add_argument("--train_dir", type=str, default="rfi_dataset/train", help="Path to the training data directory")
    parser.add_argument("--val_dir", type=str, default="rfi_dataset/val", help="Path to the validation data directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda or cpu)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--in_channels", type=int, default=8, help="Number of input channels to the UNet")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    model_save_path = os.path.join(args.checkpoint_dir, f"unet_rfi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")

    # Data
    train_dataset = RFIMaskDataset(args.train_dir)
    val_dataset = RFIMaskDataset(args.val_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model
    model = UNet(in_channels=args.in_channels, out_channels=1).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0

        tqdm_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}", leave=False)
        for data, mask in tqdm_bar:
            data, mask = data.to(args.device), mask.to(args.device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, mask) + dice_loss(output, mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            tqdm_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{args.num_epochs}] - Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, mask in val_loader:
                data, mask = data.to(args.device), mask.to(args.device)
                output = model(data)
                loss = criterion(output, mask) + dice_loss(output, mask)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{args.num_epochs}] - Val Loss: {avg_val_loss:.4f}")

        # Save checkpoint every few epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()
