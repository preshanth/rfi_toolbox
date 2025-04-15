import argparse
import os
import logging
from tqdm import trange
import numpy as np
from rfi_toolbox.core.simulator import RFISimulator
from torch.utils.data import Dataset
import glob
import torch

class RFIMaskDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.sample_dirs = sorted([d for d in glob.glob(os.path.join(root_dir, '*')) if os.path.isdir(d)])
        self.transform = transform

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        input_stack = []
        for pol in ['RR', 'RL', 'LR', 'LL']:
            arr = np.load(os.path.join(sample_dir, f'{pol}.npy'))
            amp = np.abs(arr)
            amp = (amp - np.mean(amp)) / (np.std(amp) + 1e-6)
            input_stack.append(amp)

        input_tensor = np.stack(input_stack, axis=0)  # shape (4, H, W)
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32)

        mask_path = os.path.join(sample_dir, 'rfi_mask.npy')
        if os.path.exists(mask_path):
            mask = np.load(mask_path).astype(np.float32)
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # shape (1, H, W)
        else:
            mask = torch.zeros((1, args.time_bins, args.freq_bins), dtype=torch.float32) # Or handle as needed

        if self.transform:
            input_tensor, mask = self.transform(input_tensor, mask)

        return input_tensor, mask

def save_example_pair_npy(tf_plane, mask, index, out_dir, generate_mask=True):
    os.makedirs(out_dir, exist_ok=True)
    input_data = np.stack([
        tf_plane['RR'].real, tf_plane['RR'].imag,
        tf_plane['RL'].real, tf_plane['RL'].imag,
        tf_plane['LR'].real, tf_plane['LR'].imag,
        tf_plane['LL'].real, tf_plane['LL'].imag
    ], axis=0)  # shape: (8, time_bins, freq_bins)

    input_path = os.path.join(out_dir, f"{index:04d}_input.npy")
    np.save(input_path, input_data)

    if generate_mask:
        mask_path = os.path.join(out_dir, f"{index:04d}_mask.npy")
        np.save(mask_path, mask)

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic RFI dataset as numpy files.")
    parser.add_argument("--samples_training", type=int, default=1000, help="Number of training samples to generate")
    parser.add_argument("--samples_validation", type=int, default=200, help="Number of validation samples to generate")
    parser.add_argument("--output_dir", type=str, default="rfi_dataset", help="Output directory for the dataset")
    parser.add_argument("--time_bins", type=int, default=1024, help="Number of time bins in the TF plane")
    parser.add_argument("--frequency_bins", type=int, default=1024, help="Number of frequency bins in the TF plane")
    parser.add_argument("--generate_mask", action='store_true', default=True, help="Enable generation of RFI masks")
    parser.add_argument("--no_mask", action='store_false', dest='generate_mask', help="Disable generation of RFI masks")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    simulator = RFISimulator(time_bins=args.time_bins, freq_bins=args.freq_bins)

    # Train samples
    train_dir = os.path.join(args.output_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    logging.info(f"Generating {args.samples_training} training samples in '{train_dir}' (mask generation: {args.generate_mask})")
    for i in trange(args.samples_training, desc="Training"):
        tf_plane, mask = simulator.generate_rfi()
        save_example_pair_npy(tf_plane, mask, index=i, out_dir=train_dir, generate_mask=args.generate_mask)

    # Validation samples
    val_dir = os.path.join(args.output_dir, "val")
    os.makedirs(val_dir, exist_ok=True)
    logging.info(f"Generating {args.samples_validation} validation samples in '{val_dir}' (mask generation: {args.generate_mask})")
    for i in trange(args.samples_validation, desc="Validation"):
        tf_plane, mask = simulator.generate_rfi()
        save_example_pair_npy(tf_plane, mask, index=i, out_dir=val_dir, generate_mask=args.generate_mask)

    logging.info("Dataset generation complete.")

if __name__ == "__main__":
    main()
