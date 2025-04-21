import argparse
import os
import logging
from tqdm import trange
import numpy as np
from rfi_toolbox.core.simulator import RFISimulator
from torch.utils.data import Dataset
import glob
import torch

# rfi_toolbox/scripts/generate_dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import os

class RFIMaskDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.sample_dirs = sorted([d for d in glob.glob(os.path.join(root_dir, '*')) if os.path.isdir(d)])
        self.transform = transform
        self._find_data_stats() # Calculate min and max values for normalization

    def _find_data_stats(self):
        all_min = []
        all_max = []
        for sample_dir in self.sample_dirs:
            input_path = os.path.join(sample_dir, 'input.npy')
            input_data = np.load(input_path)
            all_min.append(np.min(input_data))
            all_max.append(np.max(input_data))
        self.global_min = np.min(all_min)
        self.global_max = np.max(all_max)
        print(f"Global Min Input Value: {self.global_min}")
        print(f"Global Max Input Value: {self.global_max}")

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        input_path = os.path.join(sample_dir, 'input.npy')
        mask_path = os.path.join(sample_dir, 'rfi_mask.npy')

        input_np = np.load(input_path)
        mask = np.load(mask_path)

        # Normalize the input data
        if self.global_max > self.global_min:
            input_normalized = (input_np - self.global_min) / (self.global_max - self.global_min)
        else:
            input_normalized = np.zeros_like(input_np) # Handle case where min equals max

        input_tensor = torch.tensor(input_normalized, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            input_tensor, mask_tensor = self.transform(input_tensor, mask_tensor)

        return input_tensor, mask_tensor

def save_example_pair_npy(tf_plane, mask, index, out_dir, generate_mask=True):
    sample_dir = os.path.join(out_dir, f"{index:04d}")
    os.makedirs(sample_dir, exist_ok=True)
    input_data = np.stack([
        tf_plane['RR'].real, tf_plane['RR'].imag,
        tf_plane['RL'].real, tf_plane['RL'].imag,
        tf_plane['LR'].real, tf_plane['LR'].imag,
        tf_plane['LL'].real, tf_plane['LL'].imag
    ], axis=0)  # shape: (8, time_bins, freq_bins)

    input_path = os.path.join(sample_dir, f"input.npy") # Changed filename
    np.save(input_path, input_data)

    if generate_mask:
        mask_path = os.path.join(sample_dir, f"rfi_mask.npy")
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

    simulator = RFISimulator(time_bins=args.time_bins, freq_bins=args.frequency_bins)

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
