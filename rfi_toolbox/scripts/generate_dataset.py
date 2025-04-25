import os
import numpy as np
import argparse
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import RobustScaler

class RFIMaskDataset(Dataset):
    def __init__(self, data_dir, transform=None, normalization='global_min_max'):
        self.data_dir = data_dir
        self.sample_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        self.transform = transform
        self.normalization = normalization
        self.global_min = np.inf
        self.global_max = -np.inf
        self.mean = None
        self.std = None
        self.robust_scaler = None

        # Calculate global min and max, mean and std for standardization, and fit RobustScaler
        all_data = []
        for sample_dir in self.sample_dirs:
            input_path = os.path.join(sample_dir, 'input.npy')
            input_np = np.load(input_path)
            all_data.append(input_np)
            self.global_min = min(self.global_min, np.min(input_np))
            self.global_max = max(self.global_max, np.max(input_np))

        all_data_np = np.concatenate([d.flatten() for d in all_data])
        self.mean = np.mean(all_data_np)
        self.std = np.std(all_data_np) + 1e-8 # Add epsilon for stability

        # Fit RobustScaler
        all_data_reshaped = all_data_np.reshape(-1, 1)
        self.robust_scaler = RobustScaler().fit(all_data_reshaped)

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        input_path = os.path.join(sample_dir, 'input.npy')
        mask_path = os.path.join(sample_dir, 'rfi_mask.npy')

        input_np = np.load(input_path)
        mask = np.load(mask_path)

        # Normalize the input data based on the chosen scheme
        if self.normalization == 'global_min_max':
            if self.global_max > self.global_min:
                input_normalized = (input_np - self.global_min) / (self.global_max - self.global_min)
            else:
                input_normalized = np.zeros_like(input_np)
        elif self.normalization == 'standardize':
            input_normalized = (input_np - self.mean) / self.std
        elif self.normalization == 'robust_scale':
            original_shape = input_np.shape
            input_reshaped = input_np.reshape(-1, 1)
            input_scaled = self.robust_scaler.transform(input_reshaped)
            input_normalized = input_scaled.reshape(original_shape)
        else:
            input_normalized = input_np # No normalization

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
