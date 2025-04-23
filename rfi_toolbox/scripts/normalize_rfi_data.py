# normalize_rfi_data.py
import argparse
import os
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, RobustScaler
import shutil

def normalize_array(data, method='standardize'):
    original_shape = data.shape
    data_flat = data.reshape(-1, 1)
    if method == 'standardize':
        scaler = StandardScaler()
    elif method == 'robust_scale':
        scaler = RobustScaler()
    elif method == 'global_min_max':
        global_min = np.min(data)
        global_max = np.max(data)
        if global_max > global_min:
            return (data - global_min) / (global_max - global_min)
        else:
            return np.zeros_like(data)
    elif method is None:
        return data
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

    if method in ['standardize', 'robust_scale']:
        normalized_data = scaler.fit_transform(data_flat)
        return normalized_data.reshape(original_shape)

def process_directory(input_dir, output_dir, normalization_method):
    os.makedirs(output_dir, exist_ok=True)
    total_files = 0
    processed_files = 0
    mask_files = 0

    for root, _, files in os.walk(input_dir):
        for filename in files:
            input_path = os.path.join(root, filename)
            relative_path = os.path.relpath(root, input_dir)
            output_subdir = os.path.join(output_dir, relative_path)
            os.makedirs(output_subdir, exist_ok=True)
            output_path = os.path.join(output_subdir, filename)

            if filename == 'input.npy':  # Changed the filename matching condition
                total_files += 1
                try:
                    input_data = np.load(input_path)
                    normalized_data = normalize_array(input_data, method=normalization_method)
                    np.save(output_path, normalized_data)
                    processed_files += 1
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")
            elif filename == 'rfi_mask.npy':
                mask_files += 1
                # Copy the mask files to the output path without normalization using shutil
                shutil.copy(input_path, output_path)

    print(f"Processed {processed_files}/{total_files} input files in '{input_dir}' with normalization: {normalization_method}")
    print(f"Copied {mask_files} mask files to '{output_dir}'.")

def main():
    parser = argparse.ArgumentParser(description="Normalize RFI dataset numpy files.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input data directory (e.g., rfi_dataset/train)")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory for normalized data")
    parser.add_argument("--normalization", type=str, default='standardize',
                        choices=['global_min_max', 'standardize', 'robust_scale', None],
                        help="Normalization scheme to apply")
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir, args.normalization)
    print("Normalization complete.")

if __name__ == "__main__":
    main()