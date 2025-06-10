import argparse
import logging
import os

import numpy as np
import torch
from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset
from tqdm import tqdm

from rfi_toolbox.core.simulator import RFISimulator

try:
    import casacore.tables as ct  # Import here

    use_casacore = True
except ImportError:
    try:
        from casatools import table

        use_casacore = False
    except ImportError:
        raise ImportError("Please install casacore or casatools to use this script.")


class RFIMaskDataset(Dataset):
    def __init__(
        self,
        data_dir,
        transform=None,
        normalization="global_min_max",
        use_ms=False,
        ms_name=None,
        field_selection=None,
    ):
        """
        Args:
            data_dir (str): Directory containing the dataset (either generated or from MS).
            transform (callable, optional): Optional transform to be applied on a sample.
            normalization (str): Normalization method ('global_min_max', 'standardize', 'robust_scale', or None).
            use_ms (bool): If True, load data from a Measurement Set.
            ms_name (str, optional): Path to the Measurement Set if use_ms is True.
            field_selection (int or list, optional):  Select specific FIELD_ID(s) from the MS.
                If None, use all fields. If int, use that field. If list, use fields in the list.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.normalization = normalization
        self.use_ms = use_ms
        self.ms_name = ms_name
        self.global_min = np.inf
        self.global_max = -np.inf
        self.mean = None
        self.std = None
        self.robust_scaler = None
        self.sample_dirs = []
        self.field_selection = field_selection

        if use_ms:
            if not ms_name:
                raise ValueError("ms_name must be provided when use_ms is True")
            if use_casacore:
                self.tb = ct.table(ms_name, readonly=True)
            else:
                tb = table()
                self.tb = tb.open(ms_name, readonly=True)
            self.num_antennas = self.tb.getcol("ANTENNA1").max() + 1
            self.spw_array = np.unique(self.tb.getcol("DATA_DESC_ID"))

            channels_per_spw_list = []
            for spw in self.spw_array:
                subtable = self.tb.query(f"DATA_DESC_ID=={spw}")
                num_channels = subtable.getcol("DATA").shape[1]
                channels_per_spw_list.append(num_channels)
            self.channels_per_spw = channels_per_spw_list
            self.num_spw = len(self.spw_array)

            subtable = self.tb.query(
                f"DATA_DESC_ID=={0} && ANTENNA1=={0} && ANTENNA2=={1}"
            )
            self.time_tb = len(subtable.getcol("TIME"))
            self.sample_dirs = self._generate_ms_samples()
            self.tb.close()

        else:
            self.sample_dirs = [
                os.path.join(data_dir, d)
                for d in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, d))
            ]

        # Calculate normalization parameters
        self._calculate_normalization_params()

    def _calculate_normalization_params(self):
        """
        Calculates global min/max, mean/std, and fits RobustScaler.
        """
        all_data = []
        for sample_dir in self.sample_dirs:
            input_path = os.path.join(sample_dir, "input.npy")
            input_np = np.load(input_path)
            all_data.append(input_np)
            self.global_min = min(self.global_min, np.min(input_np))
            self.global_max = max(self.global_max, np.max(input_np))

        all_data_np = np.concatenate([d.flatten() for d in all_data])
        self.mean = np.mean(all_data_np)
        self.std = np.std(all_data_np) + 1e-8
        all_data_reshaped = all_data_np.reshape(-1, 1)
        self.robust_scaler = RobustScaler().fit(all_data_reshaped)

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        input_path = os.path.join(sample_dir, "input.npy")
        mask_path = os.path.join(sample_dir, "rfi_mask.npy")

        input_np = np.load(input_path)
        mask = np.load(mask_path)

        # Normalize
        input_normalized = self._normalize_input(input_np)

        input_tensor = torch.tensor(input_normalized, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            input_tensor, mask_tensor = self.transform(input_tensor, mask_tensor)
        return input_tensor, mask_tensor

    def _normalize_input(self, input_np):
        if self.normalization == "global_min_max":
            if self.global_max > self.global_min:
                return (input_np - self.global_min) / (
                    self.global_max - self.global_min
                )
            else:
                return np.zeros_like(input_np)
        elif self.normalization == "standardize":
            return (input_np - self.mean) / self.std
        elif self.normalization == "robust_scale":
            original_shape = input_np.shape
            input_reshaped = input_np.reshape(-1, 1)
            input_scaled = self.robust_scaler.transform(input_reshaped)
            return input_scaled.reshape(original_shape)
        else:
            return input_np

    def _generate_ms_samples(self):
        """
        Generates sample directories from the MS data, similar to the original dataset structure.
        Returns:
            list: A list of sample directories.
        """
        sample_dirs = []
        antenna_baseline_map = []

        same_spw_array = []
        same_channels_per_spw_array = []
        for spw, spw_numchan in zip(self.spw_array, self.channels_per_spw):
            if spw_numchan == self.channels_per_spw[0]:
                same_spw_array.append(spw)
                same_channels_per_spw_array.append(spw_numchan)
        init_chan = same_channels_per_spw_array[0]
        same_num_spw = len(same_spw_array)

        # Apply field selection if provided
        if self.field_selection is not None:
            if isinstance(self.field_selection, int):
                field_ids = [self.field_selection]
                self.tb = self.tb.query(
                    f"FIELD_ID == {self.field_selection}"
                )  # Corrected line
            else:
                field_ids = self.field_selection
                self.tb = self.tb.query(
                    f"FIELD_ID IN {tuple(field_ids)}"
                )  # Corrected line

        num_rows = self.tb.nrows()
        logging.info(f"Processing {num_rows} rows from MS...")
        for row_num in tqdm(range(num_rows), desc="Processing MS Rows"):
            row = self.tb[row_num]
            antenna1 = row["ANTENNA1"]
            antenna2 = row["ANTENNA2"]
            field_id = row["FIELD_ID"]

            # Skip if field is not selected
            if self.field_selection is not None:
                if isinstance(self.field_selection, int):
                    if field_id != self.field_selection:
                        continue
                elif field_id not in self.field_selection:
                    continue

            if antenna1 < antenna2:
                i = antenna1
                j = antenna2
            else:
                i = antenna2
                j = antenna1

            combined_data = np.zeros(
                [4, same_num_spw * init_chan, self.time_tb], dtype="complex128"
            )
            for spw_spec, spw, num_chan in zip(
                same_spw_array, range(same_num_spw), same_channels_per_spw_array
            ):
                subtable = self.tb.query(
                    f"DATA_DESC_ID=={spw_spec} && ANTENNA1=={i} && ANTENNA2=={j}"
                )
                if subtable.nrows() > 0:
                    spw_data = subtable.getcol("DATA")
                    # Transpose spw_data to (npol, nchan, ntimes)
                    spw_data = np.transpose(spw_data, (0, 1, 2))
                    combined_data[
                        :, spw * init_chan : (spw + 1) * init_chan, :
                    ] += spw_data

            # Create a sample directory for this baseline
            sample_dir = os.path.join(self.data_dir, f"ant{i}_ant{j}")
            os.makedirs(sample_dir, exist_ok=True)
            antenna_baseline_map.append((i, j))
            # Save input data as numpy array (real and imag parts)
            input_data = np.stack(
                [
                    combined_data[0].real,
                    combined_data[0].imag,
                    combined_data[1].real,
                    combined_data[1].imag,
                    combined_data[2].real,
                    combined_data[2].imag,
                    combined_data[3].real,
                    combined_data[3].imag,
                ],
                axis=0,
            )  # shape: (8, freq_bins, time_bins)

            input_path = os.path.join(sample_dir, "input.npy")
            np.save(input_path, input_data)

            # create dummy mask.
            mask = np.zeros((self.time_tb, same_num_spw * init_chan), dtype=np.float32)
            mask_path = os.path.join(sample_dir, "rfi_mask.npy")
            np.save(mask_path, mask)
            sample_dirs.append(sample_dir)
        self.antenna_baseline_map = antenna_baseline_map
        return sample_dirs


def save_example_pair_npy(tf_plane, mask, index, out_dir, generate_mask=True):
    """
    Saves input and mask numpy arrays.

    Args:
        tf_plane (dict): Dictionary containing the visibility data.
        mask (ndarray): The RFI mask.
        index (int): Index of the sample.
        out_dir (str): Output directory.
        generate_mask (bool): Whether to save the mask.
    """
    sample_dir = os.path.join(out_dir, f"{index:04d}")
    os.makedirs(sample_dir, exist_ok=True)
    input_data = np.stack(
        [
            tf_plane["RR"].real,
            tf_plane["RR"].imag,
            tf_plane["RL"].real,
            tf_plane["RL"].imag,
            tf_plane["LR"].real,
            tf_plane["LR"].imag,
            tf_plane["LL"].real,
            tf_plane["LL"].imag,
        ],
        axis=0,
    )  # shape: (8, time_bins, freq_bins)

    input_path = os.path.join(sample_dir, "input.npy")
    np.save(input_path, input_data)

    if generate_mask:
        mask_path = os.path.join(sample_dir, "rfi_mask.npy")
        np.save(mask_path, mask)


def main():
    """
    Main function to generate synthetic RFI dataset or read from an MS.
    """
    parser = argparse.ArgumentParser(
        description="Generate or load RFI dataset as numpy files."
    )
    parser.add_argument(
        "--samples_training", type=int, default=1000, help="Number of training samples."
    )
    parser.add_argument(
        "--samples_validation",
        type=int,
        default=200,
        help="Number of validation samples.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="rfi_dataset", help="Output directory."
    )
    parser.add_argument(
        "--only_clean",
        action="store_true",
        help="Generate only clean data without RFI (incompatible with --use_ms).",
    )
    parser.add_argument(
        "--time_bins",
        type=int,
        default=1024,
        help="Number of time bins in the TF plane.",
    )
    parser.add_argument(
        "--frequency_bins",
        type=int,
        default=1024,
        help="Number of frequency bins in the TF plane.",
    )
    parser.add_argument(
        "--generate_mask", action="store_true", default=True, help="Generate RFI masks."
    )
    parser.add_argument(
        "--no_generate_mask",
        action="store_false",
        dest="generate_mask",
        help="Disable mask generation.",
    )
    parser.add_argument(
        "--use_ms",
        action="store_true",
        default=False,
        help="Load data from a Measurement Set.",
    )
    parser.add_argument(
        "--ms_name", type=str, default=None, help="Path to the Measurement Set."
    )
    parser.add_argument(
        "--train_field", type=int, help="FIELD_ID to use for training set."
    )
    parser.add_argument(
        "--val_field", type=int, help="FIELD_ID to use for validation set."
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if args.use_ms:
        if not args.ms_name:
            logging.error("Error: --ms_name must be specified when --use_ms is used.")
            return

        if args.only_clean:
            logging.error(
                "Error: --only_clean is incompatible with --use_ms. --only_clean will be ignored."
            )
            return

        logging.info(f"Loading data from Measurement Set: {args.ms_name}")
        # Create a single directory to hold all MS samples
        ms_output_dir = os.path.join(args.output_dir, "ms_data")
        os.makedirs(ms_output_dir, exist_ok=True)

        # Create training and validation datasets, selecting fields if specified
        train_dataset = RFIMaskDataset(
            data_dir=ms_output_dir,
            use_ms=True,
            ms_name=args.ms_name,
            field_selection=args.train_field,
        )
        val_dataset = RFIMaskDataset(
            data_dir=ms_output_dir,
            use_ms=True,
            ms_name=args.ms_name,
            field_selection=args.val_field,
        )

        logging.info(f"Number of training samples from MS: {len(train_dataset)}")
        logging.info(f"Number of validation samples from MS: {len(val_dataset)}")

    else:
        simulator = RFISimulator(
            time_bins=args.time_bins, freq_bins=args.frequency_bins
        )

        if args.only_clean:
            logging.info("Generating only clean data without RFI.")
            train_dir = os.path.join(args.output_dir, "train")
            os.makedirs(train_dir, exist_ok=True)
            logging.info(
                f"Generating {args.samples_training} clean samples in '{train_dir}' (mask generation: {args.generate_mask})"
            )

            for i in tqdm(range(args.samples_training), desc="Clean"):
                tf_plane, mask = simulator.generate_clean_data()
                save_example_pair_npy(
                    tf_plane,
                    mask,
                    index=i,
                    out_dir=train_dir,
                    generate_mask=args.generate_mask,
                )

            tf_plane = simulator.tf_plane
            mask = np.zeros_like(tf_plane["RR"], dtype=bool)
        else:
            # Train samples
            train_dir = os.path.join(args.output_dir, "train")
            os.makedirs(train_dir, exist_ok=True)
            logging.info(
                f"Generating {args.samples_training} training samples in '{train_dir}' (mask generation: {args.generate_mask})"
            )
            for i in tqdm(range(args.samples_training), desc="Training"):
                tf_plane, mask = simulator.generate_rfi()
                save_example_pair_npy(
                    tf_plane,
                    mask,
                    index=i,
                    out_dir=train_dir,
                    generate_mask=args.generate_mask,
                )

            # Validation samples
            val_dir = os.path.join(args.output_dir, "val")
            os.makedirs(val_dir, exist_ok=True)
            logging.info(
                f"Generating {args.samples_validation} validation samples in '{val_dir}' (mask generation: {args.generate_mask})"
            )
            for i in tqdm(range(args.samples_validation), desc="Validation"):
                tf_plane, mask = simulator.generate_rfi()
                save_example_pair_npy(
                    tf_plane,
                    mask,
                    index=i,
                    out_dir=val_dir,
                    generate_mask=args.generate_mask,
                )
        logging.info("Dataset generation complete.")


if __name__ == "__main__":
    main()
