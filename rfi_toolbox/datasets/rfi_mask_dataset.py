"""
RFI Mask Dataset for PyTorch training.

Provides dataset class for loading RFI training data from either:
- Pre-generated synthetic data directories
- CASA Measurement Sets (MS)
"""

import logging
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

try:
    import casacore.tables as ct

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
        self.robust_median = None
        self.robust_iqr = None
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

        if self.normalization == "robust_scale":
            # Robust scaling: median centering + IQR scaling (pure numpy)
            self.robust_median = np.median(all_data_np)
            q25 = np.percentile(all_data_np, 25)
            q75 = np.percentile(all_data_np, 75)
            self.robust_iqr = q75 - q25 + 1e-8  # avoid division by zero

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
            return (input_np - self.robust_median) / self.robust_iqr
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
