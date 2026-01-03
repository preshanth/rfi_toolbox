"""
MS Loader - Load CASA measurement sets for RFI analysis

Clean rewrite of RadioRFI functionality, focused on data loading only.
"""

import numpy as np
from tqdm import tqdm
import gc

print("[DEBUG ms_loader] Attempting casatools import")
try:
    from casatools import table

    print("[DEBUG ms_loader] casatools.table imported successfully")
except Exception as e:
    print(f"[DEBUG ms_loader] casatools import failed: {e}")
    raise ImportError(
        "MSLoader requires CASA to be properly installed and configured.\n"
        "Install with: pip install rfi-toolbox[casa]\n"
        "See: https://casadocs.readthedocs.io/\n"
        f"Original error: {e}"
    ) from e

print("[DEBUG ms_loader] About to define MSLoader class")


class MSLoader:
    """
    Load complex visibilities from CASA measurement sets.

    Simplified interface:
    >>> loader = MSLoader('observation.ms', field_id=0)
    >>> loader.load(num_antennas=5, mode='DATA')
    >>> data = loader.data  # Shape: (baselines, pols, channels, times)
    >>> flags = loader.load_flags()  # Load existing flags

    Field handling:
    >>> fields = loader.get_available_fields()  # Get list of all field IDs
    >>> loader.load(field_id=1)  # Load specific field
    """

    def __init__(self, ms_path, field_id=None):
        """
        Initialize MS loader.

        Args:
            ms_path: Path to measurement set
            field_id: Optional FIELD_ID to load. If None, loads all fields.
        """
        self.ms_path = str(ms_path)
        self.field_id = field_id

        # Open MS and read metadata
        tb = table()

        # Number of antennas
        tb.open(self.ms_path + "/ANTENNA")
        self.num_antennas = tb.nrows()
        tb.close()

        # Number of spectral windows and channels
        tb.open(self.ms_path + "/SPECTRAL_WINDOW")
        self.num_spw = tb.nrows()
        self.channels_per_spw = tb.getcol("NUM_CHAN")
        tb.close()

        # Main table
        self.tb = table()
        self.tb.open(self.ms_path, nomodify=False)

        # Get number of time samples
        field_filter = (
            f" && FIELD_ID=={self.field_id}" if self.field_id is not None else ""
        )
        subtable = self.tb.query(
            f"DATA_DESC_ID==0 && ANTENNA1==0 && ANTENNA2==1{field_filter}"
        )
        self.num_times = len(subtable.getcol("TIME"))
        subtable.close()

        # Storage for loaded data
        self.data = None
        self.flags = None
        self.antenna_baseline_map = None
        self.spw_list = None

    def get_metadata(self, num_antennas=None, mode="DATA"):
        """
        Get MS metadata without loading data (fast).

        Args:
            num_antennas: Number of antennas (default: all)
            mode: Column to get metadata for

        Returns:
            dict with keys: num_baselines, num_pols, num_channels, num_times, baseline_map
        """
        if num_antennas is None:
            num_antennas = self.num_antennas

        # Get shape from dminfo (no data loading)
        dminfo = self.tb.getdminfo()

        # Find the storage manager for the DATA column
        data_sm = None
        for key, info in dminfo.items():
            if mode in info.get("COLUMNS", []):
                data_sm = info
                break

        if data_sm is None:
            raise ValueError(f"Column {mode} not found in MS")

        # Extract shape from first hypercube
        hypercubes = data_sm["SPEC"]["HYPERCUBES"]
        if hypercubes:
            first_cube = list(hypercubes.values())[0]
            cell_shape = first_cube["CellShape"]
            num_pols, num_channels = cell_shape[0], cell_shape[1]
        else:
            raise ValueError(f"No hypercube info for {mode}")

        # Build baseline map
        baseline_map = []
        for i in range(num_antennas):
            for j in range(i + 1, num_antennas):
                baseline_map.append((i, j))

        num_baselines = len(baseline_map)
        num_times = self.num_times

        return {
            "num_baselines": num_baselines,
            "num_pols": num_pols,
            "num_channels": num_channels,
            "num_times": num_times,
            "baseline_map": baseline_map,
            "shape": (num_baselines, num_pols, num_channels, num_times),
        }

    def load(self, num_antennas=None, mode="DATA", field_id=None):
        """
        Load complex visibilities from MS.

        Args:
            num_antennas: Number of antennas to load (default: all)
            mode: Column to load ('DATA', 'CORRECTED_DATA', etc.)
            field_id: Optional FIELD_ID to load. If provided, overrides field_id from __init__.

        Returns:
            Loaded data shape: (num_baselines, num_pols, num_channels, num_times)
        """
        if num_antennas is None:
            num_antennas = self.num_antennas

        # Allow field_id parameter to override instance field_id
        if field_id is not None:
            self.field_id = field_id

        # Filter to SPWs with same number of channels
        same_spw_list = []
        same_channels_list = []

        for spw, num_chan in enumerate(self.channels_per_spw):
            if num_chan == self.channels_per_spw[0]:
                same_spw_list.append(spw)
                same_channels_list.append(num_chan)

        num_channels = same_channels_list[0]
        num_spw = len(same_spw_list)
        total_channels = num_spw * num_channels

        # Load baselines
        data_list = []
        baseline_map = []

        print(f"\nLoading {mode} from {self.ms_path}...")
        print(f"  Antennas: {num_antennas}/{self.num_antennas}")
        print(
            f"  SPWs: {num_spw} ({num_channels} channels each = {total_channels} total)"
        )
        print(f"  Times: {self.num_times}")
        if self.field_id is not None:
            print(f"  Field ID: {self.field_id}")

        # Build field filter string for queries
        field_filter = (
            f" && FIELD_ID=={self.field_id}" if self.field_id is not None else ""
        )

        for i in tqdm(range(num_antennas), desc="Antenna 1"):
            for j in range(i + 1, self.num_antennas):
                # Allocate array for this baseline
                baseline_data = np.zeros(
                    [4, total_channels, self.num_times], dtype="complex128"
                )

                # Check if this baseline has any data
                has_data = False

                # Load all SPWs for this baseline
                for spw_idx, spw in enumerate(same_spw_list):
                    subtable = self.tb.query(
                        f"DATA_DESC_ID=={spw} && ANTENNA1=={i} && ANTENNA2=={j}{field_filter}"
                    )

                    # Skip if no data for this baseline/SPW
                    if subtable.nrows() == 0:
                        subtable.close()
                        continue

                    has_data = True

                    # Extract data for this SPW
                    spw_data = subtable.getcol(mode)

                    # Place in combined array
                    start_ch = spw_idx * num_channels
                    end_ch = (spw_idx + 1) * num_channels
                    baseline_data[:, start_ch:end_ch, :] = spw_data

                    subtable.close()

                # Only add baseline if it has data
                if has_data:
                    data_list.append(baseline_data)
                    baseline_map.append((i, j))

        # Stack all baselines
        self.data = np.stack(data_list)  # Shape: (baselines, pols, channels, times)
        self.antenna_baseline_map = baseline_map
        self.spw_list = same_spw_list
        self.channels_per_spw_list = same_channels_list

        print(f"  Loaded shape: {self.data.shape}")

        return self.data

    def load_single_baseline(
        self, ant1=0, ant2=1, pol_idx=0, mode="DATA", field_id=None
    ):
        """
        Load single baseline, single polarization.

        Args:
            ant1: First antenna
            ant2: Second antenna
            pol_idx: Polarization index (0=XX, 1=XY, 2=YX, 3=YY)
            mode: Column to load ('DATA', 'CORRECTED_DATA', etc.)
            field_id: Optional FIELD_ID to load. If provided, overrides field_id from __init__.

        Returns:
            Complex array shape: (total_channels, num_times)
        """
        # Allow field_id parameter to override instance field_id
        if field_id is not None:
            self.field_id = field_id
        # Filter to SPWs with same number of channels
        same_spw_list = []
        same_channels_list = []

        for spw, num_chan in enumerate(self.channels_per_spw):
            if num_chan == self.channels_per_spw[0]:
                same_spw_list.append(spw)
                same_channels_list.append(num_chan)

        num_channels = same_channels_list[0]
        num_spw = len(same_spw_list)
        total_channels = num_spw * num_channels

        print(f"\nLoading single baseline from {self.ms_path}...")
        print(f"  Baseline: {ant1}-{ant2}, Pol: {pol_idx}")
        print(
            f"  SPWs: {num_spw} ({num_channels} channels each = {total_channels} total)"
        )
        print(f"  Times: {self.num_times}")
        if self.field_id is not None:
            print(f"  Field ID: {self.field_id}")

        # Build field filter string for queries
        field_filter = (
            f" && FIELD_ID=={self.field_id}" if self.field_id is not None else ""
        )

        # Allocate array for this baseline
        baseline_data = np.zeros([total_channels, self.num_times], dtype="complex128")

        # Load all SPWs for this baseline
        for spw_idx, spw in enumerate(same_spw_list):
            subtable = self.tb.query(
                f"DATA_DESC_ID=={spw} && ANTENNA1=={ant1} && ANTENNA2=={ant2}{field_filter}"
            )

            if subtable.nrows() == 0:
                subtable.close()
                raise ValueError(f"No data for baseline {ant1}-{ant2} in SPW {spw}")

            # Extract data for this SPW, single pol
            spw_data = subtable.getcol(mode)  # Shape: (pols, channels, times)
            spw_data_pol = spw_data[pol_idx, :, :]  # Shape: (channels, times)

            # Place in combined array
            start_ch = spw_idx * num_channels
            end_ch = (spw_idx + 1) * num_channels
            baseline_data[start_ch:end_ch, :] = spw_data_pol

            subtable.close()

        print(f"  Loaded shape: {baseline_data.shape}")

        return baseline_data

    def load_baseline(self, ant1, ant2, mode="DATA", field_id=None):
        """
        Load one baseline, all pols. Opens/closes table per call.

        Args:
            ant1, ant2: Antenna pair
            mode: Column ('DATA', 'CORRECTED_DATA', etc.)
            field_id: Optional FIELD_ID

        Returns:
            Complex array (pols, channels, times)
        """
        tb = table()
        tb.open(self.ms_path, nomodify=False)

        # Get SPW info
        tb_spw = table()
        tb_spw.open(self.ms_path + "/SPECTRAL_WINDOW")
        channels_per_spw = tb_spw.getcol("NUM_CHAN")
        tb_spw.close()

        # Use SPWs with same channel count
        same_spw_list = []
        for spw, num_chan in enumerate(channels_per_spw):
            if num_chan == channels_per_spw[0]:
                same_spw_list.append(spw)

        num_channels = channels_per_spw[0]
        total_channels = len(same_spw_list) * num_channels

        # Get num times (query first SPW to get shape)
        field_filter = f" && FIELD_ID=={field_id}" if field_id is not None else ""
        test_sub = tb.query(
            f"DATA_DESC_ID=={same_spw_list[0]} && ANTENNA1=={ant1} && ANTENNA2=={ant2}{field_filter}"
        )
        num_times = test_sub.nrows()
        test_sub.close()

        # Allocate
        baseline_data = np.zeros([4, total_channels, num_times], dtype="complex128")

        # Load each SPW
        for spw_idx, spw in enumerate(same_spw_list):
            subtable = tb.query(
                f"DATA_DESC_ID=={spw} && ANTENNA1=={ant1} && ANTENNA2=={ant2}{field_filter}"
            )

            if subtable.nrows() == 0:
                subtable.close()
                continue

            spw_data = subtable.getcol(mode)  # (pols, channels, times)

            start_ch = spw_idx * num_channels
            end_ch = (spw_idx + 1) * num_channels
            baseline_data[:, start_ch:end_ch, :] = spw_data

            subtable.close()

        tb.close()
        return baseline_data

    def save_baseline_flags(self, ant1, ant2, flags, field_id=None):
        """
        Write flags for one baseline. Opens/closes table per call.

        Args:
            ant1, ant2: Antenna pair
            flags: Boolean array (pols, channels, times)
            field_id: Optional FIELD_ID
        """
        tb = table()
        tb.open(self.ms_path, nomodify=False)

        # Get SPW info
        tb_spw = table()
        tb_spw.open(self.ms_path + "/SPECTRAL_WINDOW")
        channels_per_spw = tb_spw.getcol("NUM_CHAN")
        tb_spw.close()

        # Use SPWs with same channel count
        same_spw_list = []
        for spw, num_chan in enumerate(channels_per_spw):
            if num_chan == channels_per_spw[0]:
                same_spw_list.append(spw)

        num_channels = channels_per_spw[0]

        field_filter = f" && FIELD_ID=={field_id}" if field_id is not None else ""

        # Write each SPW
        for spw_idx, spw in enumerate(same_spw_list):
            start_ch = spw_idx * num_channels
            end_ch = (spw_idx + 1) * num_channels
            spw_flags = flags[:, start_ch:end_ch, :]

            subtable = tb.query(
                f"DATA_DESC_ID=={spw} && ANTENNA1=={ant1} && ANTENNA2=={ant2}{field_filter}"
            )

            if subtable.nrows() > 0:
                subtable.putcol("FLAG", spw_flags)

            subtable.close()

        tb.close()

    def get_baseline_pairs(self, num_antennas=None):
        """
        Get list of baseline pairs.

        Returns:
            List of (ant1, ant2) tuples
        """
        if num_antennas is None:
            num_antennas = self.num_antennas

        pairs = []
        for i in range(num_antennas):
            for j in range(i + 1, num_antennas):
                pairs.append((i, j))
        return pairs

    def load_flags(self):
        """
        Load existing flags from MS.

        Returns:
            Flags shape: (num_baselines, num_pols, num_channels, num_times)
        """
        if self.antenna_baseline_map is None:
            raise ValueError("Must call load() first to establish baseline map")

        print("\nLoading flags from MS...")
        if self.field_id is not None:
            print(f"  Field ID: {self.field_id}")

        # Build field filter string for queries
        field_filter = (
            f" && FIELD_ID=={self.field_id}" if self.field_id is not None else ""
        )

        flags_list = []
        num_channels = self.channels_per_spw_list[0]
        num_spw = len(self.spw_list)
        total_channels = num_spw * num_channels

        for ant1, ant2 in tqdm(self.antenna_baseline_map, desc="Baselines"):
            baseline_flags = np.zeros([4, total_channels, self.num_times], dtype=bool)

            for spw_idx, spw in enumerate(self.spw_list):
                subtable = self.tb.query(
                    f"DATA_DESC_ID=={spw} && ANTENNA1=={ant1} && ANTENNA2=={ant2}{field_filter}"
                )

                spw_flags = subtable.getcol("FLAG")

                start_ch = spw_idx * num_channels
                end_ch = (spw_idx + 1) * num_channels
                baseline_flags[:, start_ch:end_ch, :] = spw_flags

                subtable.close()

            flags_list.append(baseline_flags)

        self.flags = np.stack(flags_list)
        print(f"  Loaded flags shape: {self.flags.shape}")

        return self.flags

    def save_flags(self, flags):
        """
        Write flags back to MS.

        Args:
            flags: Flag array shape (num_baselines, num_pols, num_channels, num_times)
        """
        if self.antenna_baseline_map is None:
            raise ValueError("Must call load() first to establish baseline map")

        print("\nSaving flags to MS...")
        if self.field_id is not None:
            print(f"  Field ID: {self.field_id}")

        # Build field filter string for queries
        field_filter = (
            f" && FIELD_ID=={self.field_id}" if self.field_id is not None else ""
        )

        num_channels = self.channels_per_spw_list[0]

        for baseline_idx, (ant1, ant2) in enumerate(
            tqdm(self.antenna_baseline_map, desc="Baselines")
        ):
            baseline_flags = flags[baseline_idx]

            for spw_idx, spw in enumerate(self.spw_list):
                # Extract flags for this SPW
                start_ch = spw_idx * num_channels
                end_ch = (spw_idx + 1) * num_channels
                spw_flags = baseline_flags[:, start_ch:end_ch, :]

                # Write to MS
                subtable = self.tb.query(
                    f"DATA_DESC_ID=={spw} && ANTENNA1=={ant1} && ANTENNA2=={ant2}{field_filter}"
                )
                subtable.putcol("FLAG", spw_flags)
                subtable.close()

        print("  Flags saved successfully")

    def get_available_fields(self):
        """
        Get list of unique FIELD_IDs present in this measurement set.

        Returns:
            list: Sorted list of field IDs
        """
        field_ids = np.unique(self.tb.getcol("FIELD_ID"))
        return sorted(field_ids.tolist())

    def close(self):
        """Close the measurement set."""
        if hasattr(self, "tb"):
            self.tb.close()
        if hasattr(self, "data"):
            del self.data
        if hasattr(self, "flags"):
            del self.flags
        gc.collect()

    def __del__(self):
        """Ensure MS is closed on deletion."""
        self.close()

    @property
    def magnitude(self):
        """Get magnitude of complex visibilities."""
        if self.data is None:
            raise ValueError("Must call load() first")
        return np.abs(self.data)


print("[DEBUG ms_loader] MSLoader class definition complete")
