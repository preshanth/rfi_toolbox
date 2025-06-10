# rfi_toolbox/core/simulator.py
import numpy as np


class RFISimulator:
    def __init__(self, time_bins=1024, freq_bins=1024):
        self.time_bins = time_bins
        self.freq_bins = freq_bins
        self.power_range = np.logspace(-6, 4, num=100)
        self.tf_plane = {
            "RR": np.empty((time_bins, freq_bins), dtype=complex),
            "RL": np.empty((time_bins, freq_bins), dtype=complex),
            "LR": np.empty((time_bins, freq_bins), dtype=complex),
            "LL": np.empty((time_bins, freq_bins), dtype=complex),
        }
        self.mask = np.zeros((self.time_bins, self.freq_bins), dtype=bool)

    def generate_clean_data(self):
        """
        Generate RFI-free Gaussian-like data
        """
        self.tf_plane = {
            "RR": np.random.normal(size=(self.time_bins, self.freq_bins))
            + 1j * np.random.normal(size=(self.time_bins, self.freq_bins)),
            "RL": np.random.normal(size=(self.time_bins, self.freq_bins))
            + 1j * np.random.normal(size=(self.time_bins, self.freq_bins)),
            "LR": np.random.normal(size=(self.time_bins, self.freq_bins))
            + 1j * np.random.normal(size=(self.time_bins, self.freq_bins)),
            "LL": np.random.normal(size=(self.time_bins, self.freq_bins))
            + 1j * np.random.normal(size=(self.time_bins, self.freq_bins)),
        }

        self.mask = np.zeros((self.time_bins, self.freq_bins), dtype=bool)

        return self.tf_plane, self.mask

    def generate_rfi(self):

        self.generate_clean_data()

        # Enhanced Broadband RFI (5%) in 2-3 separated frequency chunks
        for _ in range(np.random.randint(2, 4)):
            max_width = self.freq_bins - 1
            freq_start = np.random.randint(0, max(1, max_width - 100))
            freq_width = np.random.randint(50, min(150, max_width - freq_start))
            time_range = slice(None)
            rfi_block = np.s_[time_range, freq_start : freq_start + freq_width]
            modulation = np.random.uniform(0.5, 2.0, size=(self.time_bins, freq_width))
            power = np.random.choice(
                self.power_range, size=(self.time_bins, freq_width)
            )
            for pol in ["RR", "LL"]:
                self.tf_plane[pol][rfi_block] += (
                    modulation
                    * power
                    * (
                        np.random.randn(self.time_bins, freq_width)
                        + 1j * np.random.randn(self.time_bins, freq_width)
                    )
                )
            self.mask[rfi_block] = True

        # Narrowband RFI (5%) with amplitude variation (frequency-based)
        for _ in range(int(self.freq_bins * 0.05)):
            freq_idx = np.random.randint(0, self.freq_bins)
            rfi_val = np.random.choice(self.power_range)
            modulation = np.random.uniform(0.5, 2.0, size=self.time_bins)
            self.tf_plane["RR"][:, freq_idx] += (
                modulation
                * rfi_val
                * (
                    np.random.randn(self.time_bins)
                    + 1j * np.random.randn(self.time_bins)
                )
            )
            self.tf_plane["LL"][:, freq_idx] += (
                modulation
                * rfi_val
                * (
                    np.random.randn(self.time_bins)
                    + 1j * np.random.randn(self.time_bins)
                )
            )
            self.mask[:, freq_idx] = True

        # Time-bursty RFI (10%)
        for _ in range(int(self.time_bins * 0.1)):
            time_idx = np.random.randint(0, self.time_bins)
            rfi_val = np.random.choice(self.power_range)
            modulation = np.random.uniform(0.5, 2.0, size=self.freq_bins)
            self.tf_plane["RR"][time_idx, :] += (
                modulation
                * rfi_val
                * (
                    np.random.randn(self.freq_bins)
                    + 1j * np.random.randn(self.freq_bins)
                )
            )
            self.tf_plane["LL"][time_idx, :] += (
                modulation
                * rfi_val
                * (
                    np.random.randn(self.freq_bins)
                    + 1j * np.random.randn(self.freq_bins)
                )
            )
            self.mask[time_idx, :] = True

        # Sweeping/bursty RFI (10%)
        for _ in range(5):
            start_t, start_f = np.random.randint(
                0, self.time_bins // 2
            ), np.random.randint(0, self.freq_bins // 2)
            slope = np.random.uniform(-2, 2)
            for i in range(self.time_bins // 2):
                f_idx = int(start_f + slope * i) % self.freq_bins
                t_idx = (start_t + i) % self.time_bins
                power_factor = np.random.choice(self.power_range)
                self.tf_plane["RR"][t_idx, f_idx] += power_factor + 1j * power_factor
                self.tf_plane["LL"][t_idx, f_idx] += power_factor + 1j * power_factor
                self.mask[t_idx, f_idx] = True

        # RFI that sweeps with time^2
        for _ in range(5):
            start_t = np.random.randint(0, self.time_bins // 4)
            start_f = np.random.randint(0, self.freq_bins // 4)
            direction = np.random.choice([-1, 1])
            for t in range(self.time_bins // 4):
                f_idx = int(start_f + direction * (t**2) // 100) % self.freq_bins
                t_idx = (start_t + t) % self.time_bins
                power_factor = np.random.choice(self.power_range)
                self.tf_plane["RR"][t_idx, f_idx] += power_factor + 1j * power_factor
                self.mask[t_idx, f_idx] = True

        # Polarized RFI for RL and LR
        for pol in ["RL", "LR"]:
            polarization_factor = np.random.uniform(
                0, 1, size=(self.time_bins, self.freq_bins)
            )
            self.tf_plane[pol] += polarization_factor * self.tf_plane["RR"]

        return self.tf_plane, self.mask
