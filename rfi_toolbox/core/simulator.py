# rfi_toolbox/core/simulator.py
import numpy as np


class RFISimulator:
    """Time-frequency RFI simulator with physically-motivated coherent phase.

    Each RFI *event* (broadband chunk, narrowband channel, time burst, sweep)
    is given a coherent geometric phase rather than per-pixel random phase, so
    that the phase carries information a detector can exploit (the realimag
    input mode). The model is phenomenological but structurally faithful to a
    terrestrial / drifting near-field emitter seen through correlator
    fringe-stopping:

        phi(t, n) = 2*pi * [ (s0 + sdot*t) * n  +  r0 * t ] + phi0

      - ``s0``   frequency slope (turns/channel): a delay -> tilted fringes
                 across the band.
      - ``sdot`` drift of that slope over the scan: ~0 for a stationary
                 terrestrial source, non-zero for a drifting emitter (e.g. a
                 LEO satellite crossing the field) -> curved fringe tracks.
      - ``r0``   time fringe rate (turns/sample): residual fringe after
                 fringe-stopping. **Scales with baseline length** -- longer
                 baselines wind faster (confirmed on real VLA data: full-track
                 phase coherence falls with baseline length while the local
                 fringe stays coherent).
      - ``phi0`` per-event phase offset.

    Amplitude is the existing ``power * modulation`` (transmitter content lives
    in amplitude; only the complex *unit* changed from random to coherent).

    Ground truth uses **full injection truth down to a detectability floor**:
    every injected pixel whose amplitude exceeds ``detect_floor`` (in units of
    the noise amplitude) is labelled RFI, including the sub-noise interiors of
    extended events. Only injected pixels below the floor (indistinguishable
    from noise) are left clean.

    Gibbs/sinc channelizer ringing is **off by default**: real VLA data
    (polyphase filterbank) shows narrowband RFI confined to its channel with no
    broad sinc tail. Set ``gibbs_ringing=True`` for FX-correlator-style data.
    """

    def __init__(self, time_bins=1024, freq_bins=1024):
        self.time_bins = time_bins
        self.freq_bins = freq_bins
        self.power_range = np.logspace(-6, 4, num=100)
        # Full-truth label floor, in units of the noise amplitude. Clean data is
        # unit-variance complex Gaussian (|z| Rayleigh, median ~1.2). 1.0 labels
        # injected RFI down to roughly the noise floor; lower it to push the
        # labelled (and thus supervised) regime further sub-noise.
        self.detect_floor = 1.0
        # Probability an event is a drifting (curved-fringe) emitter.
        self.drift_prob = 0.3
        # Fringe-density ceilings (fringes across the relevant extent at the
        # longest baseline). Time fringes scale with baseline length.
        self.max_time_fringes = 30.0
        self.max_freq_fringes = 8.0
        # Gibbs/sinc ringing: off by default (VLA PFB suppresses it).
        self.gibbs_ringing = False
        self._gibbs_kernel = self._make_gibbs_kernel(n_side=8, stretch=2.0)
        self.baseline_frac = 0.5  # set per generate_rfi() call
        self.tf_plane = {
            pol: np.empty((time_bins, freq_bins), dtype=complex)
            for pol in ("RR", "RL", "LR", "LL")
        }
        self.mask = np.zeros((self.time_bins, self.freq_bins), dtype=bool)

    # ------------------------------------------------------------------ phase
    def _draw_event_phase(self, width_channels, n_times, drifting=False):
        """Draw (s0, sdot, r0, phi0) scaled to the event extent and the current
        baseline length (``self.baseline_frac`` in [0, 1])."""
        rng = np.random
        sign = lambda: rng.choice([-1.0, 1.0])  # noqa: E731
        w = max(int(width_channels), 1)
        nt = max(int(n_times), 1)
        bl = self.baseline_frac
        # Time fringe rate grows with baseline length (the validated effect).
        n_ft = rng.uniform(0.5, 1.0 + bl * self.max_time_fringes)
        r0 = (n_ft / nt) * sign()
        # Frequency slope (delay) -- also grows modestly with baseline.
        n_ff = rng.uniform(0.5, 1.0 + bl * self.max_freq_fringes)
        s0 = (n_ff / w) * sign()
        phi0 = rng.uniform(0, 2 * np.pi)
        if drifting:
            s_end = (rng.uniform(0.5, 1.0 + bl * self.max_freq_fringes) / w) * sign()
            sdot = (s_end - s0) / nt
        else:
            sdot = 0.0
        return s0, sdot, r0, phi0

    @staticmethod
    def _phase_grid(t_idx, n_idx, params):
        s0, sdot, r0, phi0 = params
        return 2 * np.pi * ((s0 + sdot * t_idx) * n_idx + r0 * t_idx) + phi0

    # ------------------------------------------------------------------ gibbs
    @staticmethod
    def _make_gibbs_kernel(n_side=8, stretch=2.0):
        x = np.arange(-n_side, n_side + 1) / float(stretch)
        k = np.sinc(x)
        return k / k.sum()

    def _spread_block(self, pols, fslice, core):
        """Add a (T, width) broadband event to ``pols``, optionally with sinc
        ringing along frequency."""
        if self.gibbs_ringing:
            k = self._gibbs_kernel
            core = np.apply_along_axis(
                lambda m: np.convolve(m, k, mode="same"), 1, core
            )
        for pol in pols:
            self.tf_plane[pol][:, fslice] += core

    def _spread_line(self, pols, line, center, axis):
        """Add a 1-D event (single channel or single time) to ``pols``. With
        ringing on, spread into a sinc window along ``axis`` (1=freq, 0=time);
        otherwise deposit on the single channel/time."""
        if not self.gibbs_ringing:
            for pol in pols:
                if axis == 1:
                    self.tf_plane[pol][:, center] += line
                else:
                    self.tf_plane[pol][center, :] += line
            return
        k = self._gibbs_kernel
        n_side = (len(k) - 1) // 2
        size = self.freq_bins if axis == 1 else self.time_bins
        lo, hi = max(0, center - n_side), min(size, center + n_side + 1)
        kslice = k[(lo - center + n_side):(hi - center + n_side)]
        for pol in pols:
            if axis == 1:
                self.tf_plane[pol][:, lo:hi] += np.outer(line, kslice)
            else:
                self.tf_plane[pol][lo:hi, :] += np.outer(kslice, line)

    # ------------------------------------------------------------------- data
    def generate_clean_data(self):
        """Generate RFI-free Gaussian data."""
        self.tf_plane = {
            pol: np.random.normal(size=(self.time_bins, self.freq_bins))
            + 1j * np.random.normal(size=(self.time_bins, self.freq_bins))
            for pol in ("RR", "RL", "LR", "LL")
        }
        self.mask = np.zeros((self.time_bins, self.freq_bins), dtype=bool)
        return self.tf_plane, self.mask

    def generate_rfi(self, baseline_frac=None):
        """Generate an RFI-contaminated plane and its full-truth mask.

        Args:
            baseline_frac: baseline length in [0, 1] (short..long) controlling
                the fringe rate. None draws a random value per call.
        """
        self.baseline_frac = (
            np.random.uniform(0.0, 1.0) if baseline_frac is None else float(baseline_frac)
        )
        self.generate_clean_data()
        T, F = self.time_bins, self.freq_bins
        t_col = np.arange(T)[:, None]
        floor = self.detect_floor

        # Broadband RFI: 2-3 separated frequency chunks.
        for _ in range(np.random.randint(2, 4)):
            max_width = F - 1
            freq_start = np.random.randint(0, max(1, max_width - 100))
            freq_width = np.random.randint(50, min(150, max_width - freq_start))
            drifting = np.random.rand() < self.drift_prob
            params = self._draw_event_phase(freq_width, T, drifting)
            modulation = np.random.uniform(0.5, 2.0, size=(T, freq_width))
            power = np.random.choice(self.power_range, size=(T, freq_width))
            n_row = np.arange(freq_start, freq_start + freq_width)[None, :]
            field = (modulation * power) * np.exp(1j * self._phase_grid(t_col, n_row, params))
            fslice = slice(freq_start, freq_start + freq_width)
            self.mask[:, fslice] |= np.abs(field) > floor
            self._spread_block(("RR", "LL"), fslice, field)

        # Narrowband RFI: single channels, ~5% of the band.
        t_lin = np.arange(T)
        for _ in range(int(F * 0.05)):
            freq_idx = np.random.randint(0, F)
            rfi_val = np.random.choice(self.power_range)
            drifting = np.random.rand() < self.drift_prob
            params = self._draw_event_phase(1, T, drifting)
            modulation = np.random.uniform(0.5, 2.0, size=T)
            field = (modulation * rfi_val) * np.exp(1j * self._phase_grid(t_lin, freq_idx, params))
            self.mask[np.abs(field) > floor, freq_idx] = True
            self._spread_line(("RR", "LL"), field, freq_idx, axis=1)

        # Time-bursty RFI: single time rows, ~10% of the scan.
        f_lin = np.arange(F)
        for _ in range(int(T * 0.1)):
            time_idx = np.random.randint(0, T)
            rfi_val = np.random.choice(self.power_range)
            params = self._draw_event_phase(F, 1, drifting=False)
            modulation = np.random.uniform(0.5, 2.0, size=F)
            field = (modulation * rfi_val) * np.exp(1j * self._phase_grid(time_idx, f_lin, params))
            self.mask[time_idx, np.abs(field) > floor] = True
            self._spread_line(("RR", "LL"), field, time_idx, axis=0)

        # Linear sweeps.
        for _ in range(5):
            start_t = np.random.randint(0, T // 2)
            start_f = np.random.randint(0, F // 2)
            slope = np.random.uniform(-2, 2)
            drifting = np.random.rand() < self.drift_prob
            params = self._draw_event_phase(1, T // 2, drifting)
            for i in range(T // 2):
                f_idx = int(start_f + slope * i) % F
                t_idx = (start_t + i) % T
                amp = np.random.choice(self.power_range)
                val = amp * np.exp(1j * self._phase_grid(t_idx, f_idx, params))
                for pol in ("RR", "LL"):
                    self.tf_plane[pol][t_idx, f_idx] += val
                if amp > floor:
                    self.mask[t_idx, f_idx] = True

        # Quadratic (time^2) sweeps.
        for _ in range(5):
            start_t = np.random.randint(0, T // 4)
            start_f = np.random.randint(0, F // 4)
            direction = np.random.choice([-1, 1])
            params = self._draw_event_phase(1, T // 4, drifting=True)
            for t in range(T // 4):
                f_idx = int(start_f + direction * (t**2) // 100) % F
                t_idx = (start_t + t) % T
                amp = np.random.choice(self.power_range)
                val = amp * np.exp(1j * self._phase_grid(t_idx, f_idx, params))
                self.tf_plane["RR"][t_idx, f_idx] += val
                if amp > floor:
                    self.mask[t_idx, f_idx] = True

        # Cross-hand RFI inherits the (coherent) parallel-hand structure.
        for pol in ("RL", "LR"):
            polarization_factor = np.random.uniform(0, 1, size=(T, F))
            self.tf_plane[pol] += polarization_factor * self.tf_plane["RR"]

        return self.tf_plane, self.mask
