"""
lif.py — Leaky Integrate-and-Fire network for the FlyCyborg brain.

Follows Shiu et al. (Nature 2024) parameters:
  tau_m  = 20 ms     membrane time constant
  V_rest = -65 mV    resting potential
  V_th   = -50 mV    spike threshold
  V_reset= -65 mV    reset after spike
  t_ref  =  2 ms     absolute refractory period
  R      =   1 MΩ    membrane resistance (normalised)

The network is event-driven in the sense that the weight matrix
multiplication skips neurons that did not spike last step (sparse).

Usage
-----
    net = LIFNet(brain.weights, dt_ms=0.5, seed=42)
    for t in range(n_steps):
        spikes = net.step(input_current)   # input_current shape (n,)
    rates_hz = net.spike_counts / (n_steps * dt_ms * 1e-3)
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Default LIF parameters (Shiu et al. 2024)
# --------------------------------------------------------------------------- #
TAU_M_MS: float = 20.0     # membrane time constant [ms]
V_REST: float = -65.0      # resting / reset potential [mV]
V_TH: float = -50.0        # threshold potential [mV]
T_REF_MS: float = 2.0      # refractory period [ms]
R_M: float = 1.0           # membrane resistance [normalised MΩ]


class LIFNet:
    """Vectorised LIF network operating on a sparse weight matrix.

    Parameters
    ----------
    weights : scipy sparse (n × n)  (post, pre) signed weight matrix.
    dt_ms   : simulation timestep in milliseconds.
    seed    : random seed for noise.
    tau_m, v_rest, v_th, t_ref, r_m : LIF biophysical parameters.
    noise_std : per-neuron Gaussian current noise (mV/√ms).
    """

    def __init__(
        self,
        weights,
        dt_ms: float = 0.5,
        seed: int = 0,
        tau_m: float = TAU_M_MS,
        v_rest: float = V_REST,
        v_th: float = V_TH,
        t_ref_ms: float = T_REF_MS,
        r_m: float = R_M,
        noise_std: float = 0.5,
    ):
        self.W = weights          # keep as sparse (csc preferred)
        self.n = weights.shape[0]
        self.dt = dt_ms
        self.tau = tau_m
        self.v_rest = v_rest
        self.v_th = v_th
        self.t_ref = t_ref_ms
        self.r_m = r_m
        self.noise_std = noise_std
        self._rng = np.random.default_rng(seed)

        # State vectors
        self.v = np.full(self.n, v_rest, dtype=np.float32)
        self.ref = np.zeros(self.n, dtype=np.float32)  # remaining refractory time [ms]
        self.t_ms: float = 0.0

        # Accumulators reset by caller between episodes/windows
        self.spike_counts = np.zeros(self.n, dtype=np.int32)
        self.last_spikes = np.zeros(self.n, dtype=bool)

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #

    def reset(self, seed: int | None = None) -> None:
        """Reset membrane potentials and spike counters."""
        self.v[:] = self.v_rest
        self.ref[:] = 0.0
        self.t_ms = 0.0
        self.spike_counts[:] = 0
        self.last_spikes[:] = False
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # Single timestep
    # ------------------------------------------------------------------ #

    def step(self, input_hz: np.ndarray) -> np.ndarray:
        """Advance the network by one dt.

        Parameters
        ----------
        input_hz : (n,) float32
            External input firing rate [Hz] for each neuron.
            Converted to a current I = R_m * rate * dt * 1e-3.

        Returns
        -------
        spikes : (n,) bool array — which neurons fired this step.
        """
        dt = self.dt
        # Convert Hz → current injection  [mV units, consistent with R_m=1]
        # I_ext = rate [Hz] * dt [ms] * 1e-3  [ms→s] * R_m [normalised]
        I_ext = input_hz.astype(np.float32) * float(dt * 1e-3) * self.r_m

        # Synaptic input from previous spikes (sparse mat-vec)
        if self.last_spikes.any():
            pre_idx = np.flatnonzero(self.last_spikes)
            # W is (post, pre) → sum columns of active pre-neurons
            I_syn = np.asarray(
                self.W[:, pre_idx].sum(axis=1), dtype=np.float32
            ).ravel() * float(dt * 1e-3)
        else:
            I_syn = np.zeros(self.n, dtype=np.float32)

        # Noise
        noise = self._rng.standard_normal(self.n).astype(np.float32) * (
            self.noise_std * float(np.sqrt(dt))
        )

        # Euler integration of LIF:
        #   dV/dt = (V_rest - V) / tau  +  R_m * (I_ext + I_syn)
        dv = (dt / self.tau) * (self.v_rest - self.v) + I_ext + I_syn + noise

        # Update — but not for neurons in refractory
        in_ref = self.ref > 0.0
        self.v = np.where(in_ref, self.v_rest, self.v + dv)

        # Detect spikes
        spikes = (~in_ref) & (self.v >= self.v_th)

        # Reset spiked neurons
        self.v[spikes] = self.v_rest

        # Update refractory timer
        self.ref = np.where(spikes, self.t_ref, np.maximum(0.0, self.ref - dt))

        self.last_spikes = spikes
        self.spike_counts += spikes.astype(np.int32)
        self.t_ms += dt

        return spikes

    # ------------------------------------------------------------------ #
    # Multi-step convenience
    # ------------------------------------------------------------------ #

    def run(self, input_hz: np.ndarray, steps: int) -> np.ndarray:
        """Run for *steps* timesteps with constant input.

        Parameters
        ----------
        input_hz : (n,) constant input firing rate.
        steps    : number of dt-steps.

        Returns
        -------
        rates_hz : (n,) mean output firing rate [Hz] over the window.
        """
        counts = np.zeros(self.n, dtype=np.int32)
        for _ in range(steps):
            counts += self.step(input_hz).astype(np.int32)
        window_s = steps * self.dt * 1e-3
        return counts / max(window_s, 1e-9)

    def tick(self, input_hz: np.ndarray, ms: float) -> dict[str, float]:
        """Run for *ms* milliseconds; return dict of group mean rates.

        Matches the interface expected by MotorDecoder.update().
        Call brain.tick(rates, ms=10.0) to advance 10 ms.

        Returns dict mapping neuron-group names to mean Hz — the caller
        is expected to pass a group_map dict.  This base method returns
        the raw spike counts; CyborgPilot wraps it.
        """
        steps = max(1, int(round(ms / self.dt)))
        for _ in range(steps):
            self.step(input_hz)
        window_s = steps * self.dt * 1e-3
        rates = self.spike_counts / max(window_s, 1e-9)
        # Reset accumulator for next window
        self.spike_counts[:] = 0
        return rates
