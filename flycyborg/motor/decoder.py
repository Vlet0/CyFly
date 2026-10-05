"""
decoder.py — AdapterOut: neuron firing rates → WheelCommand.

The decoder is the **second evolved component** of FlyCyborg.

Biological mapping (Dual-Mode as described in contex.txt):
  Wing motor neurons  (MN-wing / DN-wing group)
      → speed signal  → robot forward velocity
  Leg motor neurons   (MN-leg group, asymmetric L vs R)
      → steering signal → robot yaw / turn rate
  Descending neurons  (DN group, asymmetric L vs R)
      → combined steering + speed modulation
  Giant Fiber         (DNp01)
      → escape reflex → full brake + backup

Architecture of the evolved adapter
------------------------------------
For each output axis (forward, turn, brake):

    axis_value = gain * Σ_i  W_i * (rate_i − baseline_i)

where W_i are evolved per-group weights and baseline_i is the resting
rate measured during a settling period.

This is the same transparent linear read-out used by FlyDrones, but
with weights evolved end-to-end from navigation task fitness rather than
fitted by regression on artificial stimuli.
"""

from __future__ import annotations

import numpy as np

from .command import WheelCommand

# ── Constants ──────────────────────────────────────────────────────────────── #
SETTLE_S: float = 1.0        # seconds to measure baseline before nav starts
SMOOTHING: float = 0.4       # EMA smoothing for output axes
DEADZONE: float = 0.05       # ignore signals below this magnitude
ESCAPE_HZ: float = 30.0      # giant-fiber rate threshold for escape reflex
ESCAPE_DURATION_S: float = 0.8

# Default group weights before evolution (hand-designed)
# Keys: neuron group names as returned by brain.group()
DEFAULT_WEIGHTS = {
    # forward axis: wing drives speed, DN bilateral sums
    "forward": {
        "wing_L": +1.0,
        "wing_R": +1.0,
        "dn_L":   +0.3,
        "dn_R":   +0.3,
    },
    # turn axis: leg and dn asymmetry drives yaw
    "turn": {
        "leg_L":  +1.0,
        "leg_R":  -1.0,
        "dn_L":   +0.5,
        "dn_R":   -0.5,
    },
    # brake axis: high bilateral activity = obstacle → slow down
    "brake": {
        "dn_L":  +0.5,
        "dn_R":  +0.5,
    },
}

GAIN = {"forward": 0.4, "turn": 0.6, "brake": 0.3}


class MotorDecoder:
    """Linear read-out from neuron group rates to WheelCommand.

    Parameters
    ----------
    brain : CyborgBrain
        Used to look up group indices and split by side.
    weights : dict | None
        Nested dict {axis: {group_side: weight}}.  If None, uses DEFAULT_WEIGHTS.
    gain : dict | None
        Per-axis gain. If None, uses GAIN.
    smoothing : float
        EMA coefficient for output smoothing.
    settle_s : float
        Duration (seconds) to run settling before baselines are locked.
    """

    def __init__(
        self,
        brain,
        weights: dict | None = None,
        gain: dict | None = None,
        smoothing: float = SMOOTHING,
        settle_s: float = SETTLE_S,
    ):
        self.brain = brain
        self.weights = weights or {k: dict(v) for k, v in DEFAULT_WEIGHTS.items()}
        self.gain = gain or dict(GAIN)
        self.smoothing = smoothing
        self.settle_s = settle_s

        # Build group → neuron index mapping (split by side)
        self._groups: dict[str, np.ndarray] = {}
        for gname in ["visual", "cx", "dn", "wing", "leg", "haltere"]:
            for side in ["L", "R"]:
                key = f"{gname}_{side}"
                self._groups[key] = brain.group(gname, side)

        # Baseline (measured during settle)
        self._baseline: dict[str, float] = {}
        self._settle_sum: dict[str, float] = {}
        self._settle_n: int = 0
        self._elapsed: float = 0.0

        # Smoothed output state
        self._state = {"forward": 0.0, "turn": 0.0, "brake": 0.0}

        # Escape reflex
        self._gf_idx = brain.group("dn")   # giant fibre is in DN group
        self._gf_smooth: float = 0.0
        self._escape_until: float = -1.0

    # ------------------------------------------------------------------ #
    # Parameter vector interface (for CMA-ES)
    # ------------------------------------------------------------------ #

    @property
    def param_size(self) -> int:
        """Total number of scalar parameters."""
        n = 0
        for axis_weights in self.weights.values():
            n += len(axis_weights)
        n += len(self.gain)
        return n

    def get_params(self) -> np.ndarray:
        vals = []
        for axis in ["forward", "turn", "brake"]:
            for v in self.weights[axis].values():
                vals.append(v)
        for axis in ["forward", "turn", "brake"]:
            vals.append(self.gain[axis])
        return np.array(vals, dtype=np.float64)

    def set_params(self, params: np.ndarray) -> None:
        params = list(params)
        idx = 0
        for axis in ["forward", "turn", "brake"]:
            for k in list(self.weights[axis].keys()):
                self.weights[axis][k] = float(params[idx])
                idx += 1
        for axis in ["forward", "turn", "brake"]:
            self.gain[axis] = float(params[idx])
            idx += 1

    # ------------------------------------------------------------------ #
    # Group rate helper
    # ------------------------------------------------------------------ #

    def _group_rate(self, rates_hz: np.ndarray, key: str) -> float:
        """Mean firing rate of neurons in group `key` (e.g. 'dn_L')."""
        idx = self._groups.get(key, np.array([]))
        if idx.size == 0:
            return 0.0
        return float(rates_hz[idx].mean())

    # ------------------------------------------------------------------ #
    # Update
    # ------------------------------------------------------------------ #

    def update(self, rates_hz: np.ndarray, dt: float) -> WheelCommand:
        """Convert firing rates to WheelCommand.

        Parameters
        ----------
        rates_hz : (n,) array — firing rate per neuron [Hz].
        dt       : elapsed time since last call [seconds].

        Returns
        -------
        WheelCommand (clamped to [-1, 1]).
        """
        self._elapsed += dt

        # ── Settling phase: measure baseline ────────────────────────── #
        if self._elapsed < self.settle_s:
            for key in self._groups:
                self._settle_sum[key] = (
                    self._settle_sum.get(key, 0.0) + self._group_rate(rates_hz, key)
                )
            self._settle_n += 1
            return WheelCommand.stop(note="settling")

        if self._settle_n and not self._baseline:
            self._baseline = {
                k: v / self._settle_n for k, v in self._settle_sum.items()
            }

        # ── Compute raw axis values ──────────────────────────────────── #
        raw = {}
        for axis in ["forward", "turn", "brake"]:
            val = 0.0
            for group_key, w in self.weights[axis].items():
                r = self._group_rate(rates_hz, group_key)
                b = self._baseline.get(group_key, r)
                val += w * (r - b)
            raw[axis] = self.gain[axis] * val

        # ── Smooth ──────────────────────────────────────────────────── #
        a = self.smoothing
        for axis in ["forward", "turn", "brake"]:
            v = max(-1.0, min(1.0, raw[axis]))
            self._state[axis] = (1 - a) * self._state[axis] + a * v

        out = {
            k: 0.0 if abs(v) < DEADZONE else v
            for k, v in self._state.items()
        }
        cmd = WheelCommand(**out)

        # ── Giant-fiber escape reflex ────────────────────────────────── #
        if self._gf_idx.size:
            gf_now = float(rates_hz[self._gf_idx].max())
            self._gf_smooth = 0.6 * self._gf_smooth + 0.4 * gf_now
            if self._gf_smooth >= ESCAPE_HZ and self._elapsed > self._escape_until:
                self._escape_until = self._elapsed + ESCAPE_DURATION_S

        if self._elapsed <= self._escape_until:
            cmd = WheelCommand(
                forward=-0.8,  # reverse
                turn=0.0,
                brake=0.0,
                note="giant-fiber escape",
            )

        return cmd.clamp()

    def reset(self) -> None:
        self._baseline = {}
        self._settle_sum = {}
        self._settle_n = 0
        self._elapsed = 0.0
        self._state = {"forward": 0.0, "turn": 0.0, "brake": 0.0}
        self._gf_smooth = 0.0
        self._escape_until = -1.0
