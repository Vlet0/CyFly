"""
encoder.py — AdapterIn: camera features → neuron input firing rates.

The encoder is the **first evolved component**.  It is a linear projection
from the 10-dimensional binocular camera feature vector to per-neuron
input rates (Hz) for each visual neuron group.

Architecture
------------
    features (10,) = [left×5 | right×5]
         │
         │  W_enc  (n_visual × 10)  ← evolved weights
         ▼
    visual_rates (n_visual,)         [Hz, clipped to [0, MAX_HZ]]
         │
         │  spread to other groups via tonic_drive + group_mask
         ▼
    input_rates (n,)                 full neuron array

The rest of the brain (CX, DN, etc.) receives only tonic background drive
unless the encoder explicitly targets them.  This matches the biological
architecture where only the visual system receives retinal input.

A hand-designed baseline encoder is provided (BaselineEncoder) that maps
brightness and depth to L/R visual neurons without any learned weights.
This is used as the initial population seed in CMA-ES.
"""

from __future__ import annotations

import numpy as np

from .camera import CameraFeatures, N_FEATURES

MAX_HZ: float = 300.0       # maximum firing rate injected
TONIC_HZ: float = 5.0       # background tonic drive to all neurons
VISUAL_SCALE: float = 150.0  # scale factor for visual input


class InputEncoder:
    """Parameterised encoder: CameraFeatures → input_rates (n_neurons,).

    Parameters
    ----------
    brain : CyborgBrain
        Needed for n and group indices.
    W_enc : np.ndarray, shape (n_visual_L, N_FEATURES) or None
        Weight matrix for left visual neurons.  If None, uses baseline.
    W_enc_r : same for right (if None, mirrors W_enc).
    tonic : float
        Constant background current to all neurons.
    """

    def __init__(
        self,
        brain,
        W_enc_l: np.ndarray | None = None,
        W_enc_r: np.ndarray | None = None,
        tonic: float = TONIC_HZ,
    ):
        self.brain = brain
        self.tonic = tonic

        vis_l = brain.group("visual", "L")
        vis_r = brain.group("visual", "R")
        self._vis_l = vis_l
        self._vis_r = vis_r
        n_l = max(vis_l.size, 1)
        n_r = max(vis_r.size, 1)

        # Initialise weights: each output neuron gets a weight per feature
        if W_enc_l is None:
            self.W_l = _baseline_weights(n_l, side="L")
        else:
            self.W_l = np.asarray(W_enc_l, dtype=np.float32).reshape(n_l, N_FEATURES)

        if W_enc_r is None:
            self.W_r = _baseline_weights(n_r, side="R")
        else:
            self.W_r = np.asarray(W_enc_r, dtype=np.float32).reshape(n_r, N_FEATURES)

    # ------------------------------------------------------------------ #
    # Parameter vector interface (for CMA-ES)
    # ------------------------------------------------------------------ #

    @property
    def param_size(self) -> int:
        return int(self.W_l.size + self.W_r.size + 1)  # +1 for tonic

    def get_params(self) -> np.ndarray:
        return np.concatenate([
            self.W_l.ravel(),
            self.W_r.ravel(),
            [self.tonic],
        ]).astype(np.float64)

    def set_params(self, params: np.ndarray) -> None:
        params = np.asarray(params, dtype=np.float32)
        n_l = self.W_l.size
        n_r = self.W_r.size
        self.W_l = params[:n_l].reshape(self.W_l.shape)
        self.W_r = params[n_l:n_l + n_r].reshape(self.W_r.shape)
        self.tonic = float(params[n_l + n_r]) if len(params) > n_l + n_r else TONIC_HZ

    # ------------------------------------------------------------------ #
    # Encoding
    # ------------------------------------------------------------------ #

    def encode(self, features: CameraFeatures) -> np.ndarray:
        """Map camera features to input firing rates for the full network.

        Returns
        -------
        input_rates : np.ndarray shape (n,) in Hz, clipped to [0, MAX_HZ].
        """
        n = self.brain.n
        rates = np.full(n, self.tonic, dtype=np.float32)

        feat_l = features.left.to_array()   # (N_FEATURES,)
        feat_r = features.right.to_array()  # (N_FEATURES,)

        if self._vis_l.size:
            v_l = self.W_l @ feat_l * VISUAL_SCALE   # (n_vis_l,)
            rates[self._vis_l] += v_l

        if self._vis_r.size:
            v_r = self.W_r @ feat_r * VISUAL_SCALE   # (n_vis_r,)
            rates[self._vis_r] += v_r

        return np.clip(rates, 0.0, MAX_HZ)


# ── Baseline (hand-designed) ───────────────────────────────────────────────── #

class BaselineEncoder(InputEncoder):
    """Simple hand-designed encoder — no learning needed.

    Maps:
      brightness   → uniform drive to all visual neurons
      depth_near   → increased drive (obstacle signal)
      edge_density → increased drive (rich texture = interesting region)

    Used as the starting point before CMA-ES evolution.
    """

    def __init__(self, brain):
        super().__init__(brain)
        # Override: set hand-designed weights
        n_l = self.W_l.shape[0]
        n_r = self.W_r.shape[0]
        self.W_l = _baseline_weights(n_l, side="L")
        self.W_r = _baseline_weights(n_r, side="R")

    def encode(self, features: CameraFeatures) -> np.ndarray:
        return super().encode(features)


def _baseline_weights(n: int, side: str) -> np.ndarray:
    """Initial hand-designed weight matrix (n × N_FEATURES).

    Feature order: [brightness, edge_density, depth_near, flow_h, looming]
    """
    W = np.zeros((n, N_FEATURES), dtype=np.float32)
    # All neurons get equal share of each feature
    W[:, 0] = 1.0 / max(n, 1)   # brightness
    W[:, 1] = 0.5 / max(n, 1)   # edge density
    W[:, 2] = 2.0 / max(n, 1)   # depth_near (obstacle signal — upweighted)
    W[:, 3] = 0.3 / max(n, 1)   # flow_h
    W[:, 4] = 3.0 / max(n, 1)   # looming (danger signal — most upweighted)
    return W
