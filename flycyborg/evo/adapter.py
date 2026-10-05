"""
adapter.py — AdapterParams: flat parameter vector ↔ encoder + decoder weights.

The adapter is the ONLY thing CMA-ES evolves.  It is a thin wrapper that
packs/unpacks the weights of InputEncoder and MotorDecoder into a single
1-D numpy vector suitable for CMA-ES.

Diagram
-------
    AdapterParams.vector  (D,)
         │
         ├─ [0 : enc_size]         → InputEncoder.set_params()
         └─ [enc_size : enc_size+dec_size] → MotorDecoder.set_params()

The connectome weights (brain.weights) are NEVER part of this vector.
"""

from __future__ import annotations

import numpy as np


class AdapterParams:
    """Flat parameter vector that owns encoder + decoder weights.

    Parameters
    ----------
    encoder : InputEncoder
    decoder : MotorDecoder
    """

    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
        self._enc_size = encoder.param_size
        self._dec_size = decoder.param_size

    # ------------------------------------------------------------------ #
    # Size
    # ------------------------------------------------------------------ #

    @property
    def size(self) -> int:
        return self._enc_size + self._dec_size

    # ------------------------------------------------------------------ #
    # Pack / unpack
    # ------------------------------------------------------------------ #

    def to_vector(self) -> np.ndarray:
        """Pack current weights into a flat float64 array."""
        return np.concatenate([
            self.encoder.get_params(),
            self.decoder.get_params(),
        ]).astype(np.float64)

    def from_vector(self, v: np.ndarray) -> None:
        """Unpack a flat vector back into encoder + decoder."""
        v = np.asarray(v, dtype=np.float64)
        if v.size != self.size:
            raise ValueError(
                f"Expected vector of size {self.size}, got {v.size}"
            )
        self.encoder.set_params(v[:self._enc_size])
        self.decoder.set_params(v[self._enc_size:])

    # ------------------------------------------------------------------ #
    # Initialisation helpers
    # ------------------------------------------------------------------ #

    def random_vector(self, rng: np.random.Generator | None = None, scale: float = 0.1) -> np.ndarray:
        """Perturb current weights with Gaussian noise."""
        if rng is None:
            rng = np.random.default_rng()
        return self.to_vector() + rng.standard_normal(self.size) * scale

    # ------------------------------------------------------------------ #
    # Save / load
    # ------------------------------------------------------------------ #

    def save(self, path) -> None:
        """Save the parameter vector to a .npz file."""
        import pathlib
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, params=self.to_vector())

    def load(self, path) -> None:
        """Load parameter vector from a .npz file."""
        z = np.load(path)
        self.from_vector(z["params"])

    def __repr__(self) -> str:
        return (
            f"AdapterParams(enc={self._enc_size}d, dec={self._dec_size}d, "
            f"total={self.size}d)"
        )
