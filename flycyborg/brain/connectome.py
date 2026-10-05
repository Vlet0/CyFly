"""
connectome.py — Load FlyIdea nav_lite.npz and expose the neuron groups
needed by the FlyCyborg encoder and decoder.

This is a thin wrapper around FlyIdea's NavLiteDB.  It:
  - Loads the .npz built by malecns_nav_lite.builder
  - Exposes the weight matrix, group indices, and side labels
  - Provides a split of visual/cx/dn/wing/leg groups by hemisphere (L/R)
  - Lazily computes a sensorimotor-core subgraph if requested

Usage
-----
    brain = CyborgBrain.from_npz("../FlyIdea/data/nav_lite.npz")
    print(brain.summary())
    idx_dn_l = brain.group("dn", side="L")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class CyborgBrain:
    """Lightweight wrapper around a NavLiteDB .npz for use in FlyCyborg.

    Attributes
    ----------
    weights : scipy.sparse.csc_matrix  (n × n), (post, pre)
        Signed synaptic weight matrix.
    types : np.ndarray[str]  (n,)
        Cell-type label per neuron.
    sides : np.ndarray[str]  (n,)
        Hemisphere: "L", "R", or "".
    groups : dict[str, np.ndarray]
        Named index arrays: visual, cx, dn, wing, leg, haltere, neck.
    n : int
        Total number of neurons.
    """

    # Groups that may carry left/right information
    BILATERAL_GROUPS = {"visual", "cx", "dn", "wing", "leg", "haltere", "neck"}

    def __init__(
        self,
        weights,
        types: np.ndarray,
        sides: np.ndarray,
        groups: dict[str, np.ndarray],
        meta: dict,
        superclass: np.ndarray | None = None,
        body_ids: np.ndarray | None = None,
    ):
        self.weights = weights
        self.types = types
        self.sides = sides
        self.superclass = superclass
        self.body_ids = body_ids
        self.groups = groups
        self.meta = meta

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def from_npz(cls, path: str | Path) -> "CyborgBrain":
        """Load directly from a nav_lite.npz produced by FlyIdea."""
        import json
        from scipy import sparse

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"nav_lite.npz not found at {path}.\n"
                "Run FlyIdea's builder first:\n"
                "  cd ../FlyIdea && python -m malecns_nav_lite build"
            )

        z = np.load(path, allow_pickle=False)
        W = sparse.csc_matrix(
            (z["data"], z["indices"], z["indptr"]),
            shape=tuple(z["shape"]),
        )
        groups = {
            k: np.asarray(v, dtype=np.int64)
            for k, v in json.loads(str(z["groups"])).items()
        }
        meta = json.loads(str(z["meta"]))

        return cls(
            weights=W,
            types=z["types"],
            sides=z["sides"],
            groups=groups,
            meta=meta,
        )

    @classmethod
    def from_flyidea(cls, flyidea_root: str | Path | None = None) -> "CyborgBrain":
        """Convenience: resolve FlyIdea root automatically and load."""
        if flyidea_root is None:
            # Default: ../FlyIdea relative to this file's grandparent
            flyidea_root = Path(__file__).parent.parent.parent.parent / "FlyIdea"
        return cls.from_npz(Path(flyidea_root) / "data" / "nav_lite.npz")

    # ------------------------------------------------------------------ #
    # Group access
    # ------------------------------------------------------------------ #

    @property
    def n(self) -> int:
        return int(self.weights.shape[0])

    def group(self, name: str, side: str | None = None) -> np.ndarray:
        """Return neuron indices for a named group, optionally filtered by side.

        Parameters
        ----------
        name : str
            One of: visual, cx, dn, wing, leg, haltere, neck.
        side : str | None
            "L", "R", or None (both).

        Returns
        -------
        np.ndarray of int64 indices into the weight matrix.
        """
        idx = self.groups.get(name, np.array([], dtype=np.int64))
        if side is None or idx.size == 0:
            return idx
        mask = np.char.upper(self.sides[idx].astype(str)) == side.upper()
        return idx[mask]

    def bilateral(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (left_indices, right_indices) for a bilateral group."""
        return self.group(name, "L"), self.group(name, "R")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #

    def summary(self) -> str:
        lines = [
            "CyborgBrain",
            f"  Source    : {self.meta.get('name', 'unknown')}",
            f"  Neurons   : {self.n:,}",
            f"  Connections: {int(self.weights.nnz):,}",
            "  Groups:",
        ]
        for name in ["visual", "cx", "dn", "wing", "leg", "haltere", "neck"]:
            idx = self.groups.get(name, np.array([]))
            l_count = (np.char.upper(self.sides[idx].astype(str)) == "L").sum() if idx.size else 0
            r_count = (np.char.upper(self.sides[idx].astype(str)) == "R").sum() if idx.size else 0
            lines.append(
                f"    {name:10s}: {idx.size:5,}  (L={l_count}, R={r_count})"
            )
        return "\n".join(lines)
