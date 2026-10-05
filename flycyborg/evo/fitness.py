"""
fitness.py — Episode fitness functions for the evolutionary loop.

Primary metric: SPL (Success weighted by Path Length)
    SPL = S * l* / max(l*, p)
where
    S  = 1 if success, 0 if not
    l* = optimal (shortest) path length
    p  = actual path length travelled

Secondary penalties:
    collision_penalty  : each collision subtracts from fitness
    timeout_penalty    : finishing early (few steps) is penalised if no success
    energy_bonus       : sparse spiking → energy efficiency bonus

The final fitness is maximised by CMA-ES (CMA-ES minimises by convention,
so we negate before passing to pycma).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ── Weights ────────────────────────────────────────────────────────────────── #
W_SPL: float = 1.0
W_COLLISION: float = 0.05     # subtracted per collision
W_ENERGY: float = 0.1         # bonus for sparse spikes
W_PROGRESS: float = 0.3       # partial credit for getting closer to goal


@dataclass
class EpisodeRecord:
    """All data collected during one navigation episode."""
    success: bool = False
    optimal_path_m: float = 1.0       # straight-line start→goal distance
    actual_path_m: float = 0.0        # odometry / step-count estimate
    n_collisions: int = 0
    n_steps: int = 0
    final_dist_m: float = float("inf")  # distance to goal at episode end
    initial_dist_m: float = float("inf")
    spike_counts: np.ndarray | None = None   # (n,) total spikes per neuron
    n_neurons: int = 0
    extra: dict = field(default_factory=dict)


@dataclass
class EpisodeFitness:
    """Computed fitness from an EpisodeRecord."""
    spl: float = 0.0
    collision_penalty: float = 0.0
    energy_bonus: float = 0.0
    progress_bonus: float = 0.0
    total: float = 0.0

    def __float__(self) -> float:
        return self.total


def compute_spl(record: EpisodeRecord) -> float:
    """Compute raw SPL score."""
    if not record.success:
        return 0.0
    l_star = max(record.optimal_path_m, 1e-3)
    p = max(record.actual_path_m, l_star)
    return float(l_star / p)


def compute_fitness(record: EpisodeRecord) -> EpisodeFitness:
    """Compute the full fitness from an episode record.

    Returns
    -------
    EpisodeFitness  (total is maximised — negate for pycma).
    """
    spl = compute_spl(record)

    # Collision penalty
    col_pen = W_COLLISION * record.n_collisions

    # Energy bonus: fraction of silent neurons
    if record.spike_counts is not None and record.n_neurons > 0:
        silent_frac = float((record.spike_counts == 0).sum()) / record.n_neurons
        energy_bonus = W_ENERGY * silent_frac
    else:
        energy_bonus = 0.0

    # Progress bonus: partial credit for getting closer even without success
    if not record.success and record.initial_dist_m > 0:
        improvement = max(
            0.0,
            (record.initial_dist_m - record.final_dist_m) / record.initial_dist_m,
        )
        progress_bonus = W_PROGRESS * improvement
    else:
        progress_bonus = 0.0

    total = (
        W_SPL * spl
        + energy_bonus
        + progress_bonus
        - col_pen
    )

    return EpisodeFitness(
        spl=spl,
        collision_penalty=col_pen,
        energy_bonus=energy_bonus,
        progress_bonus=progress_bonus,
        total=total,
    )


def aggregate_fitness(records: list[EpisodeRecord]) -> float:
    """Mean fitness across multiple episodes (one generation evaluation)."""
    if not records:
        return 0.0
    return float(np.mean([compute_fitness(r).total for r in records]))
