"""
01_run_single_episode.py
========================
Run one FlyCyborg navigation episode with the baseline (hand-designed) adapter.

Modes
-----
  --mock          No AI2-THOR, no nav_lite.npz needed.  Uses MockThorEnv +
                  a tiny random weight matrix.  Good for checking the loop runs.

  --real          Requires ai2thor installed (Python 3.8-3.10) and nav_lite.npz
                  from FlyIdea.  Downloads ~500MB Unity build on first run.

Usage
-----
    # Quick sanity check (no dependencies beyond numpy/scipy):
    python examples/01_run_single_episode.py --mock

    # Real brain + real simulator:
    python examples/01_run_single_episode.py --real --scene FloorPlan1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))


def make_mock_brain():
    """Create a tiny synthetic brain for testing (no nav_lite.npz needed)."""
    from scipy import sparse
    from flycyborg.brain.connectome import CyborgBrain

    n = 200
    rng = np.random.default_rng(0)
    # Random sparse weight matrix
    density = 0.05
    data = rng.standard_normal(int(n * n * density)).astype(np.float32)
    row = rng.integers(0, n, len(data))
    col = rng.integers(0, n, len(data))
    W = sparse.csc_matrix((data, (row, col)), shape=(n, n))

    types = np.array(
        ["R1"] * 20 + ["T4a"] * 20 + ["EPG"] * 30 + ["DNg02"] * 20
        + ["MN-wing"] * 20 + ["leg MN"] * 20 + ["haltere"] * 10
        + ["other"] * (n - 140),
        dtype=str,
    )
    sides = np.array(
        ["L"] * (n // 2) + ["R"] * (n - n // 2), dtype=str
    )

    groups = {
        "visual":  np.arange(0, 40, dtype=np.int64),
        "cx":      np.arange(40, 70, dtype=np.int64),
        "dn":      np.arange(70, 90, dtype=np.int64),
        "wing":    np.arange(90, 110, dtype=np.int64),
        "leg":     np.arange(110, 130, dtype=np.int64),
        "haltere": np.arange(130, 140, dtype=np.int64),
        "neck":    np.array([], dtype=np.int64),
    }

    return CyborgBrain(
        weights=W,
        types=types,
        sides=sides,
        superclass=None,
        body_ids=None,
        groups=groups,
        meta={"name": "mock-synthetic", "note": "Testing only"},
    )


def main():
    parser = argparse.ArgumentParser(description="Run a single FlyCyborg episode")
    parser.add_argument("--mock", action="store_true", default=True,
                        help="Use MockThorEnv + synthetic brain (default)")
    parser.add_argument("--real", action="store_true",
                        help="Use real AI2-THOR + nav_lite.npz")
    parser.add_argument("--nav-lite", default="../FlyIdea/data/nav_lite.npz")
    parser.add_argument("--scene", default="FloorPlan1")
    parser.add_argument("--adapter", default=None, help="Load evolved adapter .npz")
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    use_real = args.real and not args.mock

    print("=" * 60)
    print("  FlyCyborg — Single Episode Demo")
    print(f"  Mode : {'REAL AI2-THOR' if use_real else 'MOCK (no AI2-THOR)'}")
    print("=" * 60)

    # ── 1. Brain ─────────────────────────────────────────────────────── #
    if use_real:
        from flycyborg.brain.connectome import CyborgBrain
        brain = CyborgBrain.from_npz(args.nav_lite)
    else:
        brain = make_mock_brain()
    print(f"\n{brain.summary()}\n")

    # ── 2. LIF network ───────────────────────────────────────────────── #
    from flycyborg.brain.lif import LIFNet
    net = LIFNet(brain.weights, dt_ms=0.5, seed=42)

    # ── 3. Encoder + Decoder ─────────────────────────────────────────── #
    from flycyborg.senses.encoder import InputEncoder
    from flycyborg.motor.decoder import MotorDecoder
    encoder = InputEncoder(brain)
    decoder = MotorDecoder(brain)

    # ── 4. Load evolved adapter if provided ──────────────────────────── #
    if args.adapter:
        from flycyborg.evo.adapter import AdapterParams
        adapter = AdapterParams(encoder, decoder)
        adapter.load(args.adapter)
        print(f"Loaded adapter: {args.adapter}")
        print(f"Adapter: {adapter}")

    # ── 5. Environment ───────────────────────────────────────────────── #
    if use_real:
        from flycyborg.env.ai2thor_env import ThorEnv
        env = ThorEnv(scene=args.scene, render_depth=True)
        env.reset()
    else:
        from flycyborg.env.ai2thor_env import MockThorEnv
        env = MockThorEnv(seed=0)
        env.reset()

    # ── 6. Task ──────────────────────────────────────────────────────── #
    from flycyborg.env.nav_task import PointNavTask
    task = PointNavTask(max_steps=args.steps)

    # ── 7. Run episode ───────────────────────────────────────────────── #
    from flycyborg.runtime import run_episode
    from flycyborg.evo.fitness import compute_fitness

    print(f"Running episode (max {args.steps} steps)...\n")
    record = run_episode(
        env=env,
        brain=brain,
        net=net,
        encoder=encoder,
        decoder=decoder,
        task=task,
        verbose=True,
        collect_steps=False,
    )

    # ── 8. Results ───────────────────────────────────────────────────── #
    fitness = compute_fitness(record)
    print("\n" + "=" * 60)
    print("  Episode Results")
    print("=" * 60)
    print(f"  Success        : {record.success}")
    print(f"  Steps taken    : {record.n_steps}")
    print(f"  Collisions     : {record.n_collisions}")
    print(f"  Path (actual)  : {record.actual_path_m:.2f} m")
    print(f"  Path (optimal) : {record.optimal_path_m:.2f} m")
    print(f"  Final dist     : {record.final_dist_m:.2f} m")
    print(f"  SPL            : {fitness.spl:.4f}")
    print(f"  Fitness total  : {fitness.total:+.4f}")
    if record.spike_counts is not None:
        sparsity = (record.spike_counts == 0).mean() * 100
        print(f"  Spike sparsity : {sparsity:.1f}% silent neurons")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
