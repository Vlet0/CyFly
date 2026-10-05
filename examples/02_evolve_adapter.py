"""
02_evolve_adapter.py
====================
Run the CMA-ES evolutionary loop to optimise the FlyCyborg adapter.

The adapter (InputEncoder + MotorDecoder weights) is evolved while the
connectome weights remain fixed.  Fitness = mean SPL + bonuses across
multiple navigation episodes per candidate.

Usage
-----
    # Quick mock run (no AI2-THOR, no data files):
    python examples/02_evolve_adapter.py --mock --generations 5

    # Full evolution with real brain + simulator:
    python examples/02_evolve_adapter.py --config configs/malecns.yaml

    # Resume from checkpoint:
    python examples/02_evolve_adapter.py --resume --config configs/malecns.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Evolve FlyCyborg adapter with CMA-ES")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mock", action="store_true",
                        help="Use MockThorEnv + synthetic brain")
    parser.add_argument("--generations", type=int, default=None,
                        help="Override n_generations from config")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint in results_dir")
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────── #
    import yaml
    cfg_path = Path(args.config)
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        print(f"[warn] Config {cfg_path} not found, using defaults")
        cfg = {}

    evo_cfg = cfg.get("evo", {})
    n_gen = args.generations or evo_cfg.get("n_generations", 50)
    results_dir = Path(evo_cfg.get("results_dir", "results"))

    # ── Build brain ──────────────────────────────────────────────────── #
    if args.mock:
        # Import helper from example 01
        sys.path.insert(0, str(Path(__file__).parent))
        from _01_run_single_episode_helpers import make_mock_brain  # noqa: F401
        # Inline it here to avoid cross-file dependency
        from scipy import sparse
        from flycyborg.brain.connectome import CyborgBrain

        n = 200
        rng = np.random.default_rng(cfg.get("seed", 0))
        data = rng.standard_normal(int(n * n * 0.05)).astype(np.float32)
        row = rng.integers(0, n, len(data))
        col = rng.integers(0, n, len(data))
        W = sparse.csc_matrix((data, (row, col)), shape=(n, n))
        types = np.array(
            ["R1"] * 20 + ["T4a"] * 20 + ["EPG"] * 30 + ["DNg02"] * 20
            + ["MN-wing"] * 20 + ["leg MN"] * 20 + ["haltere"] * 10
            + ["other"] * (n - 140), dtype=str,
        )
        sides = np.array(["L"] * (n // 2) + ["R"] * (n - n // 2), dtype=str)
        groups = {
            "visual":  np.arange(0, 40, dtype=np.int64),
            "cx":      np.arange(40, 70, dtype=np.int64),
            "dn":      np.arange(70, 90, dtype=np.int64),
            "wing":    np.arange(90, 110, dtype=np.int64),
            "leg":     np.arange(110, 130, dtype=np.int64),
            "haltere": np.arange(130, 140, dtype=np.int64),
            "neck":    np.array([], dtype=np.int64),
        }
        brain = CyborgBrain(weights=W, types=types, sides=sides,
                            superclass=None, body_ids=None,
                            groups=groups, meta={"name": "mock"})
    else:
        from flycyborg.brain.connectome import CyborgBrain
        nav_lite = cfg.get("brain", {}).get("nav_lite_path", "../FlyIdea/data/nav_lite.npz")
        brain = CyborgBrain.from_npz(nav_lite)

    print(brain.summary())

    # ── LIF, encoder, decoder ────────────────────────────────────────── #
    from flycyborg.brain.lif import LIFNet
    from flycyborg.senses.encoder import InputEncoder
    from flycyborg.motor.decoder import MotorDecoder
    from flycyborg.evo.adapter import AdapterParams

    net = LIFNet(brain.weights, dt_ms=cfg.get("brain", {}).get("dt_ms", 0.5),
                 seed=cfg.get("seed", 42))
    encoder = InputEncoder(brain)
    decoder = MotorDecoder(brain)
    adapter = AdapterParams(encoder, decoder)
    print(f"\nAdapter: {adapter}")

    # ── Environment ──────────────────────────────────────────────────── #
    if args.mock:
        from flycyborg.env.ai2thor_env import MockThorEnv
        env = MockThorEnv(seed=cfg.get("seed", 0))
        env.reset()
        scenes = ["MockScene"]
    else:
        from flycyborg.env.ai2thor_env import ThorEnv
        env_cfg = cfg.get("env", {})
        env = ThorEnv(**{k: env_cfg[k] for k in
                        ["scene", "width", "height", "fov", "grid_size",
                         "headless", "render_depth"] if k in env_cfg})
        env.reset()
        scenes = evo_cfg.get("scenes", ["FloorPlan1", "FloorPlan2", "FloorPlan3"])

    # ── Resume ───────────────────────────────────────────────────────── #
    if args.resume:
        ckpt = results_dir / "best_adapter.npz"
        if ckpt.exists():
            adapter.load(ckpt)
            print(f"Resumed adapter from {ckpt}")

    # ── Build eval function ──────────────────────────────────────────── #
    from flycyborg.runtime import make_eval_fn
    from flycyborg.env.nav_task import PointNavTask

    eval_fn = make_eval_fn(
        env=env,
        brain=brain,
        net=net,
        encoder=encoder,
        decoder=decoder,
        adapter=adapter,
        task_cls=PointNavTask,
        scenes=scenes,
        episodes_per_candidate=evo_cfg.get("episodes_per_candidate", 3),
        verbose=False,
    )

    # ── Run CMA-ES ───────────────────────────────────────────────────── #
    from flycyborg.evo.cmaes import EvolutionLoop

    evo = EvolutionLoop(
        adapter=adapter,
        eval_fn=eval_fn,
        sigma0=evo_cfg.get("sigma0", 0.3),
        popsize=evo_cfg.get("popsize", None),
        results_dir=results_dir,
        seed=cfg.get("seed", 42),
    )

    best_vec = evo.run(
        n_generations=n_gen,
        checkpoint_every=evo_cfg.get("checkpoint_every", 10),
    )

    # ── Final evaluation ────────────────────────────────────────────── #
    print("\nFinal evaluation with best adapter (5 episodes)...")
    adapter.from_vector(best_vec)
    from flycyborg.env.nav_task import PointNavTask
    from flycyborg.runtime import run_episode
    from flycyborg.evo.fitness import compute_fitness, aggregate_fitness

    final_records = []
    for i, sc in enumerate(scenes[:5]):
        if not args.mock:
            env.reset(scene=sc)
        task = PointNavTask()
        net.reset()
        decoder.reset()
        rec = run_episode(env, brain, net, encoder, decoder, task, verbose=False)
        f = compute_fitness(rec)
        final_records.append(rec)
        print(f"  Scene {sc}: success={rec.success} SPL={f.spl:.3f} fitness={f.total:+.4f}")

    print(f"\nMean fitness: {aggregate_fitness(final_records):+.4f}")
    env.close()


if __name__ == "__main__":
    main()
