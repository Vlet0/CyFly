"""CLI entry point: python -m flycyborg or flycyborg <command>"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="flycyborg",
        description="FlyCyborg — Drosophila connectome controller for wheeled robot navigation",
    )
    sub = parser.add_subparsers(dest="command")

    # ── run ────────────────────────────────────────────────────────────── #
    p_run = sub.add_parser("run", help="Run a single navigation episode")
    p_run.add_argument("--config", default="configs/default.yaml")
    p_run.add_argument("--adapter", default=None, help="Path to evolved adapter .npz")
    p_run.add_argument("--scene", default=None)
    p_run.add_argument("--mock", action="store_true", help="Use MockThorEnv (no ai2thor)")
    p_run.add_argument("--verbose", action="store_true")

    # ── evolve ─────────────────────────────────────────────────────────── #
    p_evo = sub.add_parser("evolve", help="Run CMA-ES evolution")
    p_evo.add_argument("--config", default="configs/default.yaml")
    p_evo.add_argument("--generations", type=int, default=None)
    p_evo.add_argument("--mock", action="store_true")
    p_evo.add_argument("--resume", action="store_true", help="Resume from last checkpoint")

    # ── inspect ────────────────────────────────────────────────────────── #
    p_ins = sub.add_parser("inspect", help="Print brain summary")
    p_ins.add_argument("--nav-lite", default="../FlyIdea/data/nav_lite.npz")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # ── Load config ────────────────────────────────────────────────────── #
    import yaml
    from pathlib import Path

    if hasattr(args, "config"):
        cfg_path = Path(args.config)
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
        else:
            print(f"[warn] Config not found: {cfg_path}. Using defaults.")
            cfg = {}
    else:
        cfg = {}

    # ── Commands ───────────────────────────────────────────────────────── #
    if args.command == "inspect":
        from .brain.connectome import CyborgBrain
        brain = CyborgBrain.from_npz(args.nav_lite)
        print(brain.summary())

    elif args.command == "run":
        from .runtime import CyborgPilot
        pilot = CyborgPilot.from_config(cfg, mock=args.mock)
        if args.adapter:
            pilot.adapter.load(args.adapter)
        scene = args.scene or cfg.get("env", {}).get("scene", "FloorPlan1")
        if not args.mock:
            pilot.env.reset(scene=scene)
        record = pilot.run_episode(scene=None, verbose=args.verbose)
        from .evo.fitness import compute_fitness
        fit = compute_fitness(record)
        print(f"\nResult: success={record.success} | SPL={fit.spl:.3f} | "
              f"fitness={fit.total:+.4f} | collisions={record.n_collisions}")

    elif args.command == "evolve":
        from .runtime import CyborgPilot
        pilot = CyborgPilot.from_config(cfg, mock=args.mock)
        evo_cfg = cfg.get("evo", {})
        n_gen = args.generations or evo_cfg.get("n_generations", 200)
        if args.resume:
            loaded = pilot.adapter.load(
                Path(evo_cfg.get("results_dir", "results")) / "best_adapter.npz"
            )
            if loaded is not None:
                print(f"Resumed from checkpoint.")
        pilot.evolve(n_generations=n_gen)


if __name__ == "__main__":
    main()
