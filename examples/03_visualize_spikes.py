"""
03_visualize_spikes.py
======================
Visualize spike rasters and neuron group activity during a navigation episode.

Plots
-----
  1. Spike raster (top N active neurons, colour-coded by group)
  2. Mean firing rate per group over time
  3. Motor command trace (forward / turn) over time
  4. Distance to goal over time

Usage
-----
    python examples/03_visualize_spikes.py --mock
    python examples/03_visualize_spikes.py --adapter results/best_adapter.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

GROUP_COLORS = {
    "visual": "#4cc9f0",
    "cx":     "#f72585",
    "dn":     "#7209b7",
    "wing":   "#3a0ca3",
    "leg":    "#4361ee",
    "haltere":"#4895ef",
    "other":  "#aaaaaa",
}


def main():
    parser = argparse.ArgumentParser(description="Visualize FlyCyborg spike activity")
    parser.add_argument("--mock", action="store_true", default=True)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--nav-lite", default="../FlyIdea/data/nav_lite.npz")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--out", default=None, help="Save figure to file instead of showing")
    args = parser.parse_args()

    use_real = args.real and not args.mock

    # ── Build components (same as example 01) ────────────────────────── #
    from scipy import sparse
    from flycyborg.brain.connectome import CyborgBrain
    from flycyborg.brain.lif import LIFNet
    from flycyborg.senses.encoder import InputEncoder
    from flycyborg.motor.decoder import MotorDecoder
    from flycyborg.senses.camera import FeatureExtractor

    if use_real:
        brain = CyborgBrain.from_npz(args.nav_lite)
    else:
        n = 200
        rng = np.random.default_rng(0)
        data = rng.standard_normal(int(n * n * 0.05)).astype(np.float32)
        row = rng.integers(0, n, len(data))
        col = rng.integers(0, n, len(data))
        W = sparse.csc_matrix((data, (row, col)), shape=(n, n))
        types = np.array(
            ["R1"]*20 + ["T4a"]*20 + ["EPG"]*30 + ["DNg02"]*20
            + ["MN-wing"]*20 + ["leg MN"]*20 + ["haltere"]*10 + ["other"]*60, dtype=str)
        sides = np.array(["L"]*(n//2) + ["R"]*(n - n//2), dtype=str)
        groups = {
            "visual": np.arange(0, 40, dtype=np.int64),
            "cx":     np.arange(40, 70, dtype=np.int64),
            "dn":     np.arange(70, 90, dtype=np.int64),
            "wing":   np.arange(90, 110, dtype=np.int64),
            "leg":    np.arange(110, 130, dtype=np.int64),
            "haltere":np.arange(130, 140, dtype=np.int64),
            "neck":   np.array([], dtype=np.int64),
        }
        brain = CyborgBrain(weights=W, types=types, sides=sides,
                            superclass=None, body_ids=None,
                            groups=groups, meta={"name": "mock"})

    net = LIFNet(brain.weights, dt_ms=0.5, seed=42)
    encoder = InputEncoder(brain)
    decoder = MotorDecoder(brain)

    if args.adapter:
        from flycyborg.evo.adapter import AdapterParams
        adapter = AdapterParams(encoder, decoder)
        adapter.load(args.adapter)
        print(f"Loaded adapter from {args.adapter}")

    if use_real:
        from flycyborg.env.ai2thor_env import ThorEnv
        env = ThorEnv(render_depth=True)
        env.reset()
    else:
        from flycyborg.env.ai2thor_env import MockThorEnv
        env = MockThorEnv(seed=1)
        env.reset()

    from flycyborg.env.nav_task import PointNavTask
    task = PointNavTask(max_steps=args.steps)
    full_meta = env.get_full_metadata()
    task.reset(full_meta)

    # ── Collect step data ────────────────────────────────────────────── #
    cam = FeatureExtractor()
    net.reset()
    decoder.reset()

    LIF_STEPS = 20   # 10 ms at dt=0.5 ms
    history_rates = []    # (T, n_groups)
    history_cmd = []      # (T, 2) forward, turn
    history_dist = []     # (T,)
    history_raster = []   # (T, n_neurons) bool

    group_names = ["visual", "cx", "dn", "wing", "leg", "haltere"]
    done = False

    while not done:
        frame, depth, _ = env.observe()
        feats = cam.extract(frame, depth)
        rates_in = encoder.encode(feats)

        step_spikes = np.zeros(brain.n, dtype=np.int32)
        for _ in range(LIF_STEPS):
            s = net.step(rates_in)
            step_spikes += s.astype(np.int32)

        rates_hz = step_spikes / max(LIF_STEPS * net.dt * 1e-3, 1e-9)
        cmd = decoder.update(rates_hz, dt=0.01)
        action = cmd.to_thor_action()
        env.step(action)

        agent_meta = {"position": {"x": env._x if hasattr(env, "_x") else 0,
                                   "y": 0, "z": env._z if hasattr(env, "_z") else 0}}
        success, done, dist = task.step(agent_meta)

        # Record
        group_rates = [float(rates_hz[brain.groups.get(g, np.array([]))].mean())
                       if brain.groups.get(g, np.array([])).size else 0.0
                       for g in group_names]
        history_rates.append(group_rates)
        history_cmd.append([cmd.forward, cmd.turn])
        history_dist.append(dist)
        history_raster.append(step_spikes > 0)

    env.close()

    # ── Plot ─────────────────────────────────────────────────────────── #
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not installed. Install: pip install matplotlib")
        return

    T = len(history_rates)
    rates_arr = np.array(history_rates)   # (T, G)
    cmd_arr = np.array(history_cmd)        # (T, 2)
    dist_arr = np.array(history_dist)      # (T,)
    raster_arr = np.array(history_raster)  # (T, N)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("FlyCyborg — Spike Activity During Navigation", fontsize=13, fontweight="bold")

    # Plot 1: Spike raster
    ax = axes[0]
    # Show top 60 neurons by total spike count
    total = raster_arr.sum(axis=0)
    top_idx = np.argsort(total)[-60:][::-1]

    # Color by group
    neuron_colors = []
    for ni in top_idx:
        color = GROUP_COLORS["other"]
        for gname in group_names:
            if ni in brain.groups.get(gname, np.array([])):
                color = GROUP_COLORS[gname]
                break
        neuron_colors.append(color)

    for row_i, (ni, col) in enumerate(zip(top_idx, neuron_colors)):
        spike_times = np.where(raster_arr[:, ni])[0]
        ax.scatter(spike_times, np.full_like(spike_times, row_i),
                   c=col, s=2, linewidths=0)
    ax.set_ylabel("Neuron (top 60)")
    ax.set_title("Spike Raster")
    legend_patches = [mpatches.Patch(color=GROUP_COLORS[g], label=g) for g in group_names]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=7, ncol=3)

    # Plot 2: Group firing rates
    ax = axes[1]
    ts = np.arange(T)
    for gi, gname in enumerate(group_names):
        ax.plot(ts, rates_arr[:, gi], label=gname, color=GROUP_COLORS[gname], linewidth=1.2)
    ax.set_ylabel("Rate (Hz)")
    ax.set_title("Mean Firing Rate per Neuron Group")
    ax.legend(fontsize=7, ncol=3, loc="upper right")

    # Plot 3: Motor commands
    ax = axes[2]
    ax.plot(ts, cmd_arr[:, 0], label="forward", color="#2ecc71", linewidth=1.5)
    ax.plot(ts, cmd_arr[:, 1], label="turn", color="#e74c3c", linewidth=1.5)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Command [-1, 1]")
    ax.set_title("Motor Output (forward / turn)")
    ax.legend(fontsize=8)
    ax.set_ylim(-1.2, 1.2)

    # Plot 4: Distance to goal
    ax = axes[3]
    ax.plot(ts, dist_arr, color="#9b59b6", linewidth=1.5)
    ax.set_ylabel("Distance (m)")
    ax.set_xlabel("Environment Step")
    ax.set_title("Distance to Goal")

    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
