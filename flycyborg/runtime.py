"""
runtime.py — The 7-step FlyCyborg closed loop.

    ┌──────────────────────────────────────────────────────────┐
    │  Step 1: SENSE    — get RGB + Depth from AI2-THOR        │
    │  Step 2: ENCODE   — AdapterIn: pixels → neuron rates     │
    │  Step 3: SIMULATE — LIF tick: rates → spikes             │
    │  Step 4: DECODE   — AdapterOut: spikes → WheelCommand    │
    │  Step 5: ACT      — send action to AI2-THOR              │
    │  Step 6: OBSERVE  — measure outcome, update task state   │
    │  Step 7: ADAPT    — (per episode) update CMA-ES fitness  │
    └──────────────────────────────────────────────────────────┘

Two entry points:
  run_episode()   — run a single episode, return EpisodeRecord
  make_eval_fn()  — factory that returns an eval_fn(vector) → float
                    suitable for EvolutionLoop
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .brain.connectome import CyborgBrain
from .brain.lif import LIFNet
from .env.ai2thor_env import ThorEnv
from .env.nav_task import PointNavTask
from .evo.fitness import EpisodeRecord, compute_fitness
from .motor.decoder import MotorDecoder
from .senses.camera import FeatureExtractor
from .senses.encoder import InputEncoder

# ── Timing ────────────────────────────────────────────────────────────────── #
LIF_MS_PER_STEP: float = 10.0   # ms of LIF simulation per environment step
LIF_DT_MS: float = 0.5          # LIF timestep


@dataclass
class StepInfo:
    """Data recorded for a single environment step."""
    t: float
    action: str
    features_l: np.ndarray      # (N_FEATURES,)
    features_r: np.ndarray
    rates_hz: np.ndarray        # (n,) mean firing rates this step
    spike_counts: np.ndarray    # (n,)
    forward: float
    turn: float
    brake: float
    success: bool
    done: bool
    dist_to_goal: float
    collision: bool


@dataclass
class EpisodeInfo:
    """All data from one complete episode."""
    scene: str
    steps: list[StepInfo] = field(default_factory=list)
    record: EpisodeRecord = field(default_factory=EpisodeRecord)
    elapsed_s: float = 0.0


# ── Core: single episode ───────────────────────────────────────────────────── #

def run_episode(
    env: ThorEnv,
    brain: CyborgBrain,
    net: LIFNet,
    encoder: InputEncoder,
    decoder: MotorDecoder,
    task: PointNavTask | None = None,
    lif_ms_per_step: float = LIF_MS_PER_STEP,
    verbose: bool = False,
    collect_steps: bool = False,
) -> EpisodeRecord:
    """Run one navigation episode and return an EpisodeRecord.

    Parameters
    ----------
    env     : ThorEnv (already reset to desired scene)
    brain   : CyborgBrain
    net     : LIFNet (will be reset at episode start)
    encoder : InputEncoder
    decoder : MotorDecoder
    task    : PointNavTask (will be reset at episode start)
    lif_ms_per_step : ms of LIF simulation per environment step
    verbose : print per-step info
    collect_steps : whether to collect StepInfo objects (memory intensive)

    Returns
    -------
    EpisodeRecord with success, SPL components, collisions, spike stats.
    """
    if task is None:
        task = PointNavTask()

    # ── Reset all stateful components ────────────────────────────────── #
    net.reset()
    decoder.reset()
    cam_extractor = FeatureExtractor()
    lif_steps = max(1, int(round(lif_ms_per_step / net.dt)))

    # ── Reset environment and task ────────────────────────────────────── #
    full_meta = env.get_full_metadata()
    goal = task.reset(full_meta)

    if verbose:
        print(f"  Episode start | goal=({goal.x:.2f}, {goal.z:.2f}) "
              f"| optimal={task.optimal_path_m:.2f}m")

    t_start = time.perf_counter()
    cumulative_spikes = np.zeros(brain.n, dtype=np.int32)
    success = False
    done = False
    dist = task.optimal_path_m
    step_infos: list[StepInfo] = []

    # ── Main loop ─────────────────────────────────────────────────────── #
    while not done:
        t0 = time.perf_counter()

        # ── Step 1: SENSE ──────────────────────────────────────────── #
        frame, depth, _ = env.observe()

        # ── Step 2: ENCODE ─────────────────────────────────────────── #
        cam_features = cam_extractor.extract(frame, depth)
        input_rates = encoder.encode(cam_features)

        # ── Step 3: SIMULATE ───────────────────────────────────────── #
        step_spike_counts = np.zeros(brain.n, dtype=np.int32)
        for _ in range(lif_steps):
            spikes = net.step(input_rates)
            step_spike_counts += spikes.astype(np.int32)
        cumulative_spikes += step_spike_counts

        window_s = lif_steps * net.dt * 1e-3
        rates_hz = step_spike_counts / max(window_s, 1e-9)

        # ── Step 4: DECODE ─────────────────────────────────────────── #
        cmd = decoder.update(rates_hz, dt=lif_ms_per_step * 1e-3)

        # ── Step 5: ACT ────────────────────────────────────────────── #
        action = cmd.to_thor_action()
        env.step(action)

        # ── Step 6: OBSERVE ────────────────────────────────────────── #
        tel = env.telemetry()
        agent_meta = env.get_full_metadata().get("agent", {})
        # Reconstruct minimal agent meta for task.step()
        agent_meta_for_task = {
            "position": {"x": tel.x, "y": 0.9, "z": tel.z},
            "rotation": {"y": tel.yaw_deg},
        }
        success, done, dist = task.step(agent_meta_for_task)

        if verbose:
            print(
                f"    step {task.step_count:3d} | {action:12s} | "
                f"dist={dist:.2f}m | "
                f"fwd={cmd.forward:+.2f} turn={cmd.turn:+.2f} | "
                f"spikes={int(step_spike_counts.sum()):4d} | "
                f"{cmd.note}"
            )

        if collect_steps:
            step_infos.append(StepInfo(
                t=time.perf_counter() - t_start,
                action=action,
                features_l=cam_features.left.to_array(),
                features_r=cam_features.right.to_array(),
                rates_hz=rates_hz.copy(),
                spike_counts=step_spike_counts.copy(),
                forward=cmd.forward,
                turn=cmd.turn,
                brake=cmd.brake,
                success=success,
                done=done,
                dist_to_goal=dist,
                collision=tel.collision,
            ))

    elapsed = time.perf_counter() - t_start

    # ── Step 7: ADAPT (assemble record for CMA-ES) ─────────────────── #
    record = EpisodeRecord(
        success=success,
        optimal_path_m=task.optimal_path_m,
        actual_path_m=task.actual_path_m,
        n_collisions=env.n_collisions,
        n_steps=task.step_count,
        final_dist_m=dist,
        initial_dist_m=task.initial_dist_m,
        spike_counts=cumulative_spikes,
        n_neurons=brain.n,
    )

    if verbose:
        fitness = compute_fitness(record)
        print(
            f"  Episode done  | success={success} | "
            f"SPL={fitness.spl:.3f} | fitness={fitness.total:+.4f} | "
            f"collisions={record.n_collisions} | {elapsed:.1f}s"
        )

    return record


# ── Eval function factory (for EvolutionLoop) ─────────────────────────────── #

def make_eval_fn(
    env: ThorEnv,
    brain: CyborgBrain,
    net: LIFNet,
    encoder: InputEncoder,
    decoder: MotorDecoder,
    adapter,                    # AdapterParams
    task_cls=PointNavTask,
    scenes: list[str] | None = None,
    episodes_per_candidate: int = 3,
    verbose: bool = False,
) -> Any:
    """Return a callable eval_fn(vector) → float for EvolutionLoop.

    Each call:
      1. Unpacks the vector into encoder + decoder via adapter.from_vector()
      2. Runs `episodes_per_candidate` episodes across `scenes`
      3. Returns mean fitness

    Parameters
    ----------
    scenes : list of AI2-THOR scene names to cycle through.
             If None, uses ["FloorPlan1", "FloorPlan2", "FloorPlan3"].
    episodes_per_candidate : number of episodes per fitness evaluation.
    """
    from .evo.fitness import aggregate_fitness

    if scenes is None:
        scenes = ["FloorPlan1", "FloorPlan2", "FloorPlan3"]

    def eval_fn(vector: np.ndarray) -> float:
        # Step 7: unpack adapter
        adapter.from_vector(vector)

        records = []
        for i in range(episodes_per_candidate):
            scene = scenes[i % len(scenes)]
            env.reset(scene=scene)
            task = task_cls()
            rec = run_episode(
                env=env,
                brain=brain,
                net=net,
                encoder=encoder,
                decoder=decoder,
                task=task,
                verbose=verbose,
            )
            records.append(rec)

        return aggregate_fitness(records)

    return eval_fn


# ── CyborgPilot: high-level convenience class ─────────────────────────────── #

class CyborgPilot:
    """All-in-one object that owns all FlyCyborg components.

    Usage
    -----
        pilot = CyborgPilot.from_config(cfg)
        pilot.env.reset("FloorPlan1")
        record = pilot.run_episode(verbose=True)
        pilot.evolve(n_generations=200)
    """

    def __init__(
        self,
        brain: CyborgBrain,
        net: LIFNet,
        encoder: InputEncoder,
        decoder: MotorDecoder,
        env: ThorEnv,
        task_cls=PointNavTask,
        scenes: list[str] | None = None,
        results_dir: str = "results",
    ):
        from .evo.adapter import AdapterParams
        self.brain = brain
        self.net = net
        self.encoder = encoder
        self.decoder = decoder
        self.env = env
        self.task_cls = task_cls
        self.scenes = scenes or ["FloorPlan1", "FloorPlan2", "FloorPlan3"]
        self.adapter = AdapterParams(encoder, decoder)
        self.results_dir = results_dir

    # ------------------------------------------------------------------ #
    # Single episode
    # ------------------------------------------------------------------ #

    def run_episode(
        self,
        scene: str | None = None,
        verbose: bool = True,
        collect_steps: bool = False,
    ) -> EpisodeRecord:
        if scene is not None:
            self.env.reset(scene=scene)
        task = self.task_cls()
        return run_episode(
            env=self.env,
            brain=self.brain,
            net=self.net,
            encoder=self.encoder,
            decoder=self.decoder,
            task=task,
            verbose=verbose,
            collect_steps=collect_steps,
        )

    # ------------------------------------------------------------------ #
    # Evolution
    # ------------------------------------------------------------------ #

    def evolve(
        self,
        n_generations: int = 200,
        episodes_per_candidate: int = 3,
        sigma0: float = 0.3,
        verbose: bool = False,
    ) -> np.ndarray:
        from .evo.cmaes import EvolutionLoop

        eval_fn = make_eval_fn(
            env=self.env,
            brain=self.brain,
            net=self.net,
            encoder=self.encoder,
            decoder=self.decoder,
            adapter=self.adapter,
            task_cls=self.task_cls,
            scenes=self.scenes,
            episodes_per_candidate=episodes_per_candidate,
            verbose=verbose,
        )

        evo = EvolutionLoop(
            adapter=self.adapter,
            eval_fn=eval_fn,
            sigma0=sigma0,
            results_dir=self.results_dir,
        )
        return evo.run(n_generations=n_generations)

    # ------------------------------------------------------------------ #
    # Factory
    # ------------------------------------------------------------------ #

    @classmethod
    def from_config(cls, cfg: dict, mock: bool = False) -> "CyborgPilot":
        """Build a CyborgPilot from a config dict (loaded from YAML).

        Parameters
        ----------
        cfg  : dict parsed from default.yaml / malecns.yaml
        mock : if True, use MockThorEnv (no ai2thor required)
        """
        from .env.ai2thor_env import MockThorEnv

        # Brain
        nav_lite_path = cfg.get("brain", {}).get(
            "nav_lite_path", "../FlyIdea/data/nav_lite.npz"
        )
        brain = CyborgBrain.from_npz(nav_lite_path)
        net = LIFNet(
            brain.weights,
            dt_ms=cfg.get("brain", {}).get("dt_ms", LIF_DT_MS),
            seed=cfg.get("seed", 42),
        )

        # Encoder + decoder
        encoder = InputEncoder(brain)
        decoder = MotorDecoder(brain)

        # Environment
        env_cfg = cfg.get("env", {})
        if mock:
            env = MockThorEnv(
                width=env_cfg.get("width", 300),
                height=env_cfg.get("height", 300),
            )
        else:
            env = ThorEnv(
                scene=env_cfg.get("scene", "FloorPlan1"),
                width=env_cfg.get("width", 300),
                height=env_cfg.get("height", 300),
                fov=env_cfg.get("fov", 90),
                grid_size=env_cfg.get("grid_size", 0.25),
                headless=env_cfg.get("headless", False),
                render_depth=env_cfg.get("render_depth", True),
            )
            env.reset()

        scenes = cfg.get("evo", {}).get("scenes", ["FloorPlan1", "FloorPlan2", "FloorPlan3"])
        results_dir = cfg.get("evo", {}).get("results_dir", "results")

        return cls(
            brain=brain,
            net=net,
            encoder=encoder,
            decoder=decoder,
            env=env,
            scenes=scenes,
            results_dir=results_dir,
        )
