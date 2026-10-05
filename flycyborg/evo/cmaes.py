"""
cmaes.py — CMA-ES evolutionary loop for the FlyCyborg adapter.

Uses pycma (pip install cma) which implements Hansen's CMA-ES.

Loop per generation:
  1. Ask CMA-ES for `popsize` candidate adapter vectors
  2. For each candidate:
       a. Unpack vector → encoder + decoder weights
       b. Run `episodes_per_candidate` navigation episodes
       c. Compute mean fitness (SPL + bonuses)
  3. Tell CMA-ES the fitness values (negated, since pycma minimises)
  4. Save best adapter, log stats, optionally checkpoint

Parallelism:
  Set n_workers > 1 to evaluate candidates in parallel using
  multiprocessing.Pool.  Each worker gets its own AI2-THOR controller.
  Note: AI2-THOR requires a separate Unity process per worker, so
  n_workers is limited by available CPU / memory.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np


class EvolutionLoop:
    """CMA-ES loop for evolving the FlyCyborg adapter.

    Parameters
    ----------
    adapter : AdapterParams
        Initial adapter (defines dimensionality and starting point).
    eval_fn : Callable[[np.ndarray], float]
        Function that takes a flat adapter vector and returns a scalar
        fitness (higher is better).  This is called once per candidate.
    sigma0 : float
        Initial CMA-ES step size.
    popsize : int | None
        Population size.  If None, pycma chooses automatically (≈4+3*ln(D)).
    results_dir : str | Path
        Directory where checkpoints and logs are saved.
    seed : int
        Random seed for CMA-ES.
    """

    def __init__(
        self,
        adapter,
        eval_fn: Callable[[np.ndarray], float],
        sigma0: float = 0.3,
        popsize: int | None = None,
        results_dir: str | Path = "results",
        seed: int = 42,
    ):
        self.adapter = adapter
        self.eval_fn = eval_fn
        self.sigma0 = sigma0
        self.popsize = popsize
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed

        # Logging
        self.log: list[dict] = []
        self.best_fitness: float = -float("inf")
        self.best_vector: np.ndarray | None = None
        self.generation: int = 0

    # ------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------ #

    def run(
        self,
        n_generations: int = 200,
        callback: Callable | None = None,
        checkpoint_every: int = 10,
    ) -> np.ndarray:
        """Run the CMA-ES loop for *n_generations* generations.

        Parameters
        ----------
        n_generations : int
            Number of generations to run.
        callback : Callable | None
            Called after each generation with (generation, log_entry).
        checkpoint_every : int
            Save best adapter every this many generations.

        Returns
        -------
        best_vector : np.ndarray
            The adapter vector with the highest fitness found.
        """
        try:
            import cma
        except ImportError:
            raise ImportError(
                "pycma is required for evolution.\n"
                "Install: pip install cma"
            )

        x0 = self.adapter.to_vector()
        options = {
            "seed": self.seed,
            "verbose": -9,  # suppress pycma output (we print ourselves)
            "tolx": 1e-8,
            "tolfun": 1e-9,
            "maxiter": n_generations,
        }
        if self.popsize is not None:
            options["popsize"] = self.popsize

        es = cma.CMAEvolutionStrategy(x0, self.sigma0, options)

        print(f"\n{'─'*60}")
        print(f"  FlyCyborg Evolution — CMA-ES")
        print(f"  Adapter dims : {self.adapter.size}")
        print(f"  Population   : {es.popsize}")
        print(f"  Generations  : {n_generations}")
        print(f"{'─'*60}\n")

        while not es.stop() and self.generation < n_generations:
            t0 = time.perf_counter()

            # Ask for candidates
            candidates = es.ask()   # list of np.ndarray

            # Evaluate (negate because pycma minimises)
            fitnesses_neg = []
            for vec in candidates:
                try:
                    f = self.eval_fn(np.asarray(vec))
                except Exception as exc:
                    print(f"  [warn] eval_fn raised: {exc}")
                    f = -1.0
                fitnesses_neg.append(-float(f))

            es.tell(candidates, fitnesses_neg)

            # Track best
            best_idx = int(np.argmin(fitnesses_neg))
            gen_best_fitness = -fitnesses_neg[best_idx]
            gen_mean_fitness = -float(np.mean(fitnesses_neg))

            if gen_best_fitness > self.best_fitness:
                self.best_fitness = gen_best_fitness
                self.best_vector = np.asarray(candidates[best_idx]).copy()

            elapsed = time.perf_counter() - t0
            entry = {
                "generation": self.generation,
                "best_fitness": gen_best_fitness,
                "mean_fitness": gen_mean_fitness,
                "global_best": self.best_fitness,
                "sigma": float(es.sigma),
                "elapsed_s": elapsed,
            }
            self.log.append(entry)

            print(
                f"  Gen {self.generation:4d} | "
                f"best={gen_best_fitness:+.4f}  mean={gen_mean_fitness:+.4f}  "
                f"global_best={self.best_fitness:+.4f}  σ={es.sigma:.4f}  "
                f"({elapsed:.1f}s)"
            )

            if callback is not None:
                callback(self.generation, entry)

            if (self.generation + 1) % checkpoint_every == 0:
                self._checkpoint()

            self.generation += 1

        # Final checkpoint
        self._checkpoint()
        self._save_log()

        if self.best_vector is not None:
            print(f"\n  Evolution complete. Best fitness: {self.best_fitness:+.4f}")
            print(f"  Best adapter saved to: {self.results_dir / 'best_adapter.npz'}")

        return self.best_vector if self.best_vector is not None else x0

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #

    def _checkpoint(self) -> None:
        if self.best_vector is None:
            return
        # Save best adapter
        np.savez_compressed(
            self.results_dir / "best_adapter.npz",
            params=self.best_vector,
            fitness=np.array([self.best_fitness]),
            generation=np.array([self.generation]),
        )
        # Save latest log
        self._save_log()

    def _save_log(self) -> None:
        with open(self.results_dir / "evolution_log.json", "w") as f:
            json.dump(self.log, f, indent=2)

    # ------------------------------------------------------------------ #
    # Resume
    # ------------------------------------------------------------------ #

    def load_best(self) -> np.ndarray | None:
        """Load best adapter from disk if available."""
        path = self.results_dir / "best_adapter.npz"
        if not path.exists():
            return None
        z = np.load(path)
        self.best_vector = z["params"]
        self.best_fitness = float(z["fitness"][0])
        self.generation = int(z["generation"][0])
        print(f"Loaded best adapter: fitness={self.best_fitness:.4f} at gen {self.generation}")
        return self.best_vector
