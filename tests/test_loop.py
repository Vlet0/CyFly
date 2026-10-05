"""
test_loop.py — Smoke tests for the FlyCyborg 7-step loop.

All tests use MockThorEnv + a tiny synthetic brain so they run without
AI2-THOR, nav_lite.npz, or any GPU.

Run: pytest tests/test_loop.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse


# ── Fixture: tiny synthetic brain ─────────────────────────────────────────── #

@pytest.fixture
def mock_brain():
    from flycyborg.brain.connectome import CyborgBrain

    n = 100
    rng = np.random.default_rng(0)
    data = rng.standard_normal(int(n * n * 0.05)).astype(np.float32)
    row = rng.integers(0, n, len(data))
    col = rng.integers(0, n, len(data))
    W = sparse.csc_matrix((data, (row, col)), shape=(n, n))
    types = np.array(["R1"] * 10 + ["EPG"] * 20 + ["DNg02"] * 10
                     + ["MN-wing"] * 10 + ["leg MN"] * 10 + ["other"] * 40, dtype=str)
    sides = np.array(["L"] * 50 + ["R"] * 50, dtype=str)
    groups = {
        "visual":  np.arange(0, 10, dtype=np.int64),
        "cx":      np.arange(10, 30, dtype=np.int64),
        "dn":      np.arange(30, 40, dtype=np.int64),
        "wing":    np.arange(40, 50, dtype=np.int64),
        "leg":     np.arange(50, 60, dtype=np.int64),
        "haltere": np.array([], dtype=np.int64),
        "neck":    np.array([], dtype=np.int64),
    }
    return CyborgBrain(weights=W, types=types, sides=sides,
                       superclass=None, body_ids=None,
                       groups=groups, meta={"name": "test"})


# ── Tests ─────────────────────────────────────────────────────────────────── #

def test_brain_summary(mock_brain):
    s = mock_brain.summary()
    assert "visual" in s
    assert "100" in s


def test_lif_runs(mock_brain):
    from flycyborg.brain.lif import LIFNet
    net = LIFNet(mock_brain.weights, dt_ms=0.5, seed=0)
    rates = np.full(mock_brain.n, 50.0, dtype=np.float32)
    spikes = net.step(rates)
    assert spikes.shape == (mock_brain.n,)
    assert spikes.dtype == bool


def test_camera_features():
    from flycyborg.senses.camera import FeatureExtractor
    ext = FeatureExtractor()
    frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    depth = np.random.uniform(0.5, 8.0, (100, 100)).astype(np.float32)
    feats = ext.extract(frame, depth)
    arr = feats.to_array()
    assert arr.shape == (10,)
    assert np.all(np.isfinite(arr))


def test_encoder_shape(mock_brain):
    from flycyborg.senses.camera import FeatureExtractor
    from flycyborg.senses.encoder import InputEncoder
    ext = FeatureExtractor()
    enc = InputEncoder(mock_brain)
    frame = np.random.randint(0, 255, (84, 84, 3), dtype=np.uint8)
    feats = ext.extract(frame, None)
    rates = enc.encode(feats)
    assert rates.shape == (mock_brain.n,)
    assert rates.min() >= 0.0


def test_decoder_runs(mock_brain):
    from flycyborg.motor.decoder import MotorDecoder
    dec = MotorDecoder(mock_brain, settle_s=0.0)
    rates = np.zeros(mock_brain.n, dtype=np.float32)
    cmd = dec.update(rates, dt=0.01)
    from flycyborg.motor.command import WheelCommand
    assert isinstance(cmd, WheelCommand)


def test_adapter_roundtrip(mock_brain):
    from flycyborg.senses.encoder import InputEncoder
    from flycyborg.motor.decoder import MotorDecoder
    from flycyborg.evo.adapter import AdapterParams
    enc = InputEncoder(mock_brain)
    dec = MotorDecoder(mock_brain)
    adapter = AdapterParams(enc, dec)
    v = adapter.to_vector()
    assert v.ndim == 1
    adapter.from_vector(v)  # should not raise


def test_full_mock_episode(mock_brain):
    from flycyborg.brain.lif import LIFNet
    from flycyborg.senses.encoder import InputEncoder
    from flycyborg.motor.decoder import MotorDecoder
    from flycyborg.env.ai2thor_env import MockThorEnv
    from flycyborg.env.nav_task import PointNavTask
    from flycyborg.runtime import run_episode

    net = LIFNet(mock_brain.weights, dt_ms=0.5, seed=0)
    enc = InputEncoder(mock_brain)
    dec = MotorDecoder(mock_brain)
    env = MockThorEnv(seed=0)
    env.reset()
    task = PointNavTask(max_steps=10)

    record = run_episode(
        env=env, brain=mock_brain, net=net,
        encoder=enc, decoder=dec, task=task,
        verbose=False,
    )
    assert record.n_steps <= 10
    assert record.n_neurons == mock_brain.n
    assert record.spike_counts is not None


def test_fitness_spl():
    from flycyborg.evo.fitness import EpisodeRecord, compute_fitness

    # Success case
    rec = EpisodeRecord(success=True, optimal_path_m=2.0, actual_path_m=4.0)
    f = compute_fitness(rec)
    assert abs(f.spl - 0.5) < 1e-6

    # Failure case
    rec2 = EpisodeRecord(success=False, optimal_path_m=2.0, actual_path_m=5.0)
    f2 = compute_fitness(rec2)
    assert f2.spl == 0.0
