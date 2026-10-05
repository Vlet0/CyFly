"""
camera.py — Extract visual features from an AI2-THOR RGB+Depth frame.

Features extracted per eye (Left hemisphere / Right hemisphere):
  brightness   : mean luminance of the half-frame
  edge_density : Canny edge density (proxy for texture / obstacle boundary)
  depth_near   : fraction of pixels closer than NEAR_THRESHOLD metres
  flow_h       : horizontal motion energy (frame diff, left-right asymmetry)
  looming      : depth-field contraction speed (approaching object signal)

These mirror the biological feature channels used by the Drosophila visual
system (T4/T5 motion detectors, LC4/LPLC2 looming detectors, L1-L5
luminance pathways) but are computed on standard RGB-D frames rather than
compound-eye optics.  The AdapterIn layer (evolved) maps these features to
actual neuron firing rates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Optional OpenCV — falls back to numpy-only if not installed
try:
    import cv2 as _cv2
    _HAS_CV2 = True
except ImportError:
    _cv2 = None  # type: ignore
    _HAS_CV2 = False

# ── Constants ──────────────────────────────────────────────────────────────── #
NEAR_THRESHOLD_M: float = 1.5   # depth closer than this → "obstacle near"
FAR_CLIP_M: float = 10.0        # clip depth values beyond this
N_FEATURES: int = 5             # per eye: brightness, edge, depth_near, flow_h, looming

FEATURE_NAMES = ["brightness", "edge_density", "depth_near", "flow_h", "looming"]


@dataclass
class EyeFeatures:
    """Per-hemisphere feature vector."""
    brightness: float = 0.0
    edge_density: float = 0.0
    depth_near: float = 0.0
    flow_h: float = 0.0
    looming: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array(
            [self.brightness, self.edge_density, self.depth_near,
             self.flow_h, self.looming],
            dtype=np.float32,
        )


@dataclass
class CameraFeatures:
    """Full binocular feature set for one frame."""
    left: EyeFeatures = field(default_factory=EyeFeatures)
    right: EyeFeatures = field(default_factory=EyeFeatures)
    # Raw frame kept for visualisation
    frame: np.ndarray | None = None

    def to_array(self) -> np.ndarray:
        """Concatenate left + right → shape (2 * N_FEATURES,)."""
        return np.concatenate([self.left.to_array(), self.right.to_array()])


# ── Feature extraction ─────────────────────────────────────────────────────── #

class FeatureExtractor:
    """Stateful extractor — keeps previous frame for motion/looming."""

    def __init__(
        self,
        near_threshold: float = NEAR_THRESHOLD_M,
        far_clip: float = FAR_CLIP_M,
    ):
        self.near_threshold = near_threshold
        self.far_clip = far_clip
        self._prev_depth: np.ndarray | None = None
        self._prev_gray: np.ndarray | None = None

    def reset(self) -> None:
        self._prev_depth = None
        self._prev_gray = None

    def extract(
        self,
        frame: np.ndarray,          # (H, W, 3) uint8 RGB
        depth: np.ndarray | None,   # (H, W) float metres, or None
    ) -> CameraFeatures:
        """Extract binocular features from one frame."""
        H, W = frame.shape[:2]
        mid = W // 2

        gray = _to_gray(frame)  # (H, W) float32 0-1

        # Split into left / right visual fields
        gray_l, gray_r = gray[:, :mid], gray[:, mid:]
        depth_l = depth_r = None
        if depth is not None:
            depth_clipped = np.clip(depth, 0.0, self.far_clip)
            depth_l, depth_r = depth_clipped[:, :mid], depth_clipped[:, mid:]

        feat_l = self._eye(gray_l, depth_l, side="L")
        feat_r = self._eye(gray_r, depth_r, side="R")

        # Horizontal flow asymmetry (whole-frame motion left-right)
        if self._prev_gray is not None:
            diff = np.abs(gray.astype(np.float32) - self._prev_gray)
            flow_total_l = diff[:, :mid].mean()
            flow_total_r = diff[:, mid:].mean()
        else:
            flow_total_l = flow_total_r = 0.0
        feat_l.flow_h = float(flow_total_l)
        feat_r.flow_h = float(flow_total_r)

        # Looming: depth decreasing in central region
        if depth is not None and self._prev_depth is not None:
            center_now = depth_clipped[H//4:3*H//4, W//4:3*W//4].mean()
            center_prev = np.clip(self._prev_depth, 0, self.far_clip)[H//4:3*H//4, W//4:3*W//4].mean()
            loom = max(0.0, float(center_prev - center_now)) / self.far_clip
            feat_l.looming = loom
            feat_r.looming = loom
        else:
            feat_l.looming = 0.0
            feat_r.looming = 0.0

        self._prev_gray = gray.copy()
        self._prev_depth = depth.copy() if depth is not None else None

        return CameraFeatures(left=feat_l, right=feat_r, frame=frame)

    def _eye(
        self,
        gray: np.ndarray,
        depth: np.ndarray | None,
        side: str,
    ) -> EyeFeatures:
        brightness = float(gray.mean())

        # Edge density
        if _HAS_CV2:
            gray_u8 = (gray * 255).astype(np.uint8)
            edges = _cv2.Canny(gray_u8, threshold1=50, threshold2=150)
            edge_density = float(edges.mean()) / 255.0
        else:
            # Numpy Sobel fallback
            gx = np.diff(gray, axis=1, prepend=gray[:, :1])
            gy = np.diff(gray, axis=0, prepend=gray[:1, :])
            edge_density = float(np.sqrt(gx**2 + gy**2).mean())

        # Depth near fraction
        if depth is not None and depth.size > 0:
            depth_near = float((depth < self.near_threshold).mean())
        else:
            depth_near = 0.0

        return EyeFeatures(
            brightness=brightness,
            edge_density=edge_density,
            depth_near=depth_near,
            flow_h=0.0,   # filled in by caller
            looming=0.0,  # filled in by caller
        )


def extract_features(
    frame: np.ndarray,
    depth: np.ndarray | None = None,
    extractor: FeatureExtractor | None = None,
) -> CameraFeatures:
    """Stateless convenience wrapper (no motion/looming)."""
    if extractor is not None:
        return extractor.extract(frame, depth)
    tmp = FeatureExtractor()
    return tmp.extract(frame, depth)


# ── Helpers ────────────────────────────────────────────────────────────────── #

def _to_gray(frame: np.ndarray) -> np.ndarray:
    """RGB uint8 (H,W,3) → float32 (H,W) luminance 0-1."""
    f = frame.astype(np.float32) / 255.0
    # BT.601 luminance
    return 0.299 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 2]
