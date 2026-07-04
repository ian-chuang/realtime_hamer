"""Hand detector via rtmlib Hand (rtmdet-nano + rtmpose-m, onnxruntime-gpu)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from rtmlib import Hand

HandSide = Literal["left", "right"]

_HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)


@dataclass
class HandDet:
    box: list[float]
    is_right: bool
    kpts: np.ndarray
    scores: np.ndarray
    score: float


def create_detector(
    hand: HandSide = "right",
    device: str = "cuda",
    score_thr: float = 0.4,
    min_kpts: int = 8,
    min_mean_score: float = 0.45,
    min_box_size: float = 20.0,
):
    """Return ``detector(frame) -> list[HandDet]`` with at most one hand.

    Uses the hand-specific RTMPose models (smaller/faster than wholebody).
    The chosen ``hand`` side is assigned to the best detection (no L/R flipping).
    """
    if hand not in ("left", "right"):
        raise ValueError("hand must be 'left' or 'right'")
    is_right = hand == "right"

    # Import torch first so CUDA libs are loaded before ORT creates sessions.
    import torch  # noqa: F401

    pose_model = Hand(mode="lightweight", backend="onnxruntime", device=device)

    def detector(frame: np.ndarray) -> list[HandDet]:
        all_kpts, all_scores = pose_model(frame)
        best: HandDet | None = None
        for kpts, scores in zip(all_kpts, all_scores):
            valid = scores > score_thr
            n_valid = int(valid.sum())
            if n_valid < min_kpts:
                continue
            mean_score = float(scores[valid].mean())
            if mean_score < min_mean_score:
                continue
            if scores[0] < score_thr:
                continue

            pts = kpts[valid]
            x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
            x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
            w, h = x1 - x0, y1 - y0
            if w <= 1 or h <= 1:
                continue
            if w > h:
                d = (w - h) / 2
                y0, y1 = y0 - d, y1 + d
            else:
                d = (h - w) / 2
                x0, x1 = x0 - d, x1 + d
            if min(x1 - x0, y1 - y0) < min_box_size:
                continue

            det = HandDet(
                box=[x0, y0, x1, y1],
                is_right=is_right,
                kpts=kpts.astype(np.float32),
                scores=scores.astype(np.float32),
                score=mean_score,
            )
            if best is None or det.score > best.score:
                best = det
        return [best] if best is not None else []

    return detector


def draw_hands(frame_bgr: np.ndarray, hands: list[HandDet]) -> np.ndarray:
    """Overlay hand keypoints, bones, and boxes on a BGR frame."""
    out = frame_bgr.copy()
    for hand in hands:
        color = (143, 120, 36) if hand.is_right else (241, 138, 133)
        x0, y0, x1, y1 = map(int, hand.box)
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
        label = f"{'R' if hand.is_right else 'L'} {hand.score:.2f}"
        cv2.putText(out, label, (x0, max(0, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        kpts, scores = hand.kpts, hand.scores
        for a, b in _HAND_EDGES:
            if scores[a] > 0.3 and scores[b] > 0.3:
                cv2.line(out, tuple(kpts[a].astype(int)), tuple(kpts[b].astype(int)), color, 2)
        for pt, sc in zip(kpts, scores):
            if sc > 0.3:
                cv2.circle(out, tuple(pt.astype(int)), 3, color, -1)
    return out
