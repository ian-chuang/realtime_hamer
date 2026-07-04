"""Wholebody RTMPose hand detector (onnxruntime-gpu, TensorRT EP when available).

Uses the same wholebody RTMPose family as hamer-demo (lightweight YOLOX-tiny +
RTMW), which is the lightweight stand-in for HaMeR's ViTDet/ViTPose detector.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from rtmlib import Wholebody

HandSide = Literal["left", "right"]

_NUM_HAND_KPTS = 21
_L_HAND = 91
_R_HAND = 91 + _NUM_HAND_KPTS

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


def _ort_providers():
    """CUDA EP only.

    ORT's TensorrtExecutionProvider links libnvinfer.so.10, which conflicts with
    tensorrt-cu12 (TRT 11) used for HaMeR engines. HaMeR uses TRT directly;
    RTMPose stays on ORT CUDA (same as a fast GPU path without that conflict).
    """
    import onnxruntime as ort

    providers = []
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def _patch_sessions_cuda(pose_model: Wholebody) -> None:
    """Ensure rtmlib sessions use CUDA EP (not CPU fallback)."""
    import onnxruntime as ort

    providers = _ort_providers()
    print(f"RTMPose ORT providers: {providers}")
    for name in ("det_model", "pose_model"):
        sub = getattr(pose_model, name)
        sub.session = ort.InferenceSession(sub.onnx_model, providers=providers)


def create_detector(
    hand: HandSide = "right",
    device: str = "cuda",
    mode: str = "lightweight",
    score_thr: float = 0.5,
    min_kpts: int = 10,
    min_mean_score: float = 0.55,
    min_box_size: float = 24.0,
):
    """Return ``detector(frame) -> list[HandDet]`` (0 or 1 hand).

    Picks the best wholebody hand slot by score, then assigns ``hand`` for HaMeR
    (avoids L/R label flicker when only one hand is present).
    """
    if hand not in ("left", "right"):
        raise ValueError("hand must be 'left' or 'right'")
    is_right = hand == "right"

    # Load torch/CUDA before ORT sessions.
    import torch  # noqa: F401

    pose_model = Wholebody(mode=mode, backend="onnxruntime", device=device)
    _patch_sessions_cuda(pose_model)

    def detector(frame: np.ndarray) -> list[HandDet]:
        all_kpts, all_scores = pose_model(frame)
        best: HandDet | None = None
        for kpts, scores in zip(all_kpts, all_scores):
            for start in (_L_HAND, _R_HAND):
                hk = kpts[start : start + _NUM_HAND_KPTS]
                hs = scores[start : start + _NUM_HAND_KPTS]
                if hs[0] < score_thr:
                    continue
                valid = hs > score_thr
                n_valid = int(valid.sum())
                if n_valid < min_kpts:
                    continue
                mean_score = float(hs[valid].mean())
                if mean_score < min_mean_score:
                    continue

                pts = hk[valid]
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
                    kpts=hk.astype(np.float32),
                    scores=hs.astype(np.float32),
                    score=mean_score,
                )
                if best is None or det.score > best.score:
                    best = det
        return [best] if best is not None else []

    return detector


def draw_hands(frame_bgr: np.ndarray, hands: list[HandDet]) -> np.ndarray:
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
