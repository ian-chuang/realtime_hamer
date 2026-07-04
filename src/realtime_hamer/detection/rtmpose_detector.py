"""Wholebody RTMPose hand detector with TensorRT engines (trtexec).

Uses balanced wholebody models (YOLOX-m + RTMW-x) — capable stand-in for
HaMeR's ViTDet/ViTPose, accelerated the same way as HaMeR (ONNX → trtexec).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from rtmlib import Wholebody

from realtime_hamer.engine_trt import TrtOrtSession, build_engine_from_onnx

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
    is_right: bool  # true side from RTMPose slot (for HaMeR flip)
    kpts: np.ndarray
    scores: np.ndarray
    score: float


def _patch_sessions_trt(pose_model: Wholebody, cache_dir: Path) -> None:
    """Replace ORT sessions with TRT engines when trtexec succeeds.

    Some detector ONNX graphs (YOLOX+NMS TopK) fail on TensorRT 11; those stay
    on ORT CUDA. Pose (RTMW) usually builds cleanly.
    """
    import onnxruntime as ort

    cache_dir.mkdir(parents=True, exist_ok=True)
    cuda_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        cuda_providers = ["CPUExecutionProvider"]

    # YOLOX ONNX embeds NMS with dynamic TopK — TRT 11 rejects / yields -1 dims.
    # Keep person detector on ORT CUDA; accelerate the heavier RTMW pose with TRT.
    det = pose_model.det_model
    det.session = ort.InferenceSession(det.onnx_model, providers=cuda_providers)
    det.backend = "onnxruntime"
    print("RTMPose det_model: ORT CUDA")

    pose = pose_model.pose_model
    pose_engine = cache_dir / "rtmw.engine"
    try:
        build_engine_from_onnx(Path(pose.onnx_model), pose_engine)
        pose.session = TrtOrtSession(pose_engine)
        print(f"RTMPose pose_model: TensorRT {pose_engine.name}")
    except Exception as exc:
        print(f"RTMPose pose_model: TRT failed ({exc}); using ORT CUDA")
        pose.session = ort.InferenceSession(pose.onnx_model, providers=cuda_providers)
    pose.backend = "onnxruntime"


def create_detector(
    hand: HandSide = "right",
    device: str = "cuda",
    mode: str = "lightweight",
    trt_cache: str | Path | None = None,
    score_thr: float = 0.5,
    min_kpts: int = 10,
    min_mean_score: float = 0.55,
    min_box_size: float = 24.0,
):
    """Return ``detector(frame) -> list[HandDet]`` (0 or 1 hand).

    Prefers the user-selected hand slot; falls back to the other if needed.
    ``HandDet.is_right`` is the *true* RTMPose side (required for correct HaMeR crops).
    """
    if hand not in ("left", "right"):
        raise ValueError("hand must be 'left' or 'right'")
    prefer_right = hand == "right"

    import torch  # noqa: F401  # load CUDA before TRT

    pose_model = Wholebody(mode=mode, backend="onnxruntime", device=device)
    if trt_cache is not None:
        _patch_sessions_trt(pose_model, Path(trt_cache))

    prefer_start = _R_HAND if prefer_right else _L_HAND
    other_start = _L_HAND if prefer_right else _R_HAND

    def _from_slot(kpts, scores, start) -> HandDet | None:
        is_right = start == _R_HAND
        hk = kpts[start : start + _NUM_HAND_KPTS]
        hs = scores[start : start + _NUM_HAND_KPTS]
        if hs[0] < score_thr:
            return None
        valid = hs > score_thr
        if int(valid.sum()) < min_kpts:
            return None
        mean_score = float(hs[valid].mean())
        if mean_score < min_mean_score:
            return None
        pts = hk[valid]
        x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
        x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
        w, h = x1 - x0, y1 - y0
        if w <= 1 or h <= 1:
            return None
        if w > h:
            d = (w - h) / 2
            y0, y1 = y0 - d, y1 + d
        else:
            d = (h - w) / 2
            x0, x1 = x0 - d, x1 + d
        if min(x1 - x0, y1 - y0) < min_box_size:
            return None
        return HandDet(
            box=[x0, y0, x1, y1],
            is_right=is_right,
            kpts=hk.astype(np.float32),
            scores=hs.astype(np.float32),
            score=mean_score,
        )

    def detector(frame: np.ndarray) -> list[HandDet]:
        all_kpts, all_scores = pose_model(frame)
        preferred: HandDet | None = None
        fallback: HandDet | None = None
        for kpts, scores in zip(all_kpts, all_scores):
            for start, bucket in ((prefer_start, "pref"), (other_start, "other")):
                det = _from_slot(kpts, scores, start)
                if det is None:
                    continue
                if bucket == "pref":
                    if preferred is None or det.score > preferred.score:
                        preferred = det
                else:
                    if fallback is None or det.score > fallback.score:
                        fallback = det
        # Prefer selected side; only use the other if preferred is missing and
        # fallback is clearly strong (avoids ghost opposite-hand slots).
        if preferred is not None and fallback is not None:
            # Prefer selected side unless the other is clearly stronger.
            return [preferred] if preferred.score + 0.05 >= fallback.score else [fallback]
        if preferred is not None:
            return [preferred]
        if fallback is not None:
            return [fallback]
        return []

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


def draw_mano_overlay(
    frame_bgr: np.ndarray,
    verts_cam: np.ndarray,
    faces: np.ndarray,
    focal: float,
    color=(36, 120, 143),
) -> np.ndarray:
    """Project MANO verts (OpenCV camera frame) and fill triangles on the image."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    cx, cy = w * 0.5, h * 0.5
    z = np.clip(verts_cam[:, 2], 1e-4, None)
    u = (focal * verts_cam[:, 0] / z + cx).astype(np.int32)
    v = (focal * verts_cam[:, 1] / z + cy).astype(np.int32)
    pts = np.stack([u, v], axis=1)

    # Painter's algorithm: far faces first.
    face_z = verts_cam[faces].mean(axis=1)[:, 2]
    order = np.argsort(-face_z)
    overlay = out.copy()
    fill = (int(color[2]), int(color[1]), int(color[0]))  # RGB -> BGR-ish for fill
    for fi in order[::8]:  # stride for speed
        tri = pts[faces[fi]]
        if np.any(tri[:, 0] < -w) or np.any(tri[:, 0] > 2 * w):
            continue
        if np.any(tri[:, 1] < -h) or np.any(tri[:, 1] > 2 * h):
            continue
        cv2.fillConvexPoly(overlay, tri, fill)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
    return out
