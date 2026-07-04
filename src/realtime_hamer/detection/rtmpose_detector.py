"""Wholebody RTMPose hand detector (YOLOX CUDA + RTMW TensorRT)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from rtmlib import Wholebody

from realtime_hamer.engine_trt import TrtOrtSession, build_engine_from_onnx

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


def _box_iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _patch_sessions_trt(pose_model: Wholebody, cache_dir: Path) -> None:
    """RTMW → TRT; YOLOX stays on ORT CUDA (NMS TopK breaks TRT 11)."""
    import onnxruntime as ort

    cache_dir.mkdir(parents=True, exist_ok=True)
    cuda_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        cuda_providers = ["CPUExecutionProvider"]

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
    device: str = "cuda",
    mode: str = "lightweight",
    trt_cache: str | Path | None = None,
    score_thr: float = 0.5,
    min_kpts: int = 10,
    min_mean_score: float = 0.55,
    min_box_size: float = 24.0,
    relative_score_thr: float = 0.75,
    max_iou: float = 0.35,
):
    """Return ``detector(frame) -> list[HandDet]`` for all real hands in the frame.

    Wholebody always fills L/R slots; ghost opposite-hand slots are dropped when
    they are much weaker or heavily overlap the stronger hand.
    """
    import torch  # noqa: F401

    pose_model = Wholebody(mode=mode, backend="onnxruntime", device=device)
    if trt_cache is not None:
        _patch_sessions_trt(pose_model, Path(trt_cache))

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
        # Best left and best right across people.
        best: dict[bool, HandDet | None] = {False: None, True: None}
        for kpts, scores in zip(all_kpts, all_scores):
            for start in (_L_HAND, _R_HAND):
                det = _from_slot(kpts, scores, start)
                if det is None:
                    continue
                prev = best[det.is_right]
                if prev is None or det.score > prev.score:
                    best[det.is_right] = det

        left, right = best[False], best[True]
        if left is None and right is None:
            return []
        if left is None:
            return [right]
        if right is None:
            return [left]

        # Both slots fired: drop ghost opposite-hand (weaker and/or overlapping).
        strong, weak = (right, left) if right.score >= left.score else (left, right)
        if weak.score < relative_score_thr * strong.score:
            return [strong]
        if _box_iou(left.box, right.box) >= max_iou:
            return [strong]
        return [left, right]

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
