"""RTMPose wholebody hand detector (onnxruntime-gpu).

Wholebody always predicts left+right slots; we only keep hands with enough
confident keypoints so a single visible hand does not spawn a ghost pair.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from rtmlib import Wholebody

# COCO-WholeBody hand keypoint indices.
_NUM_HAND_KPTS = 21
_L_HAND = 91
_R_HAND = 91 + _NUM_HAND_KPTS

# Skeleton edges within one hand (wrist=0, fingers along 1..20).
_HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)


@dataclass
class HandDet:
    box: list[float]  # x0, y0, x1, y1
    is_right: bool
    kpts: np.ndarray  # (21, 2)
    scores: np.ndarray  # (21,)
    score: float  # mean of valid keypoint scores


def create_detector(
    mode: str = "lightweight",
    device: str = "cuda",
    score_thr: float = 0.5,
    min_kpts: int = 10,
    min_mean_score: float = 0.55,
    min_box_size: float = 24.0,
    relative_score_thr: float = 0.7,
):
    """Return ``detector(frame) -> list[HandDet]`` (0–2 hands).

    Wholebody always fills L/R slots; ghost hands usually have fewer / weaker
    keypoints than a real hand, so we filter hard and drop a weak second hand.
    """
    pose_model = Wholebody(mode=mode, backend="onnxruntime", device=device)

    def _hand_from_kpts(kpts: np.ndarray, scores: np.ndarray, is_right: bool) -> HandDet | None:
        # Wrist must be visible — ghosts often lack a confident wrist.
        if scores[0] < score_thr:
            return None
        valid = scores > score_thr
        n_valid = int(valid.sum())
        if n_valid < min_kpts:
            return None
        mean_score = float(scores[valid].mean())
        if mean_score < min_mean_score:
            return None

        pts = kpts[valid]
        x0, y0 = pts[:, 0].min(), pts[:, 1].min()
        x1, y1 = pts[:, 0].max(), pts[:, 1].max()
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
            box=[float(x0), float(y0), float(x1), float(y1)],
            is_right=is_right,
            kpts=kpts.astype(np.float32),
            scores=scores.astype(np.float32),
            score=mean_score,
        )

    def detector(frame: np.ndarray) -> list[HandDet]:
        all_kpts, all_scores = pose_model(frame)
        best: dict[bool, HandDet | None] = {False: None, True: None}
        for kpts, scores in zip(all_kpts, all_scores):
            for is_right, start in ((False, _L_HAND), (True, _R_HAND)):
                hand = _hand_from_kpts(
                    kpts[start : start + _NUM_HAND_KPTS],
                    scores[start : start + _NUM_HAND_KPTS],
                    is_right,
                )
                if hand is None:
                    continue
                prev = best[is_right]
                if prev is None or hand.score > prev.score:
                    best[is_right] = hand

        hands = [h for h in (best[False], best[True]) if h is not None]
        # If one hand is clearly weaker, treat it as a ghost.
        if len(hands) == 2:
            top = max(h.score for h in hands)
            hands = [h for h in hands if h.score >= relative_score_thr * top]
        return hands

    return detector


def draw_hands(frame_bgr: np.ndarray, hands: list[HandDet]) -> np.ndarray:
    """Overlay hand keypoints, bones, and boxes on a BGR frame."""
    out = frame_bgr.copy()
    for hand in hands:
        color = (143, 120, 36) if hand.is_right else (241, 138, 133)  # BGR
        x0, y0, x1, y1 = map(int, hand.box)
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
        label = f"{'R' if hand.is_right else 'L'} {hand.score:.2f}"
        cv2.putText(out, label, (x0, max(0, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        kpts, scores = hand.kpts, hand.scores
        for a, b in _HAND_EDGES:
            if scores[a] > 0.3 and scores[b] > 0.3:
                pa = tuple(kpts[a].astype(int))
                pb = tuple(kpts[b].astype(int))
                cv2.line(out, pa, pb, color, 2)
        for i, (pt, sc) in enumerate(zip(kpts, scores)):
            if sc > 0.3:
                cv2.circle(out, tuple(pt.astype(int)), 3, color, -1)
    return out
