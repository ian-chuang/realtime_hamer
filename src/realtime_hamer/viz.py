"""Optional visualization / temporal helpers for ``HandMesh`` outputs.

``HandPoseEstimator`` only reconstructs hands. Use these on the consumer side
(e.g. ``test_hamer.py``) when you want overlays or EMA smoothing.
"""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import torch

from realtime_hamer.hand_pose_estimator import HandMesh

_HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)

# Continuous fields EMA'd per hand side (highest-score hand only).
_SMOOTH_FIELDS = (
    "vertices",
    "vertices_mano",
    "vertices_cam",
    "joints_cam",
    "cam_t",
    "global_orient",
    "hand_pose",
    "betas",
    "pred_cam",
    "box",
    "kpts",
)


def draw_keypoints_overlay(frame_bgr: np.ndarray, hands: list[HandMesh]) -> np.ndarray:
    """Draw RTMPose boxes + 2D keypoints for every hand onto a BGR image."""
    out = frame_bgr.copy()
    for hand in hands:
        color = (143, 120, 36) if hand.is_right else (241, 138, 133)
        x0, y0, x1, y1 = map(int, hand.box)
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
        label = f"{'R' if hand.is_right else 'L'} {hand.score:.2f}"
        cv2.putText(out, label, (x0, max(0, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        kpts, scores = hand.kpts, hand.kpt_scores
        for a, b in _HAND_EDGES:
            if scores[a] > 0.3 and scores[b] > 0.3:
                cv2.line(out, tuple(kpts[a].astype(int)), tuple(kpts[b].astype(int)), color, 2)
        for pt, sc in zip(kpts, scores):
            if sc > 0.3:
                cv2.circle(out, tuple(pt.astype(int)), 3, color, -1)
    return out


class HandMeshOverlayRenderer:
    """PyTorch3D MANO overlay for a list of ``HandMesh`` (hamer-demo path)."""

    def __init__(
        self,
        faces_right: np.ndarray,
        device: str | torch.device = "cuda:0",
    ):
        self.device = torch.device(device)
        self.faces_right = np.asarray(faces_right, dtype=np.int64)
        self._renderer = None

    def render(self, frame_bgr: np.ndarray, hands: list[HandMesh]) -> np.ndarray:
        """Composite all hands onto ``frame_bgr``. Returns a new BGR image."""
        if not hands:
            return frame_bgr.copy()

        from realtime_hamer.utils.pytorch3d_renderer import MeshPyTorch3DRenderer

        h, w = frame_bgr.shape[:2]
        focal = hands[0].focal
        if self._renderer is None:
            self._renderer = MeshPyTorch3DRenderer(
                faces=self.faces_right,
                device=self.device,
                render_res=(w, h),
                focal_length=focal,
            )
        else:
            self._renderer.maybe_resize((w, h), focal)

        verts_list = [hand.vertices_mano for hand in hands]
        cam_list = [hand.cam_t for hand in hands]
        rights = [1 if hand.is_right else 0 for hand in hands]
        rgba = self._renderer.render_rgba(verts_list, cam_list, is_right=rights)

        frame = frame_bgr.astype(np.float32)[:, :, ::-1] / 255.0
        a = rgba[:, :, 3:]
        out = frame * (1.0 - a) + rgba[:, :, :3] * a
        return (out[:, :, ::-1] * 255.0).clip(0, 255).astype(np.uint8)


class HandSmoother:
    """Per-side EMA on the highest-score hand of each side.

    Extra hands of the same side are passed through unsmoothed. When a side
    disappears, its EMA state is reset.
    """

    def __init__(self, alpha: float = 0.65):
        self.alpha = float(np.clip(alpha, 0.05, 1.0))
        self._prev: dict[bool, HandMesh | None] = {False: None, True: None}

    def __call__(self, hands: list[HandMesh]) -> list[HandMesh]:
        by_side: dict[bool, list[HandMesh]] = {False: [], True: []}
        for hand in hands:
            by_side[hand.is_right].append(hand)

        out: list[HandMesh] = []
        for side in (False, True):
            side_hands = sorted(by_side[side], key=lambda h: h.score, reverse=True)
            if not side_hands:
                self._prev[side] = None
                continue
            best = side_hands[0]
            prev = self._prev[side]
            smoothed = best if prev is None else self._blend(prev, best)
            self._prev[side] = smoothed
            out.append(smoothed)
            out.extend(side_hands[1:])
        return out

    def _blend(self, prev: HandMesh, new: HandMesh) -> HandMesh:
        a = self.alpha
        kwargs = {}
        for name in _SMOOTH_FIELDS:
            p = getattr(prev, name)
            n = getattr(new, name)
            kwargs[name] = (a * n + (1.0 - a) * p).astype(n.dtype, copy=False)
        # Non-smoothed fields (faces, scores, focal, …) come from ``new``.
        return replace(new, **kwargs)
