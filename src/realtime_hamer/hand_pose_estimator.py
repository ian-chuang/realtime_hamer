"""Single-hand RTMPose + HaMeR estimator (fixed-root mesh, FP16 CUDA)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from realtime_hamer.datasets.vitdet_dataset import ViTDetDataset
from realtime_hamer.detection import HandDet, create_detector, draw_hands
from realtime_hamer.models import load_hamer
from realtime_hamer.utils import recursive_to

HandSide = Literal["left", "right"]


@dataclass
class HandEstimate:
    """One frame of hand estimation."""

    vertices: np.ndarray | None
    """(778, 3) mesh at the origin (wrist-centered, identity global orient), or None."""

    faces: np.ndarray
    """(F, 3) triangle indices for ``vertices``."""

    overlay_bgr: np.ndarray
    """Input frame with RTMPose keypoints / box drawn."""

    detected: bool
    det_ms: float
    hamer_ms: float
    total_ms: float


class HandPoseEstimator:
    """Detect one hand and reconstruct a fixed-root MANO mesh."""

    def __init__(
        self,
        hand: HandSide = "right",
        assets_dir: str | Path = "assets",
        checkpoint: str | Path | None = None,
        device: str = "cuda:0",
        rescale_factor: float = 2.0,
        scale: float = 1.0,
    ):
        """
        Args:
            hand: Which hand label to assign (best detection is always used).
            assets_dir: Directory with ``hamer_ckpts/`` and ``data/mano/``.
            checkpoint: HaMeR ckpt path.
            device: Torch device (CUDA required).
            rescale_factor: HaMeR crop padding.
            scale: Uniform scale on the fixed-root mesh.
        """
        if hand not in ("left", "right"):
            raise ValueError("hand must be 'left' or 'right'")

        # Load torch/CUDA before onnxruntime sessions are created.
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError("CUDA is required for HandPoseEstimator")

        self.hand: HandSide = hand
        self.is_right = hand == "right"
        self.rescale_factor = rescale_factor
        self.scale = scale

        assets_dir = Path(assets_dir).resolve()
        import realtime_hamer.configs as configs
        import realtime_hamer.models as models

        configs.CACHE_DIR_HAMER = str(assets_dir)
        models.DEFAULT_CHECKPOINT = f"{assets_dir}/hamer_ckpts/checkpoints/new_hamer_weights.ckpt"
        ckpt = Path(checkpoint) if checkpoint is not None else Path(models.DEFAULT_CHECKPOINT)

        self.model, self.model_cfg = load_hamer(str(ckpt))
        self.model = self.model.to(self.device).eval()

        faces = np.asarray(self.model.mano.faces, dtype=np.uint32)
        self.faces = faces if self.is_right else faces[:, [0, 2, 1]].copy()
        self._detector = create_detector(hand=hand, device="cuda")

    @torch.inference_mode()
    def estimate(self, frame_bgr: np.ndarray) -> HandEstimate:
        """Run detection + HaMeR on one BGR frame."""
        t0 = time.perf_counter()

        t1 = time.perf_counter()
        hands = self._detector(frame_bgr)
        det_ms = (time.perf_counter() - t1) * 1000.0
        overlay = draw_hands(frame_bgr, hands)

        verts = None
        hamer_ms = 0.0
        if hands:
            t2 = time.perf_counter()
            verts = self._reconstruct(frame_bgr, hands[0])
            hamer_ms = (time.perf_counter() - t2) * 1000.0

        return HandEstimate(
            vertices=verts,
            faces=self.faces,
            overlay_bgr=overlay,
            detected=verts is not None,
            det_ms=det_ms,
            hamer_ms=hamer_ms,
            total_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def _reconstruct(self, frame_bgr: np.ndarray, hand: HandDet) -> np.ndarray:
        """Fixed-root mesh: identity wrist orient, wrist at origin."""
        boxes = np.asarray([hand.box], dtype=np.float32)
        is_right = np.asarray([1.0 if self.is_right else 0.0], dtype=np.float32)
        dataset = ViTDetDataset(
            self.model_cfg, frame_bgr, boxes, is_right, rescale_factor=self.rescale_factor
        )
        batch = recursive_to(
            next(iter(torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0))),
            self.device,
        )
        with torch.autocast("cuda", dtype=torch.float16):
            out = self.model(batch)
        torch.cuda.synchronize()

        eye = torch.eye(3, device=self.device, dtype=torch.float32).view(1, 1, 3, 3)
        mano_out = self.model.mano(
            global_orient=eye,
            hand_pose=out["pred_mano_params"]["hand_pose"].float(),
            betas=out["pred_mano_params"]["betas"].float(),
            pose2rot=False,
        )
        verts = mano_out.vertices[0] - mano_out.joints[0, 0]
        verts = verts.detach().cpu().numpy()
        if not self.is_right:
            verts[:, 0] *= -1.0
        verts[:, 1] *= -1.0  # OpenCV y-down -> viser y-up
        if self.scale != 1.0:
            verts = verts * self.scale
        return verts.astype(np.float32)
