"""Single-hand RTMPose (TRT) + HaMeR (TRT) estimator."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from realtime_hamer.datasets.vitdet_dataset import ViTDetDataset
from realtime_hamer.detection import HandDet, create_detector, draw_hands, draw_mano_overlay
from realtime_hamer.engine_trt import TrtRunner
from realtime_hamer.models import load_hamer
from realtime_hamer.scripts.export_hamer_trt import ensure_hamer_engine
from realtime_hamer.utils import recursive_to
from realtime_hamer.utils.geometry import cam_crop_to_full

HandSide = Literal["left", "right"]


@dataclass
class HandEstimate:
    """One frame of hand estimation."""

    vertices: np.ndarray | None
    """(778, 3) fixed-root mesh for 3D viser, or None if not requested / no hand."""

    faces: np.ndarray
    overlay_bgr: np.ndarray | None
    """RTMPose overlay, or None if not requested."""

    mesh_overlay_bgr: np.ndarray | None
    """MANO mesh on video, or None if not requested."""

    detected: bool
    det_ms: float
    hamer_ms: float
    """TRT + MANO only (no 2D drawing)."""

    pose_overlay_ms: float
    mesh_overlay_ms: float
    total_ms: float


class HandPoseEstimator:
    """Detect one hand and reconstruct with TensorRT HaMeR + MANO."""

    def __init__(
        self,
        hand: HandSide = "right",
        assets_dir: str | Path = "assets",
        checkpoint: str | Path | None = None,
        trt_cache: str | Path | None = None,
        device: str = "cuda:0",
        rescale_factor: float = 2.0,
        scale: float = 1.0,
        smooth: float = 0.65,
        build_trt: bool = True,
    ):
        if hand not in ("left", "right"):
            raise ValueError("hand must be 'left' or 'right'")

        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError("CUDA is required")

        self.hand = hand
        self.is_right = hand == "right"
        self.rescale_factor = rescale_factor
        self.scale = scale
        self.smooth = float(np.clip(smooth, 0.05, 1.0))

        assets_dir = Path(assets_dir).resolve()
        cache_dir = Path(trt_cache) if trt_cache is not None else assets_dir / "trt_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        import realtime_hamer.configs as configs
        import realtime_hamer.models as models

        configs.CACHE_DIR_HAMER = str(assets_dir)
        models.DEFAULT_CHECKPOINT = f"{assets_dir}/hamer_ckpts/checkpoints/new_hamer_weights.ckpt"
        ckpt = Path(checkpoint) if checkpoint is not None else Path(models.DEFAULT_CHECKPOINT)

        self.model, self.model_cfg = load_hamer(str(ckpt))
        self.model = self.model.to(self.device).eval()
        self.faces_right = np.asarray(self.model.mano.faces, dtype=np.uint32)
        self.faces_left = self.faces_right[:, [0, 2, 1]].copy()
        self.faces = self.faces_right if self.is_right else self.faces_left

        if build_trt:
            engine_path = ensure_hamer_engine(assets_dir, cache_dir, fp16=True)
        else:
            engine_path = cache_dir / "hamer.engine"
            if not engine_path.is_file():
                raise FileNotFoundError(
                    f"Missing {engine_path}. Run: "
                    f"python -m realtime_hamer.scripts.export_hamer_trt --assets-dir {assets_dir}"
                )
        self._trt = TrtRunner(str(engine_path))

        self._detector = create_detector(
            hand=hand,
            device="cuda",
            mode="lightweight",
            trt_cache=cache_dir / "rtmpose",
        )

        self._ema_box: np.ndarray | None = None
        self._ema_hand_pose: torch.Tensor | None = None
        self._ema_betas: torch.Tensor | None = None
        self._ema_global: torch.Tensor | None = None
        self._ema_cam: torch.Tensor | None = None

    def estimate(
        self,
        frame_bgr: np.ndarray,
        *,
        draw_pose: bool = True,
        draw_mesh: bool = True,
        want_3d: bool = True,
    ) -> HandEstimate:
        """Run detection + HaMeR. Skip unused viz work when flags are False."""
        t0 = time.perf_counter()

        t1 = time.perf_counter()
        hands = self._detector(frame_bgr)
        det_ms = (time.perf_counter() - t1) * 1000.0

        pose_overlay = None
        pose_overlay_ms = 0.0
        if draw_pose:
            t_po = time.perf_counter()
            pose_overlay = draw_hands(frame_bgr, hands)
            pose_overlay_ms = (time.perf_counter() - t_po) * 1000.0

        verts_root = None
        mesh_overlay = None
        mesh_overlay_ms = 0.0
        hamer_ms = 0.0

        if hands:
            t2 = time.perf_counter()
            verts_root, mesh_overlay, mesh_overlay_ms = self._reconstruct(
                frame_bgr,
                hands[0],
                draw_mesh=draw_mesh,
                want_3d=want_3d,
            )
            hamer_ms = (time.perf_counter() - t2) * 1000.0 - mesh_overlay_ms
        else:
            self._ema_box = None
            self._ema_hand_pose = None
            self._ema_betas = None
            self._ema_global = None
            self._ema_cam = None

        return HandEstimate(
            vertices=verts_root,
            faces=self.faces,
            overlay_bgr=pose_overlay,
            mesh_overlay_bgr=mesh_overlay,
            detected=bool(hands),
            det_ms=det_ms,
            hamer_ms=max(hamer_ms, 0.0),
            pose_overlay_ms=pose_overlay_ms,
            mesh_overlay_ms=mesh_overlay_ms,
            total_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def _ema(self, prev, new):
        a = self.smooth
        if prev is None:
            return new
        return a * new + (1.0 - a) * prev

    @torch.inference_mode()
    def _reconstruct(
        self,
        frame_bgr: np.ndarray,
        hand: HandDet,
        *,
        draw_mesh: bool,
        want_3d: bool,
    ):
        box = np.asarray(hand.box, dtype=np.float32)
        self._ema_box = self._ema(self._ema_box, box)
        box = self._ema_box

        is_right = bool(hand.is_right)
        boxes = box[None, :]
        right = np.asarray([1.0 if is_right else 0.0], dtype=np.float32)
        dataset = ViTDetDataset(
            self.model_cfg, frame_bgr, boxes, right, rescale_factor=self.rescale_factor
        )
        batch = recursive_to(
            next(iter(torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0))),
            self.device,
        )

        outs = self._trt({"img": batch["img"]})
        global_orient = outs["global_orient"].float()
        hand_pose = outs["hand_pose"].float()
        betas = outs["betas"].float()
        pred_cam = outs["pred_cam"].float()

        self._ema_global = self._ema(self._ema_global, global_orient)
        self._ema_hand_pose = self._ema(self._ema_hand_pose, hand_pose)
        self._ema_betas = self._ema(self._ema_betas, betas)
        self._ema_cam = self._ema(self._ema_cam, pred_cam)
        global_orient = self._ema_global
        hand_pose = self._ema_hand_pose
        betas = self._ema_betas
        pred_cam = self._ema_cam

        mano_out = self.model.mano(
            global_orient=global_orient,
            hand_pose=hand_pose,
            betas=betas,
            pose2rot=False,
        )
        torch.cuda.synchronize()

        verts_mano = mano_out.vertices[0]
        wrist_mano = mano_out.joints[0, 0]
        faces = self.faces_right if is_right else self.faces_left
        self.faces = faces

        mesh_overlay = None
        mesh_overlay_ms = 0.0
        if draw_mesh:
            t_m = time.perf_counter()
            pred_cam_use = pred_cam.clone()
            verts_full = verts_mano.clone()
            if not is_right:
                verts_full[:, 0] *= -1.0
                pred_cam_use[:, 1] *= -1.0
            focal = (
                self.model_cfg.EXTRA.FOCAL_LENGTH
                / self.model_cfg.MODEL.IMAGE_SIZE
                * batch["img_size"].float().max()
            )
            cam_t = cam_crop_to_full(
                pred_cam_use,
                batch["box_center"].float(),
                batch["box_size"].float(),
                batch["img_size"].float(),
                focal,
            )
            verts_cam = (verts_full + cam_t[0]).detach().cpu().numpy()
            mesh_overlay = draw_mano_overlay(
                frame_bgr, verts_cam, faces, float(focal.item()), color=(36, 120, 143)
            )
            mesh_overlay_ms = (time.perf_counter() - t_m) * 1000.0

        verts_root = None
        if want_3d:
            R = global_orient[0, 0]
            verts_root = (R.T @ (verts_mano - wrist_mano).T).T.detach().cpu().numpy()
            if not is_right:
                verts_root[:, 0] *= -1.0
            verts_root[:, 1] *= -1.0
            if self.scale != 1.0:
                verts_root = verts_root * self.scale
            verts_root = verts_root.astype(np.float32)

        return verts_root, mesh_overlay, mesh_overlay_ms
