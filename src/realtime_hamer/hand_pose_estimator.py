"""Multi-hand RTMPose (TRT) + HaMeR (TRT) reconstruction."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from realtime_hamer.datasets.vitdet_dataset import prepare_hand_batch
from realtime_hamer.detection import HandDet, create_detector
from realtime_hamer.engine_trt import TrtRunner
from realtime_hamer.models import load_hamer
from realtime_hamer.scripts.export_hamer_trt import ensure_hamer_engine
from realtime_hamer.utils.geometry import cam_crop_to_full


@dataclass
class HandMesh:
    """Full reconstruction for one hand."""

    # Fixed-root mesh for 3D (wrist at origin, global rotation cancelled).
    vertices: np.ndarray  # (778, 3)
    faces: np.ndarray  # (F, 3)
    # MANO verts with left-hand x-flip applied (HaMeR overlay convention).
    vertices_mano: np.ndarray  # (778, 3)
    # Camera-frame mesh / joints (OpenCV coords) for overlay / retargeting.
    vertices_cam: np.ndarray  # (778, 3)
    joints_cam: np.ndarray  # (21, 3)
    cam_t: np.ndarray  # (3,)
    focal: float
    # MANO parameters (rotation matrices / betas).
    global_orient: np.ndarray  # (1, 3, 3)
    hand_pose: np.ndarray  # (15, 3, 3)
    betas: np.ndarray  # (10,)
    pred_cam: np.ndarray  # (3,) weak-perspective in crop
    box: np.ndarray  # (4,) xyxy
    # RTMPose 2D keypoints (for optional overlay helpers).
    kpts: np.ndarray  # (21, 2)
    kpt_scores: np.ndarray  # (21,)
    is_right: bool
    score: float


@dataclass
class FrameEstimate:
    """All hands reconstructed in one frame."""

    hands: list[HandMesh] = field(default_factory=list)
    det_ms: float = 0.0
    trt_ms: float = 0.0
    """TensorRT HaMeR net only (sum over hands)."""

    hamer_ms: float = 0.0
    """Crop + TRT + MANO for all hands."""

    total_ms: float = 0.0


class HandPoseEstimator:
    """Detect and reconstruct every real hand in the frame.

    No smoothing or rendering — use ``realtime_hamer.viz`` helpers for that.
    """

    def __init__(
        self,
        assets_dir: str | Path = "assets",
        checkpoint: str | Path | None = None,
        trt_cache: str | Path | None = None,
        device: str = "cuda:0",
        rescale_factor: float = 2.0,
        scale: float = 1.0,
        build_trt: bool = True,
    ):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError("CUDA is required")

        self.rescale_factor = rescale_factor
        self.scale = scale

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
        self.faces_right = np.asarray(self.model.mano.faces, dtype=np.int64)
        self.faces_left = self.faces_right[:, [0, 2, 1]].copy()

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
            device="cuda",
            mode="lightweight",
            trt_cache=cache_dir / "rtmpose",
        )

    def estimate(self, frame_bgr: np.ndarray) -> FrameEstimate:
        """Detect all hands and run HaMeR on each. Returns full reconstructions only."""
        t0 = time.perf_counter()

        t1 = time.perf_counter()
        dets = self._detector(frame_bgr)
        det_ms = (time.perf_counter() - t1) * 1000.0

        hands: list[HandMesh] = []
        hamer_ms = 0.0
        trt_ms = 0.0
        if dets:
            t2 = time.perf_counter()
            trt_acc = 0.0
            for det in dets:
                hand, dt = self._reconstruct(frame_bgr, det)
                hands.append(hand)
                trt_acc += dt
            trt_ms = trt_acc
            hamer_ms = (time.perf_counter() - t2) * 1000.0

        return FrameEstimate(
            hands=hands,
            det_ms=det_ms,
            trt_ms=trt_ms,
            hamer_ms=hamer_ms,
            total_ms=(time.perf_counter() - t0) * 1000.0,
        )

    @torch.inference_mode()
    def _reconstruct(self, frame_bgr: np.ndarray, hand: HandDet) -> tuple[HandMesh, float]:
        is_right = bool(hand.is_right)
        box = np.asarray(hand.box, dtype=np.float32)

        batch = prepare_hand_batch(
            self.model_cfg, frame_bgr, box, is_right, self.device, self.rescale_factor
        )

        t_trt = time.perf_counter()
        outs = self._trt({"img": batch["img"]})
        torch.cuda.current_stream().wait_stream(self._trt.stream)
        torch.cuda.synchronize()
        trt_ms = (time.perf_counter() - t_trt) * 1000.0

        # Clone: TrtRunner reuses output buffers across calls.
        global_orient = outs["global_orient"].float().clone()
        hand_pose = outs["hand_pose"].float().clone()
        betas = outs["betas"].float().clone()
        pred_cam = outs["pred_cam"].float().clone()

        mano_out = self.model.mano(
            global_orient=global_orient,
            hand_pose=hand_pose,
            betas=betas,
            pose2rot=False,
        )

        verts_mano = mano_out.vertices[0]
        joints_mano = mano_out.joints[0]
        wrist = joints_mano[0]
        # Match HaMeR demo: flip weak-perspective tx and mesh x for left hands.
        pred_cam_use = pred_cam.clone()
        verts_full = verts_mano.clone()
        joints_full = joints_mano.clone()
        if not is_right:
            verts_full[:, 0] *= -1.0
            joints_full[:, 0] *= -1.0
            pred_cam_use[0, 1] *= -1.0

        focal = (
            self.model_cfg.EXTRA.FOCAL_LENGTH
            / self.model_cfg.MODEL.IMAGE_SIZE
            * batch["img_size"][0].max()
        )
        cam_t = cam_crop_to_full(
            pred_cam_use,
            batch["box_center"],
            batch["box_size"],
            batch["img_size"],
            focal,
        )[0]

        verts_cam = (verts_full + cam_t).detach().cpu().numpy()
        joints_cam = (joints_full + cam_t).detach().cpu().numpy()
        cam_t_np = cam_t.detach().cpu().numpy()
        verts_mano_np = verts_full.detach().cpu().numpy()

        # Fixed-root 3D: wrist at origin, cancel global rotation.
        R = global_orient[0, 0]
        verts_root = (R.T @ (verts_mano - wrist).T).T.detach().cpu().numpy()
        if not is_right:
            verts_root[:, 0] *= -1.0
        verts_root[:, 1] *= -1.0
        if self.scale != 1.0:
            verts_root = verts_root * self.scale

        faces = self.faces_right if is_right else self.faces_left
        mesh = HandMesh(
            vertices=verts_root.astype(np.float32),
            faces=faces,
            vertices_mano=verts_mano_np.astype(np.float32),
            vertices_cam=verts_cam.astype(np.float32),
            joints_cam=joints_cam.astype(np.float32),
            cam_t=cam_t_np.astype(np.float32),
            focal=float(focal.item()),
            global_orient=global_orient[0].detach().cpu().numpy().astype(np.float32),
            hand_pose=hand_pose[0].detach().cpu().numpy().astype(np.float32),
            betas=betas[0].detach().cpu().numpy().astype(np.float32),
            pred_cam=pred_cam[0].detach().cpu().numpy().astype(np.float32),
            box=box.copy(),
            kpts=np.asarray(hand.kpts, dtype=np.float32),
            kpt_scores=np.asarray(hand.scores, dtype=np.float32),
            is_right=is_right,
            score=float(hand.score),
        )
        return mesh, trt_ms
