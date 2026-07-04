"""Multi-hand RTMPose (TRT) + HaMeR (TRT) estimator."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from realtime_hamer.datasets.vitdet_dataset import ViTDetDataset
from realtime_hamer.detection import HandDet, create_detector, draw_hands
from realtime_hamer.engine_trt import TrtRunner
from realtime_hamer.models import load_hamer
from realtime_hamer.scripts.export_hamer_trt import ensure_hamer_engine
from realtime_hamer.utils import recursive_to


@dataclass
class HandMesh:
    """One reconstructed hand (fixed-root for 3D viz)."""

    vertices: np.ndarray  # (778, 3)
    faces: np.ndarray
    is_right: bool
    score: float


@dataclass
class FrameEstimate:
    """All hands in one frame."""

    hands: list[HandMesh] = field(default_factory=list)
    overlay_bgr: np.ndarray | None = None
    """RTMPose overlay, or None if ``draw_overlay=False``."""

    det_ms: float = 0.0
    hamer_ms: float = 0.0
    total_ms: float = 0.0


class HandPoseEstimator:
    """Detect and reconstruct every real hand in the frame."""

    def __init__(
        self,
        assets_dir: str | Path = "assets",
        checkpoint: str | Path | None = None,
        trt_cache: str | Path | None = None,
        device: str = "cuda:0",
        rescale_factor: float = 2.0,
        scale: float = 1.0,
        smooth: float = 0.65,
        build_trt: bool = True,
    ):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError("CUDA is required")

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

        # Per-side EMA state (at most one left + one right after ghost filtering).
        self._ema_state: dict[bool, dict] = {
            False: {"box": None, "global": None, "pose": None, "betas": None},
            True: {"box": None, "global": None, "pose": None, "betas": None},
        }

    def estimate(self, frame_bgr: np.ndarray, *, draw_overlay: bool = True) -> FrameEstimate:
        """Detect all hands and run HaMeR on each.

        Args:
            draw_overlay: If False, skip RTMPose drawing (saves time).
        """
        t0 = time.perf_counter()

        t1 = time.perf_counter()
        dets = self._detector(frame_bgr)
        det_ms = (time.perf_counter() - t1) * 1000.0

        overlay = draw_hands(frame_bgr, dets) if draw_overlay else None

        hands: list[HandMesh] = []
        hamer_ms = 0.0
        seen = {False: False, True: False}
        if dets:
            t2 = time.perf_counter()
            for det in dets:
                hands.append(self._reconstruct(frame_bgr, det))
                seen[det.is_right] = True
            hamer_ms = (time.perf_counter() - t2) * 1000.0

        for side in (False, True):
            if not seen[side]:
                self._ema_state[side] = {"box": None, "global": None, "pose": None, "betas": None}

        return FrameEstimate(
            hands=hands,
            overlay_bgr=overlay,
            det_ms=det_ms,
            hamer_ms=hamer_ms,
            total_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def _ema(self, prev, new):
        a = self.smooth
        if prev is None:
            return new
        return a * new + (1.0 - a) * prev

    @torch.inference_mode()
    def _reconstruct(self, frame_bgr: np.ndarray, hand: HandDet) -> HandMesh:
        is_right = bool(hand.is_right)
        state = self._ema_state[is_right]

        box = np.asarray(hand.box, dtype=np.float32)
        state["box"] = self._ema(state["box"], box)
        box = state["box"]

        right = np.asarray([1.0 if is_right else 0.0], dtype=np.float32)
        dataset = ViTDetDataset(
            self.model_cfg, frame_bgr, box[None, :], right, rescale_factor=self.rescale_factor
        )
        batch = recursive_to(
            next(iter(torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0))),
            self.device,
        )

        outs = self._trt({"img": batch["img"]})
        global_orient = outs["global_orient"].float()
        hand_pose = outs["hand_pose"].float()
        betas = outs["betas"].float()

        state["global"] = self._ema(state["global"], global_orient)
        state["pose"] = self._ema(state["pose"], hand_pose)
        state["betas"] = self._ema(state["betas"], betas)
        global_orient = state["global"]
        hand_pose = state["pose"]
        betas = state["betas"]

        mano_out = self.model.mano(
            global_orient=global_orient,
            hand_pose=hand_pose,
            betas=betas,
            pose2rot=False,
        )
        torch.cuda.synchronize()

        # Fixed-root: wrist at origin, cancel global rotation.
        R = global_orient[0, 0]
        wrist = mano_out.joints[0, 0]
        verts = (R.T @ (mano_out.vertices[0] - wrist).T).T.detach().cpu().numpy()
        if not is_right:
            verts[:, 0] *= -1.0
        verts[:, 1] *= -1.0
        if self.scale != 1.0:
            verts = verts * self.scale

        faces = self.faces_right if is_right else self.faces_left
        return HandMesh(
            vertices=verts.astype(np.float32),
            faces=faces,
            is_right=is_right,
            score=float(hand.score),
        )
