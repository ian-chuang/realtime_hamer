"""Realtime HaMeR: RTMPose (onnxruntime-gpu) + TensorRT HaMeR + viser.

Video loops like a webcam. Zero / one / two hands are all fine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import tyro
import viser

from realtime_hamer.datasets.vitdet_dataset import ViTDetDataset
from realtime_hamer.detection import HandDet, create_detector, draw_hands
from realtime_hamer.models import DEFAULT_CHECKPOINT, load_hamer
from realtime_hamer.trt_runtime import import_torch_tensorrt, preload_gpu_libs
from realtime_hamer.utils import recursive_to
from realtime_hamer.utils.geometry import cam_crop_to_full

LEFT_COLOR = (133, 138, 241)
RIGHT_COLOR = (36, 120, 143)
# Fixed-root display slots in viser (y-up), meters.
LEFT_SLOT = np.array([-0.12, 0.0, 0.35], dtype=np.float32)
RIGHT_SLOT = np.array([0.12, 0.0, 0.35], dtype=np.float32)


@dataclass
class Args:
    video: Path | None = None
    """Video path (loops forever). If unset, use --camera."""

    camera: int = 0
    """Webcam index when --video is not set."""

    checkpoint: Path = Path(DEFAULT_CHECKPOINT)
    """HaMeR checkpoint."""

    trt_cache: Path = Path("assets/trt_cache")
    """Cached TensorRT modules (written on first run)."""

    max_hands: int = 2
    """Max hands per frame (TRT batch upper bound)."""

    rescale_factor: float = 2.0
    """Hand-crop padding for HaMeR."""

    port: int = 8080
    """Viser port."""


def open_capture(video: Path | None, camera: int) -> tuple[cv2.VideoCapture, bool]:
    cap = cv2.VideoCapture(str(video) if video is not None else camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {'video ' + str(video) if video else f'camera {camera}'}")
    return cap, video is not None


def read_frame(cap: cv2.VideoCapture, loop: bool) -> np.ndarray | None:
    ok, frame = cap.read()
    if ok:
        return frame
    if not loop:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame = cap.read()
    return frame if ok else None


def compile_or_load_trt(model, device: torch.device, cache_dir: Path, max_hands: int):
    """Compile backbone + transformer once; reload from disk on later runs."""
    torch_tensorrt = import_torch_tensorrt()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"hamer_trt_bs{max_hands}.pt"

    if cache_path.is_file():
        print(f"Loading TensorRT cache: {cache_path}")
        blob = torch.load(cache_path, map_location=device, weights_only=False)
        model.backbone = blob["backbone"]
        model.mano_head.transformer = blob["transformer"]
        return model

    print("Compiling TensorRT (first run only)...")
    opt = min(2, max_hands)
    model.backbone = torch_tensorrt.compile(
        model.backbone,
        inputs=[
            torch_tensorrt.Input(
                min_shape=(1, 3, 256, 192),
                opt_shape=(opt, 3, 256, 192),
                max_shape=(max_hands, 3, 256, 192),
            )
        ],
        enabled_precisions={torch.float16},
        device=device,
    )
    model.mano_head.transformer = torch_tensorrt.compile(
        model.mano_head.transformer,
        inputs=[
            torch_tensorrt.Input(
                min_shape=(1, 1, 1),
                opt_shape=(opt, 1, 1),
                max_shape=(max_hands, 1, 1),
            ),
            torch_tensorrt.Input(
                min_shape=(1, 192, 1280),
                opt_shape=(opt, 192, 1280),
                max_shape=(max_hands, 192, 1280),
            ),
        ],
        enabled_precisions={torch.float16},
        device=device,
    )
    print(f"Saving TensorRT cache: {cache_path}")
    torch.save(
        {"backbone": model.backbone, "transformer": model.mano_head.transformer},
        cache_path,
    )
    return model


def run_hamer(
    model,
    model_cfg,
    frame_bgr,
    hands: list[HandDet],
    device,
    max_hands: int,
    rescale_factor: float,
    focal: float,
    depth_scale: float,
    fix_root: bool,
):
    """Return list of (verts_viser, is_right)."""
    boxes = np.stack([h.box for h in hands])
    is_right = np.array([h.is_right for h in hands], dtype=np.float32)
    dataset = ViTDetDataset(model_cfg, frame_bgr, boxes, is_right, rescale_factor=rescale_factor)
    batch = recursive_to(
        next(iter(torch.utils.data.DataLoader(dataset, batch_size=max_hands, num_workers=0))),
        device,
    )
    with torch.no_grad():
        out = model(batch)
    torch.cuda.synchronize()

    right = batch["right"]
    results = []

    if fix_root:
        # Finger articulation only: identity wrist orient, wrist at a fixed slot.
        B = right.shape[0]
        eye = torch.eye(3, device=device, dtype=out["pred_mano_params"]["hand_pose"].dtype)
        eye = eye.view(1, 1, 3, 3).expand(B, 1, 3, 3)
        mano_out = model.mano(
            global_orient=eye,
            hand_pose=out["pred_mano_params"]["hand_pose"].float(),
            betas=out["pred_mano_params"]["betas"].float(),
            pose2rot=False,
        )
        verts = mano_out.vertices.clone()
        wrist = mano_out.joints[:, 0:1, :]
        verts = verts - wrist
        for i in range(B):
            is_r = bool(right[i].item() > 0.5)
            v = verts[i].detach().cpu().numpy()
            v[:, 0] *= 1.0 if is_r else -1.0
            v = to_viser(v)
            v = v + (RIGHT_SLOT if is_r else LEFT_SLOT)
            results.append((v.astype(np.float32), is_r))
        return results

    # Free mode: place hands in camera frame with adjustable intrinsics/depth.
    pred_cam = out["pred_cam"].clone()
    pred_cam[:, 1] = (2 * right - 1) * pred_cam[:, 1]
    cam_t = cam_crop_to_full(
        pred_cam,
        batch["box_center"].float(),
        batch["box_size"].float(),
        batch["img_size"].float(),
        focal,
    )
    cam_t = cam_t * depth_scale

    for i in range(batch["img"].shape[0]):
        verts = out["pred_vertices"][i].detach().clone()
        is_r = bool(right[i].item() > 0.5)
        verts[:, 0] *= 1.0 if is_r else -1.0
        verts_cam = (verts + cam_t[i]).cpu().numpy()
        results.append((to_viser(verts_cam).astype(np.float32), is_r))
    return results


def to_viser(pts: np.ndarray) -> np.ndarray:
    """OpenCV cam (y-down) -> viser (y-up)."""
    out = pts.copy()
    out[:, 1] *= -1.0
    return out


def default_focal(width: int, height: int) -> float:
    """Reasonable pinhole focal for a hand ~10–30 cm from a webcam (~50–60° FOV)."""
    return float(max(width, height))


def main(args: Args) -> None:
    preload_gpu_libs()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA required")

    model, model_cfg = load_hamer(str(args.checkpoint))
    model = model.to(device).eval()
    model = compile_or_load_trt(model, device, args.trt_cache, args.max_hands)

    faces_r = np.asarray(model.mano.faces, dtype=np.uint32)
    faces_l = faces_r[:, [0, 2, 1]].copy()
    detector = create_detector(device="cuda")
    cap, loop = open_capture(args.video, args.camera)

    # Peek one frame for default focal.
    frame0 = read_frame(cap, loop)
    if frame0 is None:
        raise RuntimeError("Empty video / camera")
    h0, w0 = frame0.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    server = viser.ViserServer(port=args.port)
    server.scene.add_frame("/world", axes_length=0.08, axes_radius=0.003)
    with server.gui.add_folder("Status"):
        gui_fps = server.gui.add_text("FPS", initial_value="—", disabled=True)
        gui_hands = server.gui.add_text("Hands", initial_value="0", disabled=True)
    with server.gui.add_folder("Display"):
        gui_fix_root = server.gui.add_checkbox(
            "Fix hand root",
            initial_value=True,
            hint="Pin wrist pose/position; show finger articulation only.",
        )
        gui_focal = server.gui.add_slider(
            "Focal (px)", min=200.0, max=3000.0, step=10.0, initial_value=default_focal(w0, h0)
        )
        gui_depth = server.gui.add_slider(
            "Depth scale", min=0.1, max=3.0, step=0.05, initial_value=1.0
        )
    gui_img = server.gui.add_image(
        np.zeros((64, 64, 3), dtype=np.uint8), label="RTMPose"
    )
    handles: dict[str, object | None] = {"left": None, "right": None}

    print(f"Viser http://localhost:{args.port}  (Ctrl+C to stop)")

    try:
        while True:
            t0 = time.perf_counter()
            frame = read_frame(cap, loop)
            if frame is None:
                break

            t1 = time.perf_counter()
            hands = detector(frame)
            det_ms = (time.perf_counter() - t1) * 1000

            overlay = draw_hands(frame, hands)
            gui_img.image = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

            seen = {"left": False, "right": False}
            hamer_ms = 0.0
            if hands:
                t2 = time.perf_counter()
                meshes = run_hamer(
                    model,
                    model_cfg,
                    frame,
                    hands,
                    device,
                    args.max_hands,
                    args.rescale_factor,
                    float(gui_focal.value),
                    float(gui_depth.value),
                    bool(gui_fix_root.value),
                )
                hamer_ms = (time.perf_counter() - t2) * 1000

                for verts, is_r in meshes:
                    name = "right" if is_r else "left"
                    seen[name] = True
                    color = RIGHT_COLOR if is_r else LEFT_COLOR
                    faces = faces_r if is_r else faces_l
                    h = handles[name]
                    if h is None:
                        handles[name] = server.scene.add_mesh_simple(
                            f"/hands/{name}",
                            vertices=verts,
                            faces=faces,
                            color=color,
                            side="double",
                        )
                    else:
                        h.vertices = verts
                        h.visible = True

            for name, h in handles.items():
                if h is not None and not seen[name]:
                    h.visible = False

            fps = 1.0 / max(time.perf_counter() - t0, 1e-6)
            gui_fps.value = f"{fps:.1f}   det {det_ms:.1f} ms   hamer {hamer_ms:.1f} ms"
            gui_hands.value = ", ".join(
                (["L"] if seen["left"] else []) + (["R"] if seen["right"] else [])
            ) or "none"
            # Disable free-mode sliders when root is fixed.
            gui_focal.disabled = bool(gui_fix_root.value)
            gui_depth.disabled = bool(gui_fix_root.value)
            print(
                f"fps={fps:5.1f}  det={det_ms:5.1f}ms  hamer={hamer_ms:5.1f}ms  "
                f"hands={gui_hands.value}",
                end="\r",
            )
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()


if __name__ == "__main__":
    main(tyro.cli(Args))
