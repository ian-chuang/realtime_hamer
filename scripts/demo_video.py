"""Realtime HaMeR demo: RTMPose detection + TensorRT HaMeR + PyTorch3D render.

Supports a video file or a live webcam/camera index.
Adapted from https://github.com/ATAboukhadra/hamer-demo
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from realtime_hamer.datasets.vitdet_dataset import ViTDetDataset
from realtime_hamer.detection import create_detector
from realtime_hamer.models import DEFAULT_CHECKPOINT, load_hamer
from realtime_hamer.trt_runtime import import_torch_tensorrt, preload_gpu_libs
from realtime_hamer.utils import recursive_to
from realtime_hamer.utils.pytorch3d_renderer import MeshPyTorch3DRenderer
from realtime_hamer.utils.renderer import cam_crop_to_full

preload_gpu_libs()
torch_tensorrt = import_torch_tensorrt()

def compile_hamer_tensorrt(model, device, max_batch_size: int):
    """Compile ViT backbone and MANO transformer decoder with TensorRT FP16."""
    print("Compiling HaMeR backbone with TensorRT (first run can take a few minutes)...")
    backbone_input = [
        torch_tensorrt.Input(
            min_shape=(1, 3, 256, 192),
            opt_shape=(2, 3, 256, 192),
            max_shape=(max_batch_size, 3, 256, 192),
        )
    ]
    model.backbone = torch_tensorrt.compile(
        model.backbone,
        inputs=backbone_input,
        enabled_precisions={torch.float16},
        device=device,
    )

    print("Compiling HaMeR transformer with TensorRT...")
    model.mano_head.transformer = torch_tensorrt.compile(
        model.mano_head.transformer,
        inputs=[
            torch_tensorrt.Input(
                min_shape=(1, 1, 1),
                opt_shape=(2, 1, 1),
                max_shape=(max_batch_size, 1, 1),
            ),
            torch_tensorrt.Input(
                min_shape=(1, 192, 1280),
                opt_shape=(2, 192, 1280),
                max_shape=(max_batch_size, 192, 1280),
            ),
        ],
        enabled_precisions={torch.float16},
        device=device,
    )
    print("TensorRT compilation done.")
    return model


def process_frame(
    img_cv2,
    model,
    model_cfg,
    detector,
    device,
    renderer,
    max_batch_size: int,
    rescale_factor: float,
    flip_display: bool,
):
    timer = time.time()
    bboxes, is_right = detector(img_cv2)
    det_ms = (time.time() - timer) * 1000

    if len(bboxes) == 0:
        output_img = cv2.flip(img_cv2, 1) if flip_display else img_cv2
        fps = 1.0 / max(time.time() - timer, 1e-6)
        return output_img, renderer, fps, det_ms, 0.0, 0.0

    boxes = np.stack(bboxes)
    right = np.stack(is_right)
    dataset = ViTDetDataset(
        model_cfg, img_cv2, boxes, right, rescale_factor=rescale_factor
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=max_batch_size, shuffle=False, num_workers=0
    )

    batch = recursive_to(next(iter(dataloader)), device)
    t_model = time.time()
    with torch.no_grad():
        out = model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    model_ms = (time.time() - t_model) * 1000

    multiplier = 2 * batch["right"] - 1
    pred_cam = out["pred_cam"]
    pred_cam[:, 1] = multiplier * pred_cam[:, 1]
    box_center = batch["box_center"].float()
    box_size = batch["box_size"].float()
    img_size = batch["img_size"].float()
    scaled_focal_length = (
        model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
    )
    pred_cam_t_full = (
        cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length)
        .detach()
        .cpu()
        .numpy()
    )

    if renderer is None:
        renderer = MeshPyTorch3DRenderer(
            model_cfg,
            model.mano.faces,
            device,
            render_res=img_size[0],
            focal_length=scaled_focal_length,
        )

    all_verts = []
    all_cam_t = []
    all_right = []
    batch_size = batch["img"].shape[0]
    for n in range(batch_size):
        verts = out["pred_vertices"][n].detach().cpu().numpy()
        is_right_n = batch["right"][n].cpu().numpy()
        verts[:, 0] = (2 * is_right_n - 1) * verts[:, 0]
        all_verts.append(verts)
        all_cam_t.append(pred_cam_t_full[n])
        all_right.append(is_right_n)

    t_render = time.time()
    cam_view = renderer.fast_render_rgb_frame_pytorch3d(
        all_verts, cam_t=all_cam_t, is_right=all_right
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    render_ms = (time.time() - t_render) * 1000

    input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
    input_img = np.concatenate(
        [input_img, np.ones_like(input_img[:, :, :1])], axis=2
    )
    input_img_overlay = (
        input_img[:, :, :3] * (1 - cam_view[:, :, 3:])
        + cam_view[:, :, :3] * cam_view[:, :, 3:]
    )
    output_img = (255 * input_img_overlay[:, :, ::-1]).astype(np.uint8)
    if flip_display:
        output_img = cv2.flip(output_img, 1)

    fps = 1.0 / max(time.time() - timer, 1e-6)
    return output_img, renderer, fps, det_ms, model_ms, render_ms


def main():
    parser = argparse.ArgumentParser(description="Realtime HaMeR video/webcam demo")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to HaMeR checkpoint",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to input video file (omit for webcam)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam / camera index when --video is not set",
    )
    parser.add_argument(
        "--out_folder",
        type=str,
        default="output",
        help="Output folder for rendered video",
    )
    parser.add_argument(
        "--out_video",
        type=str,
        default=None,
        help="Output video path (default: <out_folder>/hand_hamer.mp4)",
    )
    parser.add_argument("--max_batch_size", type=int, default=2)
    parser.add_argument("--rescale_factor", type=float, default=2.0)
    parser.add_argument(
        "--no_trt",
        action="store_true",
        help="Skip TensorRT compilation (PyTorch only)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show live OpenCV window (also used for webcam by default)",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Stop after N frames (0 = all)",
    )
    parser.add_argument(
        "--flip",
        action="store_true",
        help="Mirror output (useful for webcam selfie view)",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this demo")

    model, model_cfg = load_hamer(args.checkpoint)
    model = model.to(device)
    model.eval()

    if not args.no_trt:
        model = compile_hamer_tensorrt(model, device, args.max_batch_size)

    print(f"Model loaded on {device}")
    detector = create_detector(device="cuda")
    renderer = None

    os.makedirs(args.out_folder, exist_ok=True)

    if args.video is not None:
        source = args.video
        is_webcam = False
        flip_display = args.flip
        show = args.show
    else:
        source = args.camera
        is_webcam = True
        flip_display = True
        show = True

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    out_video = args.out_video
    if not is_webcam:
        if out_video is None:
            stem = Path(args.video).stem
            out_video = os.path.join(args.out_folder, f"{stem}_hamer.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_video, fourcc, fps_in, (width, height))
        print(f"Writing output to {out_video}")

    frame_idx = 0
    try:
        while True:
            ret, img_cv2 = cap.read()
            if not ret:
                break

            output_img, renderer, fps, det_ms, model_ms, render_ms = process_frame(
                img_cv2,
                model,
                model_cfg,
                detector,
                device,
                renderer,
                args.max_batch_size,
                args.rescale_factor,
                flip_display=flip_display,
            )

            cv2.putText(
                output_img,
                f"fps: {fps:.1f}  det:{det_ms:.0f}ms  hamer:{model_ms:.0f}ms  rend:{render_ms:.0f}ms",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if writer is not None:
                # Writer expects original orientation; undo flip if applied only for display.
                write_img = cv2.flip(output_img, 1) if flip_display else output_img
                if write_img.shape[1] != width or write_img.shape[0] != height:
                    write_img = cv2.resize(write_img, (width, height))
                writer.write(write_img)

            if show:
                cv2.imshow("realtime_hamer", output_img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if frame_idx % 10 == 0:
                print(
                    f"frame {frame_idx}: fps={fps:.1f} det={det_ms:.1f}ms "
                    f"hamer={model_ms:.1f}ms render={render_ms:.1f}ms"
                )
            if args.max_frames and frame_idx >= args.max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    print(f"Processed {frame_idx} frames")
    if out_video is not None:
        print(f"Saved: {out_video}")


if __name__ == "__main__":
    main()
