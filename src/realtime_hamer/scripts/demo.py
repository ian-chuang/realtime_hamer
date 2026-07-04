"""Realtime single-hand HaMeR demo (viser)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import tyro
import viser

from realtime_hamer.hand_pose_estimator import HandPoseEstimator

HandSide = Literal["left", "right"]
HAND_COLOR = {"right": (36, 120, 143), "left": (133, 138, 241)}


@dataclass
class Args:
    video: Path | None = None
    """Video path (loops forever). If unset, use --camera."""

    camera: int = 0
    """Webcam index when --video is not set."""

    hand: HandSide = "right"
    """Which hand to track."""

    assets_dir: Path = Path("assets")
    """Directory with hamer_ckpts/ and data/mano/."""

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


def main(args: Args) -> None:
    estimator = HandPoseEstimator(hand=args.hand, assets_dir=args.assets_dir)
    cap, loop = open_capture(args.video, args.camera)

    server = viser.ViserServer(port=args.port)
    server.scene.add_frame("/world", axes_length=0.05, axes_radius=0.002)
    gui_fps = server.gui.add_text("FPS", initial_value="—", disabled=True)
    gui_img = server.gui.add_image(np.zeros((64, 64, 3), dtype=np.uint8), label="RTMPose")
    mesh_handle = None

    print(f"Viser http://localhost:{args.port}  hand={args.hand}  (Ctrl+C to stop)")

    try:
        while True:
            frame = read_frame(cap, loop)
            if frame is None:
                break

            est = estimator.estimate(frame)
            gui_img.image = cv2.cvtColor(est.overlay_bgr, cv2.COLOR_BGR2RGB)
            fps = 1000.0 / max(est.total_ms, 1e-3)
            gui_fps.value = (
                f"{fps:.1f} fps   det {est.det_ms:.1f} ms   hamer {est.hamer_ms:.1f} ms   "
                f"{'hand' if est.detected else 'none'}"
            )

            if est.detected:
                assert est.vertices is not None
                if mesh_handle is None:
                    mesh_handle = server.scene.add_mesh_simple(
                        "/hand",
                        vertices=est.vertices,
                        faces=est.faces,
                        color=HAND_COLOR[args.hand],
                        side="double",
                    )
                else:
                    mesh_handle.vertices = est.vertices
                    mesh_handle.visible = True
            elif mesh_handle is not None:
                mesh_handle.visible = False

            print(gui_fps.value, end="\r")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()


if __name__ == "__main__":
    main(tyro.cli(Args))
