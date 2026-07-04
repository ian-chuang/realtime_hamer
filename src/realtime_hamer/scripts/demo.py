"""Realtime single-hand HaMeR demo (viser)."""

from __future__ import annotations

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
    """Only this hand slot is used."""

    assets_dir: Path = Path("assets")
    """Directory with hamer_ckpts/ and data/mano/."""

    smooth: float = 0.65
    """EMA weight on new observations (1 = no smoothing)."""

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
    estimator = HandPoseEstimator(
        hand=args.hand,
        assets_dir=args.assets_dir,
        smooth=args.smooth,
    )
    cap, loop = open_capture(args.video, args.camera)

    server = viser.ViserServer(port=args.port)
    server.scene.add_frame("/world", axes_length=0.05, axes_radius=0.002)

    with server.gui.add_folder("Timing"):
        gui_fps = server.gui.add_text("FPS", initial_value="—", disabled=True)
        gui_detail = server.gui.add_text("Steps", initial_value="—", disabled=True)

    with server.gui.add_folder("Controls"):
        gui_smooth = server.gui.add_slider(
            "Smooth", min=0.05, max=1.0, step=0.05, initial_value=args.smooth
        )
        gui_show_pose = server.gui.add_checkbox("Show RTMPose", initial_value=True)
        gui_show_mesh2d = server.gui.add_checkbox("Show HaMeR mesh 2D", initial_value=True)
        gui_show_3d = server.gui.add_checkbox("Show 3D mesh", initial_value=True)

    gui_pose = server.gui.add_image(np.zeros((64, 64, 3), dtype=np.uint8), label="RTMPose")
    gui_mesh = server.gui.add_image(np.zeros((64, 64, 3), dtype=np.uint8), label="HaMeR mesh")
    mesh_handle = None
    blank = np.zeros((64, 64, 3), dtype=np.uint8)

    print(f"Viser http://localhost:{args.port}  hand={args.hand}  (Ctrl+C to stop)")

    try:
        while True:
            frame = read_frame(cap, loop)
            if frame is None:
                break

            estimator.smooth = float(gui_smooth.value)
            show_pose = bool(gui_show_pose.value)
            show_mesh2d = bool(gui_show_mesh2d.value)
            show_3d = bool(gui_show_3d.value)

            est = estimator.estimate(
                frame,
                draw_pose=show_pose,
                draw_mesh=show_mesh2d,
                want_3d=show_3d,
            )

            gui_pose.image = (
                cv2.cvtColor(est.overlay_bgr, cv2.COLOR_BGR2RGB)
                if show_pose and est.overlay_bgr is not None
                else blank
            )
            gui_mesh.image = (
                cv2.cvtColor(est.mesh_overlay_bgr, cv2.COLOR_BGR2RGB)
                if show_mesh2d and est.mesh_overlay_bgr is not None
                else blank
            )

            if show_3d and est.detected and est.vertices is not None:
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

            fps = 1000.0 / max(est.total_ms, 1e-3)
            gui_fps.value = (
                f"{fps:.1f} fps   total {est.total_ms:.1f} ms   "
                f"{'hand' if est.detected else 'none'}"
            )
            gui_detail.value = (
                f"det {est.det_ms:.1f}  hamer {est.hamer_ms:.1f}  "
                f"pose2d {est.pose_overlay_ms:.1f}  mesh2d {est.mesh_overlay_ms:.1f}"
            )
            print(f"{gui_fps.value}  |  {gui_detail.value}", end="\r")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()


if __name__ == "__main__":
    main(tyro.cli(Args))
