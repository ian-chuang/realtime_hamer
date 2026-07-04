"""Realtime multi-hand HaMeR demo (viser)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tyro
import viser

from realtime_hamer.hand_pose_estimator import HandPoseEstimator

HAND_COLOR = {True: (36, 120, 143), False: (133, 138, 241)}


@dataclass
class Args:
    video: Path | None = None
    """Video path (loops forever). If unset, use --camera."""

    camera: int = 0
    assets_dir: Path = Path("assets")
    port: int = 8080
    smooth: float = 0.65


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
    estimator = HandPoseEstimator(assets_dir=args.assets_dir, smooth=args.smooth)
    cap, loop = open_capture(args.video, args.camera)

    server = viser.ViserServer(port=args.port)
    server.scene.add_frame("/world", axes_length=0.05, axes_radius=0.002)
    gui_fps = server.gui.add_text("FPS", initial_value="—", disabled=True)
    gui_smooth = server.gui.add_slider("Smooth", min=0.05, max=1.0, step=0.05, initial_value=args.smooth)
    gui_show_rtm = server.gui.add_checkbox("Show RTMPose", initial_value=True)
    gui_show_3d = server.gui.add_checkbox("Show 3D hands", initial_value=True)
    gui_pose = server.gui.add_image(np.zeros((64, 64, 3), dtype=np.uint8), label="RTMPose")
    handles: dict[bool, object | None] = {False: None, True: None}

    print(f"Viser http://localhost:{args.port}  (Ctrl+C to stop)")

    try:
        while True:
            frame = read_frame(cap, loop)
            if frame is None:
                break

            estimator.smooth = float(gui_smooth.value)
            show_rtm = bool(gui_show_rtm.value)
            show_3d = bool(gui_show_3d.value)

            est = estimator.estimate(frame, draw_overlay=show_rtm)
            if show_rtm and est.overlay_bgr is not None:
                gui_pose.image = cv2.cvtColor(est.overlay_bgr, cv2.COLOR_BGR2RGB)
                gui_pose.visible = True
            else:
                gui_pose.visible = False

            seen = {False: False, True: False}
            if show_3d:
                for hand in est.hands:
                    seen[hand.is_right] = True
                    # Offset left/right slightly so both are visible.
                    verts = hand.vertices.copy()
                    verts[:, 0] += -0.12 if not hand.is_right else 0.12
                    h = handles[hand.is_right]
                    if h is None:
                        handles[hand.is_right] = server.scene.add_mesh_simple(
                            f"/hand/{'right' if hand.is_right else 'left'}",
                            vertices=verts,
                            faces=hand.faces,
                            color=HAND_COLOR[hand.is_right],
                            side="double",
                        )
                    else:
                        h.vertices = verts
                        h.visible = True
            for side, h in handles.items():
                if h is not None and (not show_3d or not seen[side]):
                    h.visible = False

            fps = 1000.0 / max(est.total_ms, 1e-3)
            sides = ",".join(
                (["L"] if any(not h.is_right for h in est.hands) else [])
                + (["R"] if any(h.is_right for h in est.hands) else [])
            ) or "none"
            gui_fps.value = (
                f"{fps:.1f} fps   det {est.det_ms:.1f} ms   hamer {est.hamer_ms:.1f} ms   "
                f"hands={sides}"
            )
            print(gui_fps.value, end="\r")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()


if __name__ == "__main__":
    main(tyro.cli(Args))
