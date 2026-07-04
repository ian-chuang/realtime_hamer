"""Realtime multi-hand HaMeR demo (viser)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tyro
import viser

from realtime_hamer.hand_pose_estimator import HandMesh, HandPoseEstimator
from realtime_hamer.viz import HandMeshOverlayRenderer, HandSmoother, draw_keypoints_overlay

HAND_COLOR = {True: (36, 120, 143), False: (133, 138, 241)}
HAND_OFFSET_X = {True: 0.12, False: -0.12}


@dataclass
class Args:
    video: Path | None = None
    """Video path (loops forever). If unset, use --camera."""

    camera: int = 0
    assets_dir: Path = Path("assets")
    port: int = 8080
    smooth: float = 0.65
    mirror: bool = True
    """Horizontally flip frames (webcam / selfie video). Default on."""


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


def _best_per_side(hands: list[HandMesh]) -> dict[bool, HandMesh | None]:
    best: dict[bool, HandMesh | None] = {False: None, True: None}
    for hand in hands:
        prev = best[hand.is_right]
        if prev is None or hand.score > prev.score:
            best[hand.is_right] = hand
    return best


def _multi_side_warning(hands: list[HandMesh]) -> str:
    n_l = sum(1 for h in hands if not h.is_right)
    n_r = sum(1 for h in hands if h.is_right)
    parts = []
    if n_l > 1:
        parts.append(f"{n_l} left hands")
    if n_r > 1:
        parts.append(f"{n_r} right hands")
    if not parts:
        return ""
    return "multiple " + " and ".join(parts) + " (3D shows highest score per side)"


def main(args: Args) -> None:
    estimator = HandPoseEstimator(assets_dir=args.assets_dir)
    smoother = HandSmoother(alpha=args.smooth)
    mesh_renderer = HandMeshOverlayRenderer(
        faces_right=estimator.faces_right, device=estimator.device
    )
    cap, loop = open_capture(args.video, args.camera)

    server = viser.ViserServer(port=args.port)
    server.scene.add_frame("/world", axes_length=0.05, axes_radius=0.002)
    gui_fps = server.gui.add_text("FPS", initial_value="—", disabled=True)
    gui_warn = server.gui.add_text("Warn", initial_value="", disabled=True)
    gui_smooth = server.gui.add_slider("Smooth", min=0.05, max=1.0, step=0.05, initial_value=args.smooth)
    gui_mirror = server.gui.add_checkbox("Mirror", initial_value=args.mirror)
    gui_show_rtm = server.gui.add_checkbox("Show RTMPose", initial_value=True)
    gui_show_mesh = server.gui.add_checkbox("Show MANO mesh", initial_value=False)
    gui_show_3d = server.gui.add_checkbox("Show 3D hands", initial_value=True)
    gui_pose = server.gui.add_image(np.zeros((64, 64, 3), dtype=np.uint8), label="RTMPose")
    gui_mesh = server.gui.add_image(np.zeros((64, 64, 3), dtype=np.uint8), label="MANO mesh")
    mesh_handles: dict[bool, object | None] = {False: None, True: None}
    label_handles: dict[bool, object | None] = {False: None, True: None}
    warned_multi = False

    print(f"Viser http://localhost:{args.port}  mirror={args.mirror}  (Ctrl+C to stop)")

    try:
        while True:
            frame = read_frame(cap, loop)
            if frame is None:
                break
            if gui_mirror.value:
                frame = cv2.flip(frame, 1)

            smoother.alpha = float(gui_smooth.value)
            show_rtm = bool(gui_show_rtm.value)
            show_mesh = bool(gui_show_mesh.value)
            show_3d = bool(gui_show_3d.value)

            est = estimator.estimate(frame)
            hands = smoother(est.hands)

            warn = _multi_side_warning(hands)
            gui_warn.value = warn
            if warn and not warned_multi:
                print(f"\n[warn] {warn}")
                warned_multi = True
            elif not warn:
                warned_multi = False

            if show_rtm:
                overlay = draw_keypoints_overlay(frame, hands)
                gui_pose.image = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                gui_pose.visible = True
            else:
                gui_pose.visible = False

            mesh_ms = 0.0
            if show_mesh and hands:
                t_mesh = time.perf_counter()
                mesh_img = mesh_renderer.render(frame, hands)
                mesh_ms = (time.perf_counter() - t_mesh) * 1000.0
                gui_mesh.image = cv2.cvtColor(mesh_img, cv2.COLOR_BGR2RGB)
                gui_mesh.visible = True
            else:
                gui_mesh.visible = False

            best = _best_per_side(hands)
            for is_right in (False, True):
                name = "right" if is_right else "left"
                hand = best[is_right]
                if show_3d and hand is not None:
                    verts = hand.vertices.copy()
                    verts[:, 0] += HAND_OFFSET_X[is_right]
                    label_pos = (float(verts[:, 0].mean()), float(verts[:, 1].max()) + 0.03, 0.0)
                    mh = mesh_handles[is_right]
                    if mh is None:
                        mesh_handles[is_right] = server.scene.add_mesh_simple(
                            f"/hand/{name}",
                            vertices=verts,
                            faces=hand.faces,
                            color=HAND_COLOR[is_right],
                            side="double",
                        )
                    else:
                        mh.vertices = verts
                        mh.visible = True
                    lh = label_handles[is_right]
                    if lh is None:
                        label_handles[is_right] = server.scene.add_label(
                            f"/hand/{name}/label",
                            text=name.upper(),
                            position=label_pos,
                        )
                    else:
                        lh.position = label_pos
                        lh.visible = True
                else:
                    if mesh_handles[is_right] is not None:
                        mesh_handles[is_right].visible = False
                    if label_handles[is_right] is not None:
                        label_handles[is_right].visible = False

            fps = 1000.0 / max(est.total_ms + mesh_ms, 1e-3)
            n_l = sum(1 for h in hands if not h.is_right)
            n_r = sum(1 for h in hands if h.is_right)
            gui_fps.value = (
                f"{fps:.1f} fps   det {est.det_ms:.1f} ms   "
                f"trt {est.trt_ms:.1f} ms   hamer {est.hamer_ms:.1f} ms"
                + (f"   mesh {mesh_ms:.1f} ms" if show_mesh else "")
                + f"   L={n_l} R={n_r}"
            )
            print(gui_fps.value, end="\r")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()


if __name__ == "__main__":
    main(tyro.cli(Args))
