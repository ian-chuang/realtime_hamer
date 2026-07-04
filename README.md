# Realtime HaMeR

RTMPose wholebody (YOLOX CUDA + **RTMW TensorRT**) → HaMeR (**TensorRT**) → full MANO reconstructions for every real hand.

Optional viz helpers (PyTorch3D mesh overlay, keypoint draw, EMA) live in `realtime_hamer.viz` — the estimator itself only reconstructs.

PyTorch3D is installed from [torch_packages_builder](https://github.com/MiroPsota/torch_packages_builder) prebuilt wheels ([hamer-demo](https://github.com/ATAboukhadra/hamer-demo) render path).

## Setup

```bash
uv sync
uv run python -m realtime_hamer.scripts.export_hamer_trt --assets-dir assets
```

Assets: `data/mano/MANO_RIGHT.pkl`, `hamer_ckpts/checkpoints/new_hamer_weights.ckpt`.

## Library

```python
from realtime_hamer.hand_pose_estimator import HandPoseEstimator
from realtime_hamer.viz import HandMeshOverlayRenderer, HandSmoother, draw_keypoints_overlay

est = HandPoseEstimator(assets_dir="assets")
smoother = HandSmoother(alpha=0.65)
mesh_renderer = HandMeshOverlayRenderer(est.faces_right, device=est.device)

out = est.estimate(frame_bgr)          # reconstruction only
hands = smoother(out.hands)            # optional per-side EMA
kpt_img = draw_keypoints_overlay(frame_bgr, hands)
mesh_img = mesh_renderer.render(frame_bgr, hands)  # optional; skip when unused
```

Each `HandMesh` includes: `vertices`, `faces`, `vertices_mano`, `vertices_cam`, `joints_cam`, `cam_t`, `focal`, `global_orient`, `hand_pose`, `betas`, `pred_cam`, `box`, `kpts`, `kpt_scores`, `is_right`, `score`.

## Demo

Webcam / selfie video is mirrored by default (`--mirror` / `--no-mirror`):

```bash
uv run python -m realtime_hamer.scripts.demo --video assets/data/hand.mp4
```
