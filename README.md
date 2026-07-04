# Realtime HaMeR

RTMPose wholebody (YOLOX CUDA + **RTMW TensorRT**) → HaMeR net (**TensorRT**) → MANO mesh.

TensorRT stack matches [fast_foundation_stereo](https://github.com/ian-chuang/fast_foundation_stereo): `tensorrt-cu12==11.0.0.114` + system `trtexec`.

## Setup

```bash
uv sync
# first-time engine build (also auto on first HandPoseEstimator init):
uv run python -m realtime_hamer.scripts.export_hamer_trt --assets-dir assets
```

Assets: `data/mano/MANO_RIGHT.pkl`, `hamer_ckpts/checkpoints/new_hamer_weights.ckpt`.

## Library

```python
from realtime_hamer.hand_pose_estimator import HandPoseEstimator

est = HandPoseEstimator(hand="right", assets_dir="assets")
out = est.estimate(frame_bgr)
# out.vertices          — fixed-root mesh for 3D
# out.overlay_bgr       — RTMPose keypoints
# out.mesh_overlay_bgr  — MANO projected on the video
# out.det_ms / hamer_ms / total_ms
```

## Demo

```bash
uv run python -m realtime_hamer.scripts.demo --video assets/data/hand.mp4 --hand right
```

Viser sidebar: RTMPose feed, HaMeR mesh overlay, FPS timings; 3D hand at origin.
