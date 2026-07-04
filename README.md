# Realtime HaMeR

RTMPose wholebody (`onnxruntime-gpu` CUDA) → HaMeR neural net (**TensorRT** engine via `trtexec`) → MANO mesh.

Same TensorRT stack as [fast_foundation_stereo](https://github.com/ian-chuang/fast_foundation_stereo): `tensorrt-cu12==11.0.0.114` + system `trtexec`.

## Setup

System TensorRT tools (once), same as FFS:

```bash
sudo apt install -y \
  libnvinfer-bin=11.0.0.114-1+cuda12.9 \
  libnvinfer11=11.0.0.114-1+cuda12.9 \
  libnvinfer-plugin11=11.0.0.114-1+cuda12.9 \
  libnvinfer-lean11=11.0.0.114-1+cuda12.9 \
  libnvinfer-dispatch11=11.0.0.114-1+cuda12.9 \
  libnvonnxparsers11=11.0.0.114-1+cuda12.9 \
  libnvinfer-vc-plugin11=11.0.0.114-1+cuda12.9
```

```bash
uv sync
```

Assets under `assets/`:

1. `data/mano/MANO_RIGHT.pkl`
2. `hamer_ckpts/checkpoints/new_hamer_weights.ckpt`

Build HaMeR engine (also done automatically on first `HandPoseEstimator` init):

```bash
uv run python -m realtime_hamer.scripts.export_hamer_trt --assets-dir assets
```

## Library

```python
from realtime_hamer.hand_pose_estimator import HandPoseEstimator

est = HandPoseEstimator(hand="right", assets_dir="assets")
out = est.estimate(frame_bgr)  # fixed-root verts, RTMPose overlay, timings
```

## Demo

```bash
uv run python -m realtime_hamer.scripts.demo --video assets/data/hand.mp4 --hand right
```
