# Realtime HaMeR

RTMPose hand models (`onnxruntime-gpu`) → HaMeR (PyTorch FP16) → fixed-root MANO mesh.

## Setup

```bash
uv sync
```

Assets under `assets/`:

1. `data/mano/MANO_RIGHT.pkl` from https://mano.is.tue.mpg.de
2. `hamer_ckpts/checkpoints/new_hamer_weights.ckpt` from https://gkarv.github.io/hand-texture-module/

## Library usage

```python
from realtime_hamer.hand_pose_estimator import HandPoseEstimator

est = HandPoseEstimator(hand="right", assets_dir="assets")
out = est.estimate(frame_bgr)  # out.vertices at origin, out.overlay_bgr, timings
```

## Demo

```bash
uv run python -m realtime_hamer.scripts.demo --video assets/data/hand.mp4 --hand right
uv run python -m realtime_hamer.scripts.demo --camera 0 --hand right
```

Open http://localhost:8080.
