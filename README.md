# Realtime HaMeR

RTMPose wholebody (YOLOX CUDA + **RTMW TensorRT**) → HaMeR (**TensorRT**) → fixed-root MANO meshes for every real hand.

## Setup

```bash
uv sync
uv run python -m realtime_hamer.scripts.export_hamer_trt --assets-dir assets
```

Assets: `data/mano/MANO_RIGHT.pkl`, `hamer_ckpts/checkpoints/new_hamer_weights.ckpt`.

## Library

```python
from realtime_hamer.hand_pose_estimator import HandPoseEstimator

est = HandPoseEstimator(assets_dir="assets", smooth=0.65)
frame = est.estimate(frame_bgr, draw_overlay=True)
# frame.hands: list[HandMesh]  (is_right, vertices, faces, score)
# frame.overlay_bgr, frame.det_ms, frame.hamer_ms, frame.total_ms
```

Ghost opposite-hand slots from wholebody are filtered (score ratio + box IoU).

## Demo

```bash
uv run python -m realtime_hamer.scripts.demo --video assets/data/hand.mp4
```
