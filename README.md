# Realtime HaMeR

RTMPose (`onnxruntime-gpu`) → HaMeR (TensorRT) → [viser](https://viser.studio) 3D view.

Based on [hamer-demo](https://github.com/ATAboukhadra/hamer-demo) and [HaMeR](https://github.com/geopavlakos/hamer).

## Setup

```bash
uv sync
```

Assets:

1. `MANO_RIGHT.pkl` from https://mano.is.tue.mpg.de → `assets/data/mano/MANO_RIGHT.pkl`
2. `new_hamer_weights.ckpt` from https://gkarv.github.io/hand-texture-module/ → `assets/hamer_ckpts/checkpoints/new_hamer_weights.ckpt`

## Run

```bash
# looping video
uv run python -m realtime_hamer.scripts.demo --video assets/data/hand.mp4

# webcam
uv run python -m realtime_hamer.scripts.demo --camera 0
```

Open http://localhost:8080. First run compiles TensorRT into `assets/trt_cache/`; later runs load the cache.

Viser sidebar:

- **RTMPose** image: video with hand keypoints / boxes
- **Fix hand root** (on by default): wrists pinned in front of the camera so you can focus on fingers
- **Focal** / **Depth scale**: used when fix-root is off, to place hands in camera space (defaults assume a hand ~10–30 cm away)
