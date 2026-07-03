# Realtime HaMeR

Fast 3D hand mesh reconstruction from video or webcam:

1. **RTMPose** wholebody detection (`rtmlib` + `onnxruntime-gpu`)
2. **HaMeR** reconstruction accelerated with **TensorRT** (`torch-tensorrt`)
3. **PyTorch3D** mesh overlay rendering

Based on [hamer-demo](https://github.com/ATAboukhadra/hamer-demo) and [HaMeR](https://github.com/geopavlakos/hamer).

## 3. Install UV

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## 4. Install Python dependencies

```bash
uv sync
```

This pulls PyTorch 2.9 / CUDA 12.8, `torch-tensorrt`, `onnxruntime-gpu` (CUDA 12), `rtmlib`, HaMeR deps, and a prebuilt [pytorch3d](https://github.com/MiroPsota/torch_packages_builder) wheel.

---

## 5. Assets

You need to download the MANO model file manually:

1. Go to the MANO website: https://mano.is.tue.mpg.de
2. Register/login and open the downloads section.
3. Download the MANO model package.
4. Use the **right-hand** model file `MANO_RIGHT.pkl`.

Then place `MANO_RIGHT.pkl` at:

```text
assets/data/mano/MANO_RIGHT.pkl
```

For the upgraded HaMeR checkpoint (`new_hamer_weights.ckpt`), download it from:

https://gkarv.github.io/hand-texture-module/

Then place it at:

```text
assets/hamer_ckpts/checkpoints/new_hamer_weights.ckpt
```

`assets/hamer_ckpts/model_config.yaml` and `assets/data/mano_mean_params.npz` should already be present.

---

## 6. Run demo

### Video file

```bash
uv run python scripts/demo_video.py \
  --video assets/data/hand.mp4 \
  --out_folder output
```

Writes `output/hand_hamer.mp4` with mesh overlays and FPS / timing HUD.

### Webcam

```bash
uv run python scripts/demo_video.py --camera 0
```

Press `q` to quit.

### Useful flags

| Flag | Description |
| --- | --- |
| `--no_trt` | Skip TensorRT compile (PyTorch only, slower) |
| `--max_frames N` | Process only the first N frames |
| `--show` | Open an OpenCV window for video-file runs |
| `--max_batch_size` | Max hands per frame for TRT engines (default 2) |

First TensorRT compile of the ViT backbone + transformer can take a few minutes; later frames are much faster.

Typical timings on a recent NVIDIA GPU (hands present): detection ~5–10 ms, HaMeR TRT ~5 ms, PyTorch3D render ~25 ms.
