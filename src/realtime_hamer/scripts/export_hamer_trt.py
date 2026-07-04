"""Export HaMeR backbone+head to ONNX and build a TensorRT engine (FFS-style)."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import tyro

from realtime_hamer.models import load_hamer


class HamerNet(nn.Module):
    """Neural net only (no MANO). Input crop is (B,3,256,256)."""

    def __init__(self, hamer):
        super().__init__()
        self.backbone = hamer.backbone
        self.mano_head = hamer.mano_head

    def forward(self, img: torch.Tensor):
        # Match HAMER.forward_step: drop 32 px on each side of width.
        feats = self.backbone(img[:, :, :, 32:-32])
        pred_mano_params, pred_cam, _ = self.mano_head(feats)
        return (
            pred_mano_params["global_orient"],
            pred_mano_params["hand_pose"],
            pred_mano_params["betas"],
            pred_cam,
        )


@dataclass
class Args:
    assets_dir: Path = Path("assets")
    """Directory with hamer_ckpts/ and data/mano/."""

    out_dir: Path = Path("assets/trt_cache")
    """Where to write ONNX + engine."""

    fp16: bool = True
    """Build FP16 TensorRT engine."""


def export_onnx(assets_dir: Path, out_dir: Path) -> Path:
    import realtime_hamer.configs as configs
    import realtime_hamer.models as models

    assets_dir = assets_dir.resolve()
    configs.CACHE_DIR_HAMER = str(assets_dir)
    models.DEFAULT_CHECKPOINT = f"{assets_dir}/hamer_ckpts/checkpoints/new_hamer_weights.ckpt"

    device = torch.device("cuda")
    model, _ = load_hamer(models.DEFAULT_CHECKPOINT)
    model = model.to(device).eval()
    net = HamerNet(model).to(device).eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "hamer.onnx"
    dummy = torch.randn(1, 3, 256, 256, device=device)

    print(f"Exporting ONNX -> {onnx_path}")
    torch.onnx.export(
        net,
        dummy,
        str(onnx_path),
        input_names=["img"],
        output_names=["global_orient", "hand_pose", "betas", "pred_cam"],
        opset_version=17,
        dynamo=False,
    )
    return onnx_path


def build_engine(onnx_path: Path, engine_path: Path, fp16: bool = True) -> Path:
    """Build engine with trtexec (TensorRT 11 strongly-typed; no --fp16 flag)."""
    build_onnx = onnx_path
    if fp16:
        # Optional FFS-style FP16 ONNX via nvidia-modelopt (if installed).
        fp16_onnx = onnx_path.with_name(onnx_path.stem + ".fp16.onnx")
        if not fp16_onnx.is_file():
            try:
                subprocess.run(
                    ["python", "-m", "modelopt.onnx.autocast", f"--onnx_path={onnx_path}"],
                    check=True,
                )
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                print(f"FP16 autocast skipped ({exc}); building FP32 engine")
        if fp16_onnx.is_file():
            build_onnx = fp16_onnx

    cmd = ["trtexec", f"--onnx={build_onnx}", f"--saveEngine={engine_path}"]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return engine_path


def ensure_hamer_engine(assets_dir: Path, out_dir: Path | None = None, fp16: bool = True) -> Path:
    """Export+build if missing; return path to ``hamer.engine``."""
    out_dir = (out_dir or (Path(assets_dir) / "trt_cache")).resolve()
    engine_path = out_dir / "hamer.engine"
    if engine_path.is_file():
        return engine_path

    onnx_path = out_dir / "hamer.onnx"
    if not onnx_path.is_file():
        export_onnx(Path(assets_dir), out_dir)
    build_engine(onnx_path, engine_path, fp16=fp16)
    return engine_path


def main(args: Args) -> None:
    engine = ensure_hamer_engine(args.assets_dir, args.out_dir, fp16=args.fp16)
    print(f"Ready: {engine}")


if __name__ == "__main__":
    main(tyro.cli(Args))
