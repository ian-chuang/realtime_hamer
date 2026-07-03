"""Preload GPU shared libraries for torch_tensorrt and onnxruntime-gpu.

torch-tensorrt 2.9 ships against CUDA 13 / TensorRT 10 wheels. onnxruntime-gpu
1.22 uses CUDA 12 libs from the torch/nvidia wheels. Those live inside the
venv but are not on the default loader path.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path


def _site_packages() -> Path:
    for path in map(Path, sys.path):
        if path.name == "site-packages" and path.is_dir():
            return path
    raise RuntimeError("Could not locate site-packages")


def _nvidia_lib_dirs(site: Path) -> list[Path]:
    nvidia = site / "nvidia"
    if not nvidia.is_dir():
        return []
    dirs = []
    for child in nvidia.iterdir():
        lib_dir = child / "lib"
        if lib_dir.is_dir():
            dirs.append(lib_dir)
    return dirs


def preload_gpu_libs() -> None:
    site = _site_packages()
    lib_dirs = [
        *_nvidia_lib_dirs(site),
        site / "tensorrt_libs",
        site / "tensorrt_lean_libs",
        site / "tensorrt_dispatch_libs",
        site / "torch" / "lib",
    ]
    existing = [str(p) for p in lib_dirs if p.is_dir()]
    if existing:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
            existing
            + ([os.environ["LD_LIBRARY_PATH"]] if os.environ.get("LD_LIBRARY_PATH") else [])
        )

    # Prefer CUDA 13 (torch_tensorrt) then CUDA 12 (onnxruntime-gpu / torch).
    preload_names = [
        "libcudart.so.13",
        "libcudart.so.12",
        "libcublasLt.so.12",
        "libcublas.so.12",
        "libcudnn.so.9",
        "libcufft.so.11",
        "libnvrtc.so.12",
        "libnvinfer.so.10",
        "libnvinfer_plugin.so.10",
    ]
    for name in preload_names:
        for lib_dir in lib_dirs:
            candidate = lib_dir / name
            if candidate.is_file():
                try:
                    ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
                break


def import_torch_tensorrt():
    preload_gpu_libs()
    import torch_tensorrt

    return torch_tensorrt
