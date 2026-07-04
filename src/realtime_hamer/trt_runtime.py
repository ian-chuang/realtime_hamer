"""Load CUDA / TensorRT shared libraries from the venv before torch_tensorrt.

``torch-tensorrt`` and ``onnxruntime-gpu`` both need NVIDIA ``.so`` files that
ship inside site-packages (via torch / tensorrt wheels). Those directories are not
on the default dynamic-linker path, so we preload them once at import time.
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
    return [p / "lib" for p in nvidia.iterdir() if (p / "lib").is_dir()]


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

    for name in (
        "libcudart.so.13",
        "libcudart.so.12",
        "libcublasLt.so.12",
        "libcublas.so.12",
        "libcudnn.so.9",
        "libcufft.so.11",
        "libnvrtc.so.12",
        "libnvinfer.so.10",
        "libnvinfer_plugin.so.10",
    ):
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
