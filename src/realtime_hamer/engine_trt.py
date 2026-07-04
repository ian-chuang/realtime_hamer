"""Minimal TensorRT engine runner (same pattern as fast_foundation_stereo)."""

from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
import torch


class TrtRunner:
    """Run a serialized TensorRT engine with torch CUDA tensors."""

    def __init__(self, engine_path: str):
        import tensorrt as trt

        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(
                f"Failed to load TRT engine {engine_path}. "
                f"Rebuild with trtexec (TensorRT {trt.__version__})."
            )
        self.context = self.engine.create_execution_context()
        self.input_names = []
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

    def _dtype(self, dt):
        trt = self.trt
        return {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.BF16: torch.bfloat16,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT8: torch.int8,
            trt.DataType.BOOL: torch.bool,
        }[dt]

    def __call__(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        trt = self.trt
        for name, tensor in inputs.items():
            expected = self._dtype(self.engine.get_tensor_dtype(name))
            if tensor.dtype != expected:
                tensor = tensor.to(expected)
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            inputs[name] = tensor
            self.context.set_input_shape(name, tuple(tensor.shape))

        outputs = {}
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = self._dtype(self.engine.get_tensor_dtype(name))
            outputs[name] = torch.empty(shape, device="cuda", dtype=dtype)

        for name, tensor in inputs.items():
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
        for name, tensor in outputs.items():
            self.context.set_tensor_address(name, int(tensor.data_ptr()))

        stream = torch.cuda.current_stream().cuda_stream
        assert self.context.execute_async_v3(stream)
        return outputs


class TrtOrtSession:
    """onnxruntime.InferenceSession-compatible wrapper over a TRT engine.

    ``input_names`` / ``output_names`` must match the original ONNX/ORT order
    (e.g. RTMPose expects simcc_x then simcc_y). TRT engine enumeration order
    can differ and will silently break postprocessing if not preserved.
    """

    def __init__(
        self,
        engine_path: str | Path,
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
    ):
        self.runner = TrtRunner(str(engine_path))
        self._input_names = list(input_names or self.runner.input_names)
        self._output_names = list(output_names or self.runner.output_names)
        # Map ORT feed keys -> engine binding names if they differ.
        self._input_alias = {}
        for ort_name, eng_name in zip(self._input_names, self.runner.input_names):
            self._input_alias[ort_name] = eng_name
            self._input_alias[eng_name] = eng_name

    def get_inputs(self):
        return [type("I", (), {"name": n})() for n in self._input_names]

    def get_outputs(self):
        return [type("O", (), {"name": n})() for n in self._output_names]

    def run(self, output_names, input_feed):
        torch_in = {}
        for name, arr in input_feed.items():
            eng_name = self._input_alias.get(name, name)
            if eng_name not in self.runner.input_names:
                eng_name = self.runner.input_names[0]
            torch_in[eng_name] = torch.from_numpy(np.ascontiguousarray(arr)).cuda()
        outs = self.runner(torch_in)
        torch.cuda.synchronize()
        names = output_names if output_names else self._output_names
        return [outs[n].float().cpu().numpy() for n in names]


def build_engine_from_onnx(onnx_path: Path, engine_path: Path) -> Path:
    """Build a TRT engine with system trtexec (TensorRT 11)."""
    if engine_path.is_file():
        return engine_path
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["trtexec", f"--onnx={onnx_path}", f"--saveEngine={engine_path}"]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return engine_path
