"""ONNX Runtime model loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from modelbench.loaders.base import LoadedModel, ModelLoader


NUMPY_DTYPE_MAP = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "int32": np.int32,
    "int64": np.int64,
}


@dataclass
class ONNXLoadedModel(LoadedModel):
    """Inference wrapper around an ONNX Runtime session."""

    session: Any
    input_name: str

    def infer(self, input_data: Any) -> Any:
        return self.session.run(None, {self.input_name: input_data})


class ONNXModelLoader(ModelLoader):
    """Loads ONNX models through onnxruntime."""

    format_name = "onnx"

    def __init__(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - import failure depends on env
            raise RuntimeError("onnxruntime is required for ONNX benchmarking") from exc
        self.ort = ort

    def load(self, model_path: str, device: str) -> ONNXLoadedModel:
        resolved_path = Path(model_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        providers = self._providers_for_device(device)
        session = self.ort.InferenceSession(str(resolved_path), providers=providers)
        input_name = session.get_inputs()[0].name
        return ONNXLoadedModel(
            model_path=str(resolved_path),
            format_name=self.format_name,
            device=device,
            backend=session,
            session=session,
            input_name=input_name,
        )

    def make_input(self, batch_size: int, input_shape: list[int], dtype: str, device: str) -> Any:
        del device
        np_dtype = NUMPY_DTYPE_MAP.get(dtype, np.float32)
        shape = [batch_size, *input_shape]
        return np.random.randn(*shape).astype(np_dtype)

    def _providers_for_device(self, device: str) -> list[str]:
        available = set(self.ort.get_available_providers())
        if device == "cuda":
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError("CUDAExecutionProvider is not available in this onnxruntime build")
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "mps":
            if "CoreMLExecutionProvider" not in available:
                raise RuntimeError("CoreMLExecutionProvider is not available in this onnxruntime build")
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

