"""PyTorch and TorchScript model loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from modelbench.loaders.base import LoadedModel, ModelLoader


DTYPE_MAP = {
    "float16": "float16",
    "float32": "float32",
    "float64": "float64",
    "int32": "int32",
    "int64": "int64",
}


@dataclass
class TorchLoadedModel(LoadedModel):
    """Shared runtime wrapper around torch models."""

    torch_module: Any
    torch_module_ref: Any

    def infer(self, input_data: Any) -> Any:
        with self.torch_module_ref.no_grad():
            return self.torch_module(input_data)

    def synchronize(self) -> None:
        if self.device == "cuda":
            self.torch_module_ref.cuda.synchronize()

    def parameter_count(self) -> Optional[int]:
        if not hasattr(self.torch_module, "parameters"):
            return None
        return int(sum(parameter.numel() for parameter in self.torch_module.parameters()))

    def non_trainable_parameter_count(self) -> Optional[int]:
        if not hasattr(self.torch_module, "parameters"):
            return None
        return int(
            sum(parameter.numel() for parameter in self.torch_module.parameters() if not parameter.requires_grad)
        )

    def reset_peak_memory(self) -> None:
        if self.device == "cuda":
            self.torch_module_ref.cuda.reset_peak_memory_stats()

    def peak_memory_bytes(self) -> Optional[int]:
        if self.device == "cuda":
            return int(self.torch_module_ref.cuda.max_memory_allocated())
        if self.device == "mps" and hasattr(self.torch_module_ref, "mps"):
            return int(self.torch_module_ref.mps.current_allocated_memory())
        return None


class PyTorchModelLoader(ModelLoader):
    """Loads PyTorch or TorchScript models from disk."""

    format_name = "pytorch"

    def __init__(self) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - import failure depends on env
            raise RuntimeError("PyTorch is required for PyTorch benchmarking") from exc
        self.torch = torch

    def load(self, model_path: str, device: str) -> TorchLoadedModel:
        resolved_path = Path(model_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        map_location = device if device != "mps" else "cpu"
        module = None
        torchscript_loaded = False

        try:
            module = self.torch.load(resolved_path, map_location=map_location)
        except Exception:
            module = self.torch.jit.load(resolved_path, map_location=map_location)
            torchscript_loaded = True

        if hasattr(module, "eval"):
            module = module.eval()
        if hasattr(module, "to"):
            module = module.to(device)

        return TorchLoadedModel(
            model_path=str(resolved_path),
            format_name="torchscript" if torchscript_loaded else self.format_name,
            device=device,
            backend=module,
            torch_module=module,
            torch_module_ref=self.torch,
        )

    def make_input(self, batch_size: int, input_shape: list[int], dtype: str, device: str) -> Any:
        torch_dtype = getattr(self.torch, DTYPE_MAP.get(dtype, "float32"))
        shape = [batch_size, *input_shape]
        if torch_dtype in {self.torch.int32, self.torch.int64}:
            return self.torch.randint(0, 10, shape, dtype=torch_dtype, device=device)
        return self.torch.randn(*shape, dtype=torch_dtype, device=device)
