"""Shared test helpers."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from modelbench.loaders.pytorch import TorchLoadedModel


class TinyLinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


def build_loaded_torch_model(tmp_path: Path) -> tuple[TorchLoadedModel, torch.Tensor]:
    model = TinyLinearModel().eval()
    model_path = tmp_path / "tiny_model.pt"
    traced = torch.jit.trace(model, torch.randn(1, 4))
    traced.save(str(model_path))
    loaded = TorchLoadedModel(
        model_path=str(model_path),
        format_name="pytorch",
        device="cpu",
        backend=model,
        torch_module=model,
        torch_module_ref=torch,
    )
    input_tensor = torch.randn(1, 4)
    return loaded, input_tensor


def build_torchscript_model_file(tmp_path: Path) -> Path:
    model = TinyLinearModel().eval()
    model_path = tmp_path / "tiny_model.pt"
    traced = torch.jit.trace(model, torch.randn(1, 4))
    traced.save(str(model_path))
    return model_path

