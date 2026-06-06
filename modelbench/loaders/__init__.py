"""Model loaders for supported formats."""

from modelbench.loaders.base import LoadedModel, ModelLoader
from modelbench.loaders.onnx import ONNXModelLoader
from modelbench.loaders.pytorch import PyTorchModelLoader

__all__ = ["LoadedModel", "ModelLoader", "ONNXModelLoader", "PyTorchModelLoader"]

