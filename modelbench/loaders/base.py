"""Abstract loader contracts and runtime wrappers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class LoadedModel(ABC):
    """A common inference wrapper returned by every loader."""

    model_path: str
    format_name: str
    device: str
    backend: Any

    @abstractmethod
    def infer(self, input_data: Any) -> Any:
        raise NotImplementedError

    def synchronize(self) -> None:
        return None

    def parameter_count(self) -> Optional[int]:
        return None

    def non_trainable_parameter_count(self) -> Optional[int]:
        return None

    def reset_peak_memory(self) -> None:
        return None

    def peak_memory_bytes(self) -> Optional[int]:
        return None


class ModelLoader(ABC):
    """Loads a model file and prepares batch-shaped inputs."""

    format_name: str

    @abstractmethod
    def load(self, model_path: str, device: str) -> LoadedModel:
        raise NotImplementedError

    @abstractmethod
    def make_input(self, batch_size: int, input_shape: list[int], dtype: str, device: str) -> Any:
        raise NotImplementedError
