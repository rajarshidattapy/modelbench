from __future__ import annotations

import numpy as np

from modelbench.loaders.onnx import ONNXModelLoader


class FakeInput:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeSession:
    def __init__(self, path: str, providers: list[str]) -> None:
        self.path = path
        self.providers = providers

    def get_inputs(self):
        return [FakeInput("input_tensor")]

    def run(self, _unused, feed_dict):
        return [feed_dict["input_tensor"]]


class FakeORT:
    def __init__(self, providers):
        self.providers = providers

    def get_available_providers(self):
        return self.providers

    def InferenceSession(self, path, providers):
        return FakeSession(path, providers)


def test_onnx_loader_make_input_and_load(tmp_path):
    model_path = tmp_path / "tiny.onnx"
    model_path.write_bytes(b"fake")

    loader = ONNXModelLoader()
    loader.ort = FakeORT(["CPUExecutionProvider"])
    loaded = loader.load(str(model_path), "cpu")
    input_data = loader.make_input(batch_size=2, input_shape=[4], dtype="float32", device="cpu")

    assert loaded.input_name == "input_tensor"
    assert loaded.infer(input_data)[0].shape == (2, 4)
    assert input_data.dtype == np.float32


def test_onnx_loader_provider_selection():
    loader = ONNXModelLoader()
    loader.ort = FakeORT(["CUDAExecutionProvider", "CPUExecutionProvider", "CoreMLExecutionProvider"])

    assert loader._providers_for_device("cpu") == ["CPUExecutionProvider"]
    assert loader._providers_for_device("cuda") == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert loader._providers_for_device("mps") == ["CoreMLExecutionProvider", "CPUExecutionProvider"]
