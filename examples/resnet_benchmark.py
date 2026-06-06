"""Benchmark torchvision ResNet50 and EfficientNet from the Python API.

Requires:
    pip install torchvision
"""

from __future__ import annotations

from pathlib import Path

import torch
from torchvision.models import efficientnet_b0, resnet50

from modelbench import BenchmarkConfig, ModelBench


def save_torchscript(model: torch.nn.Module, destination: Path) -> str:
    model.eval()
    traced = torch.jit.trace(model, torch.randn(1, 3, 224, 224))
    traced.save(str(destination))
    return str(destination)


def main() -> None:
    examples_dir = Path("./examples/artifacts")
    examples_dir.mkdir(parents=True, exist_ok=True)

    resnet_path = save_torchscript(resnet50(weights=None), examples_dir / "resnet50.pt")
    efficientnet_path = save_torchscript(
        efficientnet_b0(weights=None),
        examples_dir / "efficientnet_b0.pt",
    )

    config = BenchmarkConfig(
        model_paths=[resnet_path, efficientnet_path],
        model_format="pytorch",
        device="cpu",
        batch_sizes=[1, 8],
        input_shape=[3, 224, 224],
        warmup_runs=2,
        benchmark_runs=5,
        throughput_duration_seconds=1.0,
        output_dir="./examples/results",
        export_formats=["json", "markdown", "terminal"],
    )

    bench = ModelBench(config)
    report = bench.run()
    paths = bench.export(report)
    print(paths)


if __name__ == "__main__":
    main()

