"""Benchmark multiple ONNX models from the Python API."""

from __future__ import annotations

from modelbench import BenchmarkConfig, ModelBench


def main() -> None:
    config = BenchmarkConfig(
        model_paths=["./models/model_a.onnx", "./models/model_b.onnx"],
        model_format="onnx",
        device="cpu",
        batch_sizes=[1, 8, 32],
        input_shape=[3, 224, 224],
        warmup_runs=5,
        benchmark_runs=20,
        throughput_duration_seconds=2.0,
        output_dir="./onnx_results",
    )

    bench = ModelBench(config)
    report = bench.run()
    print(report.comparison)
    bench.export(report)


if __name__ == "__main__":
    main()

