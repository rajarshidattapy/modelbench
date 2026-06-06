# ModelBench

Benchmark models beyond accuracy.

ModelBench is a CLI-first Python framework for comparing trained models on the metrics that matter in production: latency, throughput, memory, model size, parameter count, and inference efficiency.

## Install

```bash
pip install modelbench
```

For development:

```bash
pip install -e .[dev]
```

## Quick Start

```bash
modelbench run \
  --model ./models/resnet50.pt \
  --model ./models/efficientnet.pt \
  --format pytorch \
  --device cpu \
  --batch-sizes 1 \
  --batch-sizes 8 \
  --input-shape 3 \
  --input-shape 224 \
  --input-shape 224 \
  --warmup 10 \
  --runs 100 \
  --output ./results \
  --export json \
  --export markdown
```

This produces:

- JSON reports for machines and downstream tooling
- Markdown reports for GitHub issues, PRs, and docs
- Optional terminal summaries with Rich formatting

## Python API

```python
from modelbench import BenchmarkConfig, ModelBench

config = BenchmarkConfig(
    model_paths=["resnet50.pt", "efficientnet.pt"],
    model_format="pytorch",
    device="cpu",
    batch_sizes=[1, 8, 32],
    input_shape=[3, 224, 224],
)

report = ModelBench(config).run()
print(report.results[0].metrics)
report.export("json", "./results")
report.export("markdown", "./results")
```

## CLI Commands

```bash
modelbench run ...
modelbench compare --results ./results --sort throughput
modelbench list-metrics
modelbench version
```

## Supported Metrics

- Latency: `p50_ms`, `p95_ms`, `p99_ms`, `mean_ms`, `std_ms`
- Throughput: `samples_per_second`
- Memory: `peak_memory_mb`
- Model size: `model_size_mb`, `param_count`, `non_trainable_param_count`
- FLOPs: `flops_per_inference`, `gflops_per_inference`

See [docs/metrics.md](docs/metrics.md) for details.

## Extending

Add custom metrics by subclassing `MetricCollector` and registering them via entry points:

```toml
[project.entry-points."modelbench.metrics"]
my_metric = "mypkg.metrics:MyMetric"
```

## Development

```bash
pytest
ruff check .
```

Because the most accurate model is not always the best model.
