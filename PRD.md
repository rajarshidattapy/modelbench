# ModelBench — Product Requirements Document

**Version:** 0.1.0  
**Status:** Draft  
**Audience:** Codex / autonomous coding agent  

---

## 1. Problem Statement

ML practitioners routinely select models by a single metric — accuracy. This fails in production:

- A 99%-accurate model with 400ms P99 latency may be unusable in a real-time API.
- A large model that exceeds GPU VRAM forces expensive hardware.
- A model with 98% accuracy but 10× the throughput often delivers more value.

There is no standardized, framework-agnostic tool that benchmarks ML models across the axes that production deployments actually care about. **ModelBench fills that gap.**

---

## 2. Product Vision

ModelBench is a CLI-first, open-source Python benchmarking framework. Given any trained model and a dataset, it produces a structured report covering latency, throughput, memory, model size, parameter count, and inference efficiency. Engineers use it to answer: *"Which model is actually best for my use case?"*

---

## 3. Goals & Non-Goals

### Goals
- Benchmark any PyTorch or ONNX model without requiring code changes to the model itself.
- Report: latency (P50/P95/P99), throughput (samples/sec), peak memory (MB), model size on disk (MB), parameter count, FLOPs per inference.
- Support CPU, CUDA, and MPS (Apple Silicon) devices.
- Produce machine-readable JSON + human-readable Markdown/terminal reports.
- Compare multiple models side-by-side in a single run.
- Ship a clean CLI and a Python API.

### Non-Goals (v0.1)
- No training-time benchmarking (inference only).
- No TensorFlow or JAX support (add in v0.2).
- No distributed/multi-GPU benchmarking.
- No cloud deployment cost estimation.
- No browser UI or hosted dashboard.

---

## 4. Target Users

| Persona | Need |
|---|---|
| ML Engineer | Compare two fine-tuned models before deploying to a REST API |
| Platform/Infra Engineer | Enforce latency SLOs on model upgrades |
| Researcher | Report efficiency metrics alongside accuracy in papers |
| OSS Contributor | Extend with new metric collectors or exporters |

---

## 5. Architecture

```
modelbench/
├── cli.py                  # Click-based CLI entry point
├── runner.py               # Orchestrates benchmark runs
├── loaders/
│   ├── base.py             # Abstract ModelLoader
│   ├── pytorch.py          # Loads nn.Module / TorchScript
│   └── onnx.py             # Loads ONNX models via onnxruntime
├── metrics/
│   ├── base.py             # Abstract MetricCollector
│   ├── latency.py          # Wall-clock latency with warmup
│   ├── throughput.py       # Samples/sec under sustained load
│   ├── memory.py           # Peak RSS + GPU memory (pynvml)
│   ├── model_size.py       # On-disk size, parameter count
│   └── flops.py            # FLOPs via fvcore or thop
├── reporters/
│   ├── base.py             # Abstract Reporter
│   ├── json_reporter.py    # Structured JSON output
│   ├── markdown_reporter.py# GitHub-friendly Markdown table
│   └── terminal_reporter.py# Rich-formatted terminal output
├── config.py               # Pydantic config schema
└── compare.py              # Side-by-side model comparison logic
```

---

## 6. Core Data Structures

```python
# config.py
from pydantic import BaseModel
from typing import List, Optional, Literal

class BenchmarkConfig(BaseModel):
    model_paths: List[str]            # One or more model file paths
    model_format: Literal["pytorch", "onnx", "torchscript"]
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    batch_sizes: List[int] = [1, 8, 32]
    warmup_runs: int = 10
    benchmark_runs: int = 100
    input_shape: List[int]            # e.g. [3, 224, 224] for image models
    dtype: str = "float32"
    output_dir: str = "./modelbench_results"
    metrics: List[str] = ["latency", "throughput", "memory", "model_size", "flops"]
    export_formats: List[str] = ["json", "markdown"]

class MetricResult(BaseModel):
    name: str
    value: float
    unit: str
    metadata: dict = {}

class ModelBenchmarkResult(BaseModel):
    model_path: str
    device: str
    batch_size: int
    metrics: List[MetricResult]
    timestamp: str
    duration_seconds: float

class BenchmarkReport(BaseModel):
    config: BenchmarkConfig
    results: List[ModelBenchmarkResult]  # one per model × batch_size combo
    comparison: Optional[dict] = None    # populated when >1 model
```

---

## 7. Metric Specifications

### 7.1 Latency (`latency.py`)
- Measure wall-clock time per forward pass.
- Warmup `config.warmup_runs` iterations before timing (eliminates JIT/CUDA graph init noise).
- Record `config.benchmark_runs` timed iterations.
- Report: `p50_ms`, `p95_ms`, `p99_ms`, `mean_ms`, `std_ms`.
- For CUDA: use `torch.cuda.synchronize()` before/after each timed call.
- Implementation pattern:
  ```python
  import time, numpy as np
  latencies = []
  for _ in range(config.benchmark_runs):
      if device == "cuda":
          torch.cuda.synchronize()
      t0 = time.perf_counter()
      model(input_tensor)
      if device == "cuda":
          torch.cuda.synchronize()
      latencies.append((time.perf_counter() - t0) * 1000)
  ```

### 7.2 Throughput (`throughput.py`)
- Sustained inference loop for 30 seconds.
- Count total samples processed; report `samples_per_second`.
- Run at each batch size in `config.batch_sizes`.

### 7.3 Memory (`memory.py`)
- **CPU:** `tracemalloc` peak + `psutil.Process().memory_info().rss` delta.
- **CUDA:** `torch.cuda.max_memory_allocated()` reset before run, captured after.
- **MPS:** Use `torch.mps.current_allocated_memory()`.
- Report: `peak_memory_mb`.

### 7.4 Model Size (`model_size.py`)
- On-disk size: `os.path.getsize(model_path) / 1e6` → `model_size_mb`.
- Parameter count (PyTorch): `sum(p.numel() for p in model.parameters())` → `param_count`.
- Non-trainable params: `sum(p.numel() for p in model.parameters() if not p.requires_grad)`.

### 7.5 FLOPs (`flops.py`)
- Primary: use `fvcore.nn.FlopCountAnalysis` (PyTorch models).
- Fallback: use `thop.profile`.
- Report: `flops_per_inference`, `gflops_per_inference`.
- If FLOPs cannot be computed (dynamic graphs), emit a warning and skip — do not fail the run.

---

## 8. CLI Interface

```
modelbench run \
  --model resnet50.pt efficientnet.pt \
  --format pytorch \
  --device cuda \
  --batch-sizes 1 8 32 \
  --input-shape 3 224 224 \
  --warmup 10 \
  --runs 100 \
  --output ./results \
  --export json markdown
```

```
modelbench compare --results ./results/run_20240601/ --sort throughput
```

```
modelbench list-metrics     # prints available metric collectors
modelbench version
```

Full CLI spec via Click:
- `run` — primary command, maps 1:1 to `BenchmarkConfig`.
- `compare` — loads existing JSON results, re-renders comparison table.
- `list-metrics` — introspects registered `MetricCollector` subclasses.

---

## 9. Python API

```python
from modelbench import ModelBench, BenchmarkConfig

config = BenchmarkConfig(
    model_paths=["resnet50.pt", "efficientnet.pt"],
    model_format="pytorch",
    device="cuda",
    batch_sizes=[1, 8, 32],
    input_shape=[3, 224, 224],
)

bench = ModelBench(config)
report = bench.run()

print(report.results[0].metrics)  # list of MetricResult
report.export("json", "./results")
report.export("markdown", "./results")
```

---

## 10. Output Formats

### JSON (`results/<model>_<timestamp>.json`)
Full `BenchmarkReport` serialized via Pydantic `.model_dump()`.

### Markdown
GitHub-renderable table. Example:

```markdown
## ModelBench Report — 2024-06-01T12:00:00

| Model          | Batch | P50 (ms) | P99 (ms) | Throughput (s/s) | Peak Mem (MB) | Params (M) | GFLOPs |
|----------------|-------|----------|----------|------------------|---------------|------------|--------|
| resnet50.pt    | 1     | 8.2      | 10.1     | 121              | 420           | 25.6       | 4.1    |
| efficientnet.pt| 1     | 5.1      | 6.3      | 194              | 210           | 5.3        | 0.4    |
```

### Terminal
Rich-formatted tables with color-coded best/worst per column.

---

## 11. Comparison Logic (`compare.py`)

When `len(config.model_paths) > 1`:
- Normalize all metric values to [0, 1] per metric across models.
- Compute a configurable weighted score:
  ```python
  DEFAULT_WEIGHTS = {
      "throughput": 0.30,
      "p99_latency": 0.25,
      "peak_memory_mb": 0.20,
      "model_size_mb": 0.15,
      "flops": 0.10,
  }
  ```
- Rank models; emit `comparison.winner` + per-metric winners in the JSON report.
- Weights are user-overridable via `--weights` CLI flag (JSON string) or `BenchmarkConfig.comparison_weights`.

---

## 12. Extensibility

### Custom Metric Collector
```python
from modelbench.metrics.base import MetricCollector, MetricResult

class MyMetric(MetricCollector):
    name = "my_metric"

    def collect(self, model, input_tensor, config) -> MetricResult:
        value = ...  # your measurement
        return MetricResult(name="my_metric", value=value, unit="ms")
```
Register via entry point in `pyproject.toml`:
```toml
[project.entry-points."modelbench.metrics"]
my_metric = "mypkg.metrics:MyMetric"
```

### Custom Reporter
Subclass `modelbench.reporters.base.Reporter`, implement `render(report: BenchmarkReport) -> str`, register similarly.

---

## 13. Dependencies

```toml
[project]
name = "modelbench"
requires-python = ">=3.9"

dependencies = [
    "torch>=2.0",
    "onnxruntime>=1.16",
    "pydantic>=2.0",
    "click>=8.1",
    "rich>=13.0",
    "psutil>=5.9",
    "numpy>=1.24",
    "fvcore>=0.1.5",     # FLOPs (optional, graceful fallback)
    "thop>=0.1.1",       # FLOPs fallback
    "pynvml>=11.0",      # CUDA memory (optional, skip on CPU)
]

[project.optional-dependencies]
dev = ["pytest", "pytest-cov", "ruff", "mypy"]
```

All optional deps (fvcore, pynvml) must fail gracefully with a warning — never a hard crash.

---

## 14. Repository Layout

```
modelbench/
├── modelbench/          # Python package (see §5)
├── tests/
│   ├── unit/            # per-collector unit tests with mock models
│   └── integration/     # end-to-end CLI + API tests
├── examples/
│   ├── resnet_benchmark.py
│   └── onnx_comparison.py
├── docs/
│   └── metrics.md       # Metric definitions and formulas
├── pyproject.toml
├── README.md
└── .github/workflows/ci.yml
```

---

## 15. Testing Requirements

- Unit tests for every `MetricCollector` using a minimal `nn.Linear` model (fast, no GPU required).
- Integration test: run full `modelbench run` on CPU with a dummy model; assert JSON output schema matches `BenchmarkReport`.
- Parametrize tests across `batch_sizes = [1, 4]` to catch shape bugs.
- CI: GitHub Actions, Python 3.9 + 3.11, CPU-only (no GPU runners required for CI).
- Coverage target: ≥80%.

---

## 16. Implementation Order for Codex

Execute in this order. Do not move to the next step until the current step passes tests.

1. **Scaffold** — `pyproject.toml`, package `__init__.py`, `config.py` with Pydantic models.
2. **Loaders** — `base.py` abstract class → `pytorch.py` → `onnx.py`.
3. **Metrics** — `base.py` → `latency.py` → `throughput.py` → `memory.py` → `model_size.py` → `flops.py`.
4. **Runner** — `runner.py` wiring config → loaders → metric collectors → `BenchmarkReport`.
5. **Reporters** — `json_reporter.py` → `markdown_reporter.py` → `terminal_reporter.py`.
6. **Compare** — `compare.py` normalization + scoring + winner detection.
7. **CLI** — `cli.py` with Click commands `run`, `compare`, `list-metrics`, `version`.
8. **Tests** — unit tests per collector, integration test for full run.
9. **Examples** — `examples/resnet_benchmark.py` using torchvision ResNet50 + EfficientNet.
10. **README** — quick-start, install, CLI reference, Python API, extending guide.

---

## 17. Acceptance Criteria

- `modelbench run --model a.pt b.pt --format pytorch --device cpu --input-shape 3 224 224` produces valid JSON and Markdown output without errors.
- JSON output parses against `BenchmarkReport` Pydantic schema with no validation errors.
- Latency P99 values are stable within ±5% across two identical runs (same model, same hardware).
- FLOPs metric fails gracefully (warning to stderr, metric omitted) on an unsupported dynamic model.
- All 80%+ unit test coverage on CPU in CI.
- `pip install modelbench` + single CLI command is the full install + run path.