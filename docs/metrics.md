# Metrics

ModelBench reports inference-focused metrics that help compare deployability, not just quality.

## Latency

- `p50_ms`, `p95_ms`, `p99_ms`, `mean_ms`, and `std_ms`
- Warmup runs are executed before timing to reduce one-time initialization noise.
- CUDA measurements synchronize before and after each forward pass.

## Throughput

- `samples_per_second`
- Measured in a sustained loop over `throughput_duration_seconds`.

## Memory

- `peak_memory_mb`
- CPU uses RSS delta and `tracemalloc` peak.
- CUDA uses `torch.cuda.max_memory_allocated()`.
- MPS uses `torch.mps.current_allocated_memory()`.

## Model Size

- `model_size_mb`
- `param_count`
- `non_trainable_param_count`

## FLOPs

- `flops_per_inference`
- `gflops_per_inference`
- Uses `fvcore` first and `thop` as a fallback.
- Unsupported models warn and skip the metric instead of failing the run.

