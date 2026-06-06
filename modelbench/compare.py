"""Comparison and ranking logic for benchmark results."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from modelbench.config import DEFAULT_COMPARISON_WEIGHTS, BenchmarkReport, ModelBenchmarkResult


LOWER_IS_BETTER = {"p50_ms", "p95_ms", "p99_ms", "mean_ms", "std_ms", "peak_memory_mb", "model_size_mb"}


def build_comparison(
    results: List[ModelBenchmarkResult],
    weights: Dict[str, float] | None = None,
) -> Dict[str, object]:
    weights = weights or dict(DEFAULT_COMPARISON_WEIGHTS)
    grouped = _group_results_by_batch_size(results)
    per_batch = {}
    overall_scores = defaultdict(float)
    overall_counts = defaultdict(int)

    for batch_size, batch_results in grouped.items():
        scored_rows = []
        metric_names = sorted({metric.name for result in batch_results for metric in result.metrics})
        metric_bounds = _metric_bounds(batch_results, metric_names)

        for result in batch_results:
            metric_map = result.metric_map()
            normalized = {}
            score = 0.0
            for metric_name, weight in weights.items():
                if metric_name not in metric_map or metric_name not in metric_bounds:
                    continue
                normalized_value = _normalize(metric_name, metric_map[metric_name].value, metric_bounds[metric_name])
                normalized[metric_name] = normalized_value
                score += normalized_value * weight

            model_name = Path(result.model_path).name
            scored_rows.append({"model": model_name, "score": score, "normalized": normalized})
            overall_scores[model_name] += score
            overall_counts[model_name] += 1

        scored_rows.sort(key=lambda row: row["score"], reverse=True)
        metric_winners = _metric_winners(batch_results)
        per_batch[str(batch_size)] = {
            "winner": scored_rows[0]["model"] if scored_rows else None,
            "ranking": scored_rows,
            "metric_winners": metric_winners,
        }

    overall_ranking = sorted(
        (
            {"model": model, "score": overall_scores[model] / max(overall_counts[model], 1)}
            for model in overall_scores
        ),
        key=lambda row: row["score"],
        reverse=True,
    )
    return {
        "winner": overall_ranking[0]["model"] if overall_ranking else None,
        "weights": weights,
        "by_batch_size": per_batch,
        "ranking": overall_ranking,
    }


def load_reports_from_path(path: str) -> List[BenchmarkReport]:
    candidate = Path(path)
    if candidate.is_file():
        return [BenchmarkReport.model_validate_json(candidate.read_text(encoding="utf-8"))]

    reports = []
    for report_path in sorted(candidate.glob("*.json")):
        reports.append(BenchmarkReport.model_validate_json(report_path.read_text(encoding="utf-8")))
    return reports


def merge_reports(reports: Iterable[BenchmarkReport]) -> BenchmarkReport:
    reports = list(reports)
    if not reports:
        raise ValueError("No reports found to compare")

    base_config = reports[0].config
    merged_results = [result for report in reports for result in report.results]
    comparison = build_comparison(merged_results, base_config.comparison_weights)
    return BenchmarkReport(config=base_config, results=merged_results, comparison=comparison)


def _group_results_by_batch_size(
    results: List[ModelBenchmarkResult],
) -> Dict[int, List[ModelBenchmarkResult]]:
    grouped = defaultdict(list)
    for result in results:
        grouped[result.batch_size].append(result)
    return dict(grouped)


def _metric_bounds(
    results: List[ModelBenchmarkResult],
    metric_names: List[str],
) -> Dict[str, Tuple[float, float]]:
    bounds = {}
    for metric_name in metric_names:
        values = [
            result.metric_map()[metric_name].value
            for result in results
            if metric_name in result.metric_map()
        ]
        if values:
            bounds[metric_name] = (min(values), max(values))
    return bounds


def _normalize(metric_name: str, value: float, bounds: Tuple[float, float]) -> float:
    minimum, maximum = bounds
    if maximum == minimum:
        return 1.0
    normalized = (value - minimum) / (maximum - minimum)
    if metric_name in LOWER_IS_BETTER:
        return 1.0 - normalized
    return normalized


def _metric_winners(results: List[ModelBenchmarkResult]) -> Dict[str, str]:
    winners = {}
    metric_names = sorted({metric.name for result in results for metric in result.metrics})
    for metric_name in metric_names:
        candidates = []
        for result in results:
            metric_map = result.metric_map()
            if metric_name in metric_map:
                candidates.append((Path(result.model_path).name, metric_map[metric_name].value))
        if not candidates:
            continue
        reverse = metric_name not in LOWER_IS_BETTER
        candidates.sort(key=lambda item: item[1], reverse=reverse)
        winners[metric_name] = candidates[0][0]
    return winners

