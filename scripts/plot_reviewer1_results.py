#!/usr/bin/env python3
"""Plot and summarize the ablation results in resources/results/reviewer1.

The reviewer CSV files are experiment indexes: many of their metric columns are
empty because the measurements live in the XML or CSV named by result_filename.
This script joins those files before producing tables, plots, and a Markdown
findings report.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "resources" / "results" / "reviewer1"
METRICS = ("accuracy", "precision", "recall", "f_score")
PLOT_METRICS = ("accuracy", "f_score")
DISPLAY_METRIC = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f_score": "F-Score",
}
PARAMETERS = (
    "lm_window_length",
    "lm_slide_length",
    "lm_top_split",
    "lm_skip",
    "lm_stem_length",
    "lm_prune",
    "lm_likelihood_weight",
    "lm_prior_weight",
    "lm_length_penalty",
    "sentencepiece_model_type",
    "sentencepiece_vocab_size",
    "sentencepiece_character_coverage",
)
DENSITY_SAMPLE_SIZE = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join, analyze, and plot reviewer1 ablation results."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Directory containing comment1-*.csv (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: <input-dir>/plots)",
    )
    parser.add_argument(
        "--density-sample-size",
        type=int,
        default=DENSITY_SAMPLE_SIZE,
        help=(
            "Maximum non-empty corpus sentences used for each graph-density "
            f"estimate (default: {DENSITY_SAMPLE_SIZE})"
        ),
    )
    return parser.parse_args()


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def direct_xml_number(root: ET.Element, tags: Sequence[str]) -> float | None:
    for tag in tags:
        for element in root.findall(f"./{tag}"):
            number = finite_float(element.text)
            if number is not None:
                return number
    return None


def metrics_from_xml(filename: Path) -> dict[str, float | None]:
    try:
        root = ET.parse(filename).getroot()
    except (ET.ParseError, OSError):
        return {}
    return {
        "accuracy": direct_xml_number(root, ("ACCURACY", "TRUE_RATE")),
        "precision": direct_xml_number(root, ("PRECISION",)),
        "recall": direct_xml_number(root, ("RECALL",)),
        "f_score": direct_xml_number(root, ("F-SCORE", "F1-MEASURE")),
    }


def metrics_from_csv(filename: Path) -> dict[str, float | None]:
    try:
        with filename.open("r", encoding="utf-8-sig", newline="") as stream:
            row = next(csv.DictReader(stream), None)
    except (OSError, csv.Error):
        return {}
    if not row:
        return {}
    return {
        "accuracy": finite_float(row.get("accuracy")),
        "precision": finite_float(row.get("precision")),
        "recall": finite_float(row.get("recall")),
        "f_score": finite_float(row.get("f_score") or row.get("f1")),
    }


def resolve_result_file(raw_filename: str, input_dir: Path) -> Path | None:
    if not raw_filename.strip():
        return None
    raw_path = Path(raw_filename)
    candidates = (
        raw_path,
        PROJECT_ROOT / raw_path,
        input_dir / raw_path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def metric_from_row(row: dict[str, str], metric: str) -> float | None:
    if metric == "f_score":
        return finite_float(row.get("f_score") or row.get("f1"))
    return finite_float(row.get(metric))


def load_results(input_dir: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    files = sorted(input_dir.glob("comment1-*.csv"))
    files = [
        filename
        for filename in files
        if filename.name != "comment1-all-evaluations.csv"
    ]
    rows: list[dict[str, Any]] = []
    for filename in files:
        with filename.open("r", encoding="utf-8-sig", newline="") as stream:
            for source_row in csv.DictReader(stream):
                if not source_row.get("task") or not source_row.get("method"):
                    continue
                row: dict[str, Any] = dict(source_row)
                row["source_csv"] = str(filename.relative_to(PROJECT_ROOT))
                result_file = resolve_result_file(
                    source_row.get("result_filename", ""), input_dir
                )
                artifact_metrics: dict[str, float | None] = {}
                if result_file and result_file.suffix.lower() == ".xml":
                    artifact_metrics = metrics_from_xml(result_file)
                elif result_file and result_file.suffix.lower() == ".csv":
                    artifact_metrics = metrics_from_csv(result_file)

                sources = set()
                for metric in METRICS:
                    value = metric_from_row(source_row, metric)
                    if value is not None:
                        sources.add("index_csv")
                    else:
                        value = artifact_metrics.get(metric)
                        if value is not None:
                            sources.add("result_file")
                    row[metric] = value
                row["resolved_result_file"] = str(result_file or "")
                row["metric_source"] = "+".join(sorted(sources)) or "missing"
                rows.append(row)
    return rows, files


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return value


def write_csv(filename: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with filename.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fields})


def values(rows: Iterable[dict[str, Any]], metric: str) -> list[float]:
    return [
        float(row[metric])
        for row in rows
        if row.get(metric) is not None and math.isfinite(float(row[metric]))
    ]


def metric_is_comparable(row: dict[str, Any], metric: str) -> bool:
    if row.get(metric) is None:
        return False
    return not (
        metric == "f_score"
        and row.get("task") in {"ner", "pos", "sentiment"}
        and (
            row.get("precision") is None
            or row.get("recall") is None
        )
    )


def comparable_values(
    rows: Iterable[dict[str, Any]],
    metric: str,
) -> list[float]:
    return [
        float(row[metric])
        for row in rows
        if metric_is_comparable(row, metric)
        and math.isfinite(float(row[metric]))
    ]


def summarize_methods(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["task"], row["method"])].append(row)

    summary: list[dict[str, Any]] = []
    for (task, method), group in sorted(groups.items()):
        output: dict[str, Any] = {
            "task": task,
            "method": method,
            "variants": len(group),
        }
        for metric in METRICS:
            scores = comparable_values(group, metric)
            output[f"{metric}_n"] = len(scores)
            output[f"{metric}_mean"] = statistics.fmean(scores) if scores else None
            output[f"{metric}_std"] = (
                statistics.stdev(scores) if len(scores) > 1 else 0.0 if scores else None
            )
            output[f"{metric}_min"] = min(scores) if scores else None
            output[f"{metric}_max"] = max(scores) if scores else None
        summary.append(output)
    return summary


def best_variants(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    tasks = sorted({row["task"] for row in rows})
    for task in tasks:
        task_rows = [row for row in rows if row["task"] == task]
        for metric in METRICS:
            if task == "analogy" and metric == "f_score":
                continue
            candidates = [
                row
                for row in task_rows
                if metric_is_comparable(row, metric)
            ]
            if not candidates:
                continue
            best = max(candidates, key=lambda row: float(row[metric]))
            output.append(
                {
                    "task": task,
                    "metric": metric,
                    "value": best[metric],
                    "method": best["method"],
                    "variant_id": best.get("variant_id", ""),
                    **{parameter: best.get(parameter, "") for parameter in PARAMETERS},
                    "result_filename": best.get("result_filename", ""),
                }
            )
    return output


def selected_metric_best_methods(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for task in sorted({row["task"] for row in rows}):
        metric = "accuracy" if task == "analogy" else "f_score"
        methods = sorted({row["method"] for row in rows if row["task"] == task})
        for method in methods:
            candidates = [
                row
                for row in rows
                if row["task"] == task
                and row["method"] == method
                and row.get(metric) is not None
            ]
            if not candidates:
                continue

            best = max(candidates, key=lambda row: float(row[metric]))
            if task in {"ner", "pos", "sentiment"}:
                comparable_candidates = [
                    row
                    for row in candidates
                    if row.get("precision") is not None
                    and row.get("recall") is not None
                ]
            else:
                comparable_candidates = candidates
            best_comparable = (
                max(comparable_candidates, key=lambda row: float(row[metric]))
                if comparable_candidates
                else None
            )

            output.append(
                {
                    "task": task,
                    "metric": metric,
                    "method": method,
                    "variant_id": best.get("variant_id", ""),
                    "score": best[metric],
                    "score_status": (
                        "legacy-missing-precision-recall"
                        if task in {"ner", "pos", "sentiment"}
                        and (
                            best.get("precision") is None
                            or best.get("recall") is None
                        )
                        else "current"
                    ),
                    "best_comparable_variant_id": (
                        best_comparable.get("variant_id", "")
                        if best_comparable
                        else ""
                    ),
                    "best_comparable_score": (
                        best_comparable[metric] if best_comparable else None
                    ),
                    "result_filename": best.get("result_filename", ""),
                }
            )
    return output


def parameter_effects(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    method_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        method_groups[(row["task"], row["method"])].append(row)

    output: list[dict[str, Any]] = []
    for (task, method), group in sorted(method_groups.items()):
        for metric in METRICS:
            if task == "analogy" and metric == "f_score":
                continue
            outcome = comparable_values(group, metric)
            outcome_range = max(outcome) - min(outcome) if outcome else 0.0
            for parameter in PARAMETERS:
                level_groups: dict[str, list[float]] = defaultdict(list)
                for row in group:
                    value = row.get(metric)
                    level = str(row.get(parameter, "")).strip()
                    if metric_is_comparable(row, metric) and level:
                        level_groups[level].append(float(value))
                if len(level_groups) < 2:
                    continue
                level_means = {
                    level: statistics.fmean(scores)
                    for level, scores in level_groups.items()
                }
                effect_range = max(level_means.values()) - min(level_means.values())
                best_level, best_mean = max(
                    level_means.items(), key=lambda item: item[1]
                )
                output.append(
                    {
                        "task": task,
                        "method": method,
                        "metric": metric,
                        "parameter": parameter,
                        "levels": len(level_groups),
                        "observations": sum(map(len, level_groups.values())),
                        "effect_range": effect_range,
                        "relative_effect": (
                            effect_range / outcome_range if outcome_range > 0 else 0.0
                        ),
                        "best_level": best_level,
                        "best_level_mean": best_mean,
                    }
                )
    return output


def selected_sensitivity_effects(
    effects: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        row
        for row in effects
        if (
            row["task"] == "analogy"
            and row["metric"] == "accuracy"
        )
        or (
            row["task"] != "analogy"
            and row["metric"] == "f_score"
        )
    ]


def resolve_corpus_file(raw_filename: str, input_dir: Path) -> Path | None:
    if not raw_filename.strip():
        return None
    normalized = raw_filename.replace("//", "/")
    raw_path = Path(normalized)
    candidates = (
        raw_path,
        PROJECT_ROOT / raw_path,
        input_dir / raw_path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def corpus_graph_density(
    filename: Path,
    sample_size: int,
) -> dict[str, Any] | None:
    """Apply the LMDataset density definition to a deterministic corpus sample.

    LMDataset counts every ordered token pair within a sentence, including
    self-pairs, so its total edge weight is the sum of squared sentence
    lengths.  The denominator is V(V-1), where V is the number of distinct
    partition tokens.  Reading the first N non-empty sentences makes repeated
    analysis runs deterministic and avoids loading each large corpus in memory.
    """
    distinct_partitions: set[str] = set()
    total_edges = 0
    total_partitions = 0
    total_distinct_sentence_ngrams = 0
    sampled_sentences = 0
    try:
        with filename.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                partitions = line.split()
                if not partitions:
                    continue
                sampled_sentences += 1
                partition_count = len(partitions)
                total_partitions += partition_count
                total_edges += partition_count * partition_count
                total_distinct_sentence_ngrams += len(set(partitions))
                distinct_partitions.update(partitions)
                if sampled_sentences >= sample_size:
                    break
    except OSError:
        return None

    vocabulary_size = len(distinct_partitions)
    if sampled_sentences == 0 or vocabulary_size < 2:
        return None
    return {
        "graph_density": (
            2.0 * total_edges / (vocabulary_size * (vocabulary_size - 1))
        ),
        "sampled_sentences": sampled_sentences,
        "distinct_partitions": vocabulary_size,
        "total_partitions": total_partitions,
        "total_edges": total_edges,
        "average_sentence_partitions": total_partitions / sampled_sentences,
        "average_distinct_ngrams_per_sentence": (
            total_distinct_sentence_ngrams / sampled_sentences
        ),
    }


def prepare_graph_density_rows(
    rows: Sequence[dict[str, Any]],
    input_dir: Path,
    sample_size: int,
) -> list[dict[str, Any]]:
    if sample_size < 1:
        raise ValueError("--density-sample-size must be at least 1")

    corpus_cache: dict[Path, dict[str, Any] | None] = {}
    direct_density: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        corpus_file = resolve_corpus_file(
            str(row.get("corpus_filename", "")), input_dir
        )
        if corpus_file is None:
            continue
        if corpus_file not in corpus_cache:
            corpus_cache[corpus_file] = corpus_graph_density(
                corpus_file, sample_size
            )
        density = corpus_cache[corpus_file]
        if density is not None:
            direct_density[
                (row["task"], row["method"], str(row.get("variant_id", "")))
            ] = {
                **density,
                "resolved_corpus_file": str(corpus_file),
                "density_basis": "task partition corpus",
            }

    reference_density: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for (task, method, variant_id), density in direct_density.items():
        if task != "morphology":
            reference_density[(method, variant_id)].append(density)

    output: list[dict[str, Any]] = []
    for row in rows:
        task = row["task"]
        method = row["method"]
        variant_id = str(row.get("variant_id", ""))
        metric = "accuracy" if task == "analogy" else "f_score"
        density = direct_density.get((task, method, variant_id))
        if density is None and task == "morphology":
            references = reference_density.get((method, variant_id), [])
            if references:
                density = {
                    "graph_density": statistics.fmean(
                        item["graph_density"] for item in references
                    ),
                    "sampled_sentences": None,
                    "distinct_partitions": None,
                    "total_partitions": None,
                    "total_edges": None,
                    "average_sentence_partitions": statistics.fmean(
                        item["average_sentence_partitions"]
                        for item in references
                    ),
                    "average_distinct_ngrams_per_sentence": statistics.fmean(
                        item["average_distinct_ngrams_per_sentence"]
                        for item in references
                    ),
                    "resolved_corpus_file": "",
                    "density_basis": (
                        "mean reference-corpus density for the same LM variant"
                    ),
                }

        score = row.get(metric)
        complete_current_metrics = (
            task not in {"ner", "pos", "sentiment"}
            or (
                row.get("precision") is not None
                and row.get("recall") is not None
            )
        )
        included = (
            density is not None
            and score is not None
            and complete_current_metrics
        )
        output.append(
            {
                "task": task,
                "method": method,
                "variant_id": variant_id,
                "lm_window_length": row.get("lm_window_length", ""),
                "metric": metric,
                "score": score,
                "graph_density": (
                    density.get("graph_density") if density is not None else None
                ),
                "density_basis": (
                    density.get("density_basis", "") if density is not None else ""
                ),
                "sampled_sentences": (
                    density.get("sampled_sentences")
                    if density is not None
                    else None
                ),
                "distinct_partitions": (
                    density.get("distinct_partitions")
                    if density is not None
                    else None
                ),
                "total_partitions": (
                    density.get("total_partitions")
                    if density is not None
                    else None
                ),
                "total_edges": (
                    density.get("total_edges") if density is not None else None
                ),
                "average_sentence_partitions": (
                    density.get("average_sentence_partitions")
                    if density is not None
                    else None
                ),
                "average_distinct_ngrams_per_sentence": (
                    density.get("average_distinct_ngrams_per_sentence")
                    if density is not None
                    else None
                ),
                "included_in_plot": included,
                "metric_status": (
                    "current"
                    if complete_current_metrics
                    else "legacy-missing-precision-recall"
                ),
                "corpus_filename": row.get("corpus_filename", ""),
                "resolved_corpus_file": (
                    density.get("resolved_corpus_file", "")
                    if density is not None
                    else ""
                ),
            }
        )
    return output


def pearson_correlation(x_values: Sequence[float], y_values: Sequence[float]) -> float | None:
    if len(x_values) < 2 or len(x_values) != len(y_values):
        return None
    x_mean = statistics.fmean(x_values)
    y_mean = statistics.fmean(y_values)
    numerator = sum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(x_values, y_values)
    )
    x_sum = sum((x - x_mean) ** 2 for x in x_values)
    y_sum = sum((y - y_mean) ** 2 for y in y_values)
    denominator = math.sqrt(x_sum * y_sum)
    return numerator / denominator if denominator > 0 else None


def graph_density_correlations(
    density_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    tasks = sorted({row["task"] for row in density_rows})
    for task in tasks:
        selected = [
            row
            for row in density_rows
            if row["task"] == task
            and row["included_in_plot"]
            and row.get("graph_density") is not None
            and float(row["graph_density"]) > 0
            and row.get("score") is not None
        ]
        if not selected:
            continue
        densities = [float(row["graph_density"]) for row in selected]
        scores = [float(row["score"]) for row in selected]
        output.append(
            {
                "task": task,
                "metric": selected[0]["metric"],
                "observations": len(selected),
                "pearson_r_density": pearson_correlation(densities, scores),
                "pearson_r_log10_density": pearson_correlation(
                    [math.log10(value) for value in densities], scores
                ),
                "density_min": min(densities),
                "density_max": max(densities),
                "score_min": min(scores),
                "score_max": max(scores),
                "density_basis": (
                    "reference-corpus estimate"
                    if task == "morphology"
                    else "task partition corpus"
                ),
            }
        )
    return output


def average_distinct_ngram_correlations(
    density_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for task in ("analogy", "morphology", "ner", "pos", "sentiment"):
        selected = [
            row
            for row in density_rows
            if row["task"] == task
            and row["included_in_plot"]
            and row.get("average_distinct_ngrams_per_sentence") is not None
            and row.get("score") is not None
        ]
        if not selected:
            continue
        averages = [
            float(row["average_distinct_ngrams_per_sentence"])
            for row in selected
        ]
        scores = [float(row["score"]) for row in selected]
        output.append(
            {
                "task": task,
                "metric": selected[0]["metric"],
                "observations": len(selected),
                "pearson_r": pearson_correlation(averages, scores),
                "average_distinct_ngrams_min": min(averages),
                "average_distinct_ngrams_max": max(averages),
                "score_min": min(scores),
                "score_max": max(scores),
            }
        )
    return output


def status_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        groups[(row["task"], row.get("status", "unknown"))] += 1
    return [
        {"task": task, "status": status, "count": count}
        for (task, status), count in sorted(groups.items())
    ]


def save_figure(figure: Any, base: Path) -> None:
    figure.savefig(
        base.with_suffix(".pdf"),
        format="pdf",
        bbox_inches="tight",
        facecolor="white",
    )


def task_method_colors(plt: Any, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    methods = sorted({row["method"] for row in rows})
    palette = plt.get_cmap("tab10")
    return {method: palette(index % 10) for index, method in enumerate(methods)}


def light_colormap(plt: Any, name: str, stop: float = 0.58) -> Any:
    from matplotlib.colors import LinearSegmentedColormap

    source = plt.get_cmap(name)
    colors = [source(0.03 + (stop - 0.03) * index / 255) for index in range(256)]
    return LinearSegmentedColormap.from_list(f"light_{name}", colors)


def plot_coverage(
    plt: Any,
    rows: Sequence[dict[str, Any]],
    output_dir: Path,
) -> None:
    tasks = sorted({row["task"] for row in rows})
    figure, axis = plt.subplots(figsize=(max(8, len(tasks) * 1.7), 5))
    width = 0.18
    x_positions = list(range(len(tasks)))
    for metric_index, metric in enumerate(PLOT_METRICS):
        ratios = []
        for task in tasks:
            task_rows = [row for row in rows if row["task"] == task]
            if task == "analogy" and metric == "f_score":
                ratios.append(math.nan)
            else:
                count = sum(row.get(metric) is not None for row in task_rows)
                ratios.append(100.0 * count / len(task_rows))
        offset = (metric_index - (len(PLOT_METRICS) - 1) / 2) * width
        axis.bar(
            [position + offset for position in x_positions],
            ratios,
            width,
            label=DISPLAY_METRIC[metric],
        )
    axis.set_xticks(x_positions, [task.upper() for task in tasks])
    axis.set_ylim(0, 105)
    axis.set_ylabel("Variants with a score (%)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(
        ncol=len(PLOT_METRICS),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
    )
    save_figure(figure, output_dir / "metric_coverage")
    plt.close(figure)


def plot_task_distributions(
    plt: Any,
    rows: Sequence[dict[str, Any]],
    output_dir: Path,
) -> None:
    colors = task_method_colors(plt, rows)
    for task in sorted({row["task"] for row in rows}):
        task_rows = [row for row in rows if row["task"] == task]
        if task in {"morphology", "ner", "pos", "sentiment"}:
            available_metrics = (
                ["f_score"] if values(task_rows, "f_score") else []
            )
        else:
            available_metrics = [
                metric
                for metric in PLOT_METRICS
                if values(task_rows, metric)
                and not (task == "analogy" and metric == "f_score")
            ]
        if not available_metrics:
            continue
        figure, axes = plt.subplots(
            1,
            len(available_metrics),
            figsize=(7 * len(available_metrics), 5),
            squeeze=False,
        )
        methods = sorted({row["method"] for row in task_rows})
        for axis, metric in zip(axes[0], available_metrics):
            plotted_methods = []
            distributions = []
            for method in methods:
                scores = comparable_values(
                    [row for row in task_rows if row["method"] == method], metric
                )
                if scores:
                    plotted_methods.append(method)
                    distributions.append(scores)
            boxplot_options = {
                "patch_artist": True,
                "showmeans": True,
                "meanline": True,
            }
            try:
                boxplot = axis.boxplot(
                    distributions,
                    tick_labels=plotted_methods,
                    **boxplot_options,
                )
            except TypeError:
                # Matplotlib versions before tick_labels used labels.
                boxplot = axis.boxplot(
                    distributions,
                    labels=plotted_methods,
                    **boxplot_options,
                )
            for patch, method in zip(boxplot["boxes"], plotted_methods):
                patch.set_facecolor(colors[method])
                patch.set_alpha(0.65)
            axis.tick_params(axis="x", rotation=35)
            axis.set_ylabel(DISPLAY_METRIC[metric])
            axis.set_xlabel(f"{task.upper()} methods")
            axis.grid(axis="y", alpha=0.25)
        save_figure(figure, output_dir / f"distributions_{task}")
        plt.close(figure)


def plot_accuracy_vs_fscore(
    plt: Any,
    rows: Sequence[dict[str, Any]],
    output_dir: Path,
) -> None:
    tasks = sorted(
        {
            row["task"]
            for row in rows
            if row["task"] != "analogy"
            and row.get("accuracy") is not None
            and row.get("f_score") is not None
        }
    )
    if not tasks:
        return
    columns = min(3, len(tasks))
    row_count = math.ceil(len(tasks) / columns)
    figure, axes = plt.subplots(
        row_count, columns, figsize=(6 * columns, 5 * row_count), squeeze=False
    )
    colors = task_method_colors(plt, rows)
    for axis, task in zip([axis for row in axes for axis in row], tasks):
        task_rows = [row for row in rows if row["task"] == task]
        for method in sorted({row["method"] for row in task_rows}):
            points = [
                row
                for row in task_rows
                if row["method"] == method
                and row.get("accuracy") is not None
                and row.get("f_score") is not None
                and metric_is_comparable(row, "f_score")
            ]
            if points:
                axis.scatter(
                    [row["accuracy"] for row in points],
                    [row["f_score"] for row in points],
                    label=method,
                    color=colors[method],
                    alpha=0.75,
                    s=35,
                )
        axis.set_xlabel(f"{task.upper()} — Accuracy")
        axis.set_ylabel("F-Score")
        axis.grid(alpha=0.25)
    for axis in [axis for row in axes for axis in row][len(tasks) :]:
        axis.set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles, labels, ncol=min(3, len(labels)), loc="lower center"
        )
        figure.subplots_adjust(bottom=0.15)
    save_figure(figure, output_dir / "classification_accuracy_vs_fscore")
    plt.close(figure)


def plot_best_method_heatmaps(
    plt: Any,
    rows: Sequence[dict[str, Any]],
    output_dir: Path,
) -> None:
    tasks = sorted({row["task"] for row in rows})
    methods = sorted({row["method"] for row in rows})
    plotted_metrics = [
        metric
        for metric in PLOT_METRICS
        if any(row.get(metric) is not None for row in rows)
    ]
    figure, axes = plt.subplots(
        1,
        len(plotted_metrics),
        figsize=(7 * len(plotted_metrics), 6),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, metric in zip(axes[0], plotted_metrics):
        raw_matrix: list[list[float | None]] = []
        color_matrix: list[list[float]] = []
        for task in tasks:
            row_values = []
            for method in methods:
                scores = (
                    []
                    if task == "analogy" and metric == "f_score"
                    else comparable_values(
                        [
                            row
                            for row in rows
                            if row["task"] == task and row["method"] == method
                        ],
                        metric,
                    )
                )
                row_values.append(max(scores) if scores else None)
            valid = [value for value in row_values if value is not None]
            low, high = (min(valid), max(valid)) if valid else (0.0, 0.0)
            raw_matrix.append(row_values)
            color_matrix.append(
                [
                    (
                        (value - low) / (high - low)
                        if value is not None and high > low
                        else 0.5 if value is not None else math.nan
                    )
                    for value in row_values
                ]
            )
        image = axis.imshow(
            color_matrix,
            cmap=light_colormap(plt, "YlGnBu"),
            vmin=0,
            vmax=1,
            aspect="auto",
        )
        for task_index, row_values in enumerate(raw_matrix):
            for method_index, value in enumerate(row_values):
                text = "—" if value is None else f"{value:.4g}"
                axis.text(
                    method_index,
                    task_index,
                    text,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )
        axis.set_xticks(range(len(methods)), methods, rotation=40, ha="right")
        axis.set_yticks(range(len(tasks)), [task.upper() for task in tasks])
        axis.set_xlabel(f"Best {DISPLAY_METRIC[metric]}")
        figure.colorbar(
            image,
            ax=axis,
            fraction=0.035,
            pad=0.02,
            label="Within-task rank",
        )
    save_figure(figure, output_dir / "best_method_heatmaps")
    plt.close(figure)


def plot_parameter_sensitivity(
    plt: Any,
    effects: Sequence[dict[str, Any]],
    output_dir: Path,
) -> None:
    if not effects:
        return
    row_keys = sorted(
        {
            (row["task"], row["metric"])
            for row in effects
            if row["metric"] in PLOT_METRICS
        }
    )
    parameters = [
        parameter
        for parameter in PARAMETERS
        if any(row["parameter"] == parameter for row in effects)
    ]
    matrix: list[list[float]] = []
    for task, metric in row_keys:
        values_by_parameter: dict[str, list[float]] = defaultdict(list)
        for row in effects:
            if row["task"] == task and row["metric"] == metric:
                values_by_parameter[row["parameter"]].append(row["relative_effect"])
        matrix.append(
            [
                statistics.median(values_by_parameter[parameter])
                if values_by_parameter[parameter]
                else math.nan
                for parameter in parameters
            ]
        )
    figure, axis = plt.subplots(
        figsize=(max(9, len(parameters) * 1.2), max(5, len(row_keys) * 0.5))
    )
    image = axis.imshow(
        matrix,
        cmap=light_colormap(plt, "YlOrRd", stop=0.52),
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    for row_index, row_values in enumerate(matrix):
        for column_index, value in enumerate(row_values):
            if math.isfinite(value):
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )
    axis.set_xticks(
        range(len(parameters)),
        [parameter.removeprefix("lm_") for parameter in parameters],
        rotation=40,
        ha="right",
    )
    axis.set_yticks(
        range(len(row_keys)),
        [
            f"{task.upper()} — {DISPLAY_METRIC[metric]}"
            for task, metric in row_keys
        ],
    )
    figure.colorbar(
        image,
        ax=axis,
        fraction=0.025,
        label="Median grouped-mean range / observed score range",
    )
    save_figure(figure, output_dir / "parameter_sensitivity")
    plt.close(figure)


def plot_lm_graph_density(
    plt: Any,
    density_rows: Sequence[dict[str, Any]],
    correlations: Sequence[dict[str, Any]],
    output_dir: Path,
) -> None:
    from matplotlib.lines import Line2D

    selected_rows = [
        row for row in density_rows if row.get("included_in_plot")
    ]
    if not selected_rows:
        return

    colors = task_method_colors(plt, selected_rows)
    windows = sorted(
        {str(row.get("lm_window_length", "")) for row in selected_rows},
        key=lambda value: finite_float(value) or math.inf,
    )
    marker_choices = ("o", "s", "^", "D", "P", "X", "v")
    markers = {
        window: marker_choices[index % len(marker_choices)]
        for index, window in enumerate(windows)
    }
    correlation_by_task = {row["task"]: row for row in correlations}

    for task in sorted({row["task"] for row in selected_rows}):
        task_rows = [row for row in selected_rows if row["task"] == task]
        metric = task_rows[0]["metric"]
        figure, axis = plt.subplots(figsize=(7.2, 5.2))
        for method in sorted({row["method"] for row in task_rows}):
            method_rows = [row for row in task_rows if row["method"] == method]
            for window in windows:
                points = [
                    row
                    for row in method_rows
                    if str(row.get("lm_window_length", "")) == window
                    and row.get("graph_density") is not None
                    and float(row["graph_density"]) > 0
                ]
                if points:
                    axis.scatter(
                        [float(row["graph_density"]) for row in points],
                        [float(row["score"]) for row in points],
                        color=colors[method],
                        marker=markers[window],
                        alpha=0.72,
                        s=46,
                        edgecolors="white",
                        linewidths=0.35,
                    )

        x_values = [
            math.log10(float(row["graph_density"])) for row in task_rows
        ]
        y_values = [float(row["score"]) for row in task_rows]
        x_mean = statistics.fmean(x_values)
        y_mean = statistics.fmean(y_values)
        x_variance = sum((value - x_mean) ** 2 for value in x_values)
        if x_variance > 0:
            slope = sum(
                (x - x_mean) * (y - y_mean)
                for x, y in zip(x_values, y_values)
            ) / x_variance
            intercept = y_mean - slope * x_mean
            line_x_log = [min(x_values), max(x_values)]
            axis.plot(
                [10 ** value for value in line_x_log],
                [intercept + slope * value for value in line_x_log],
                color="#555555",
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
            )

        axis.set_xscale("log")
        axis.set_xlabel("LM graph density (log scale)")
        axis.set_ylabel(DISPLAY_METRIC[metric])
        axis.grid(alpha=0.22)
        correlation = correlation_by_task.get(task)
        if correlation and correlation.get("pearson_r_log10_density") is not None:
            axis.text(
                0.02,
                0.98,
                (
                    f"Pearson $r$ = "
                    f"{correlation['pearson_r_log10_density']:.3f}; "
                    f"$n$ = {correlation['observations']}"
                ),
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "#cccccc",
                    "alpha": 0.88,
                },
            )

        method_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                color=colors[method],
                label=method,
                markersize=6,
            )
            for method in sorted({row["method"] for row in task_rows})
        ]
        task_windows = [
            window
            for window in windows
            if any(
                str(row.get("lm_window_length", "")) == window
                for row in task_rows
            )
        ]
        window_handles = [
            Line2D(
                [0],
                [0],
                marker=markers[window],
                linestyle="none",
                markerfacecolor="#777777",
                markeredgecolor="white",
                color="#777777",
                label=f"window={window}",
                markersize=6,
            )
            for window in task_windows
        ]
        handles = method_handles + window_handles
        axis.legend(
            handles=handles,
            ncol=min(5, max(1, len(handles))),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            frameon=False,
            fontsize=8,
        )
        figure.subplots_adjust(bottom=0.27)
        save_figure(figure, output_dir / f"lm_graph_density_{task}")
        plt.close(figure)


def plot_lm_graph_density_vs_fscore(
    plt: Any,
    density_rows: Sequence[dict[str, Any]],
    correlations: Sequence[dict[str, Any]],
    output_dir: Path,
) -> None:
    from matplotlib.lines import Line2D

    tasks = [
        task
        for task in ("morphology", "ner", "pos", "sentiment")
        if any(
            row.get("included_in_plot")
            and row["task"] == task
            and row["metric"] == "f_score"
            for row in density_rows
        )
    ]
    if not tasks:
        return

    selected_rows = [
        row
        for row in density_rows
        if row.get("included_in_plot")
        and row["task"] in tasks
        and row["metric"] == "f_score"
    ]
    colors = task_method_colors(plt, selected_rows)
    windows = sorted(
        {str(row.get("lm_window_length", "")) for row in selected_rows},
        key=lambda value: finite_float(value) or math.inf,
    )
    marker_choices = ("o", "s", "^", "D", "P", "X", "v")
    markers = {
        window: marker_choices[index % len(marker_choices)]
        for index, window in enumerate(windows)
    }
    correlation_by_task = {row["task"]: row for row in correlations}

    figure, axes = plt.subplots(2, 2, figsize=(13, 9.5), squeeze=False)
    flat_axes = [axis for axis_row in axes for axis in axis_row]
    for axis, task in zip(flat_axes, tasks):
        task_rows = [row for row in selected_rows if row["task"] == task]
        for method in sorted({row["method"] for row in task_rows}):
            method_rows = [row for row in task_rows if row["method"] == method]
            for window in windows:
                points = [
                    row
                    for row in method_rows
                    if str(row.get("lm_window_length", "")) == window
                    and row.get("graph_density") is not None
                    and float(row["graph_density"]) > 0
                ]
                if points:
                    axis.scatter(
                        [float(row["graph_density"]) for row in points],
                        [float(row["score"]) for row in points],
                        color=colors[method],
                        marker=markers[window],
                        alpha=0.72,
                        s=46,
                        edgecolors="white",
                        linewidths=0.35,
                    )

        x_values = [
            math.log10(float(row["graph_density"])) for row in task_rows
        ]
        y_values = [float(row["score"]) for row in task_rows]
        x_mean = statistics.fmean(x_values)
        y_mean = statistics.fmean(y_values)
        x_variance = sum((value - x_mean) ** 2 for value in x_values)
        if x_variance > 0:
            slope = sum(
                (x - x_mean) * (y - y_mean)
                for x, y in zip(x_values, y_values)
            ) / x_variance
            intercept = y_mean - slope * x_mean
            line_x_log = [min(x_values), max(x_values)]
            axis.plot(
                [10 ** value for value in line_x_log],
                [intercept + slope * value for value in line_x_log],
                color="#555555",
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
            )

        axis.set_xscale("log")
        axis.set_xlabel("LM graph density (log scale)")
        axis.set_ylabel(f"{task.upper()} — F-Measure")
        axis.grid(alpha=0.22)
        correlation = correlation_by_task.get(task)
        if correlation and correlation.get("pearson_r_log10_density") is not None:
            note = (
                f"Pearson $r$ = "
                f"{correlation['pearson_r_log10_density']:.3f}; "
                f"$n$ = {correlation['observations']}"
            )
            axis.text(
                0.02,
                0.98,
                note,
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "#cccccc",
                    "alpha": 0.88,
                },
            )

    for axis in flat_axes[len(tasks):]:
        axis.set_visible(False)

    method_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=colors[method],
            label=method,
            markersize=6,
        )
        for method in sorted({row["method"] for row in selected_rows})
    ]
    window_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[window],
            linestyle="none",
            markerfacecolor="#777777",
            markeredgecolor="white",
            color="#777777",
            label=f"window={window}",
            markersize=6,
        )
        for window in windows
    ]
    handles = method_handles + window_handles
    figure.legend(
        handles=handles,
        ncol=min(6, max(1, len(handles))),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        frameon=False,
        fontsize=9,
    )
    figure.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.98,
        bottom=0.15,
        hspace=0.30,
        wspace=0.25,
    )
    save_figure(figure, output_dir / "lm_graph_density_vs_fscore")
    plt.close(figure)


def plot_average_distinct_ngrams(
    plt: Any,
    density_rows: Sequence[dict[str, Any]],
    correlations: Sequence[dict[str, Any]],
    output_dir: Path,
) -> None:
    from matplotlib.lines import Line2D

    task_filenames = {
        "analogy": "avg-intrinsic",
        "morphology": "avg-morphology",
        "ner": "avg-ner",
        "pos": "avg-pos",
        "sentiment": "avg-sentiment",
    }
    selected_rows = [
        row
        for row in density_rows
        if row.get("included_in_plot")
        and row["task"] in task_filenames
        and row.get("average_distinct_ngrams_per_sentence") is not None
    ]
    if not selected_rows:
        return

    colors = task_method_colors(plt, selected_rows)
    windows = sorted(
        {str(row.get("lm_window_length", "")) for row in selected_rows},
        key=lambda value: finite_float(value) or math.inf,
    )
    marker_choices = ("o", "s", "^", "D", "P", "X", "v")
    markers = {
        window: marker_choices[index % len(marker_choices)]
        for index, window in enumerate(windows)
    }
    correlation_by_task = {row["task"]: row for row in correlations}

    for task, output_name in task_filenames.items():
        task_rows = [row for row in selected_rows if row["task"] == task]
        if not task_rows:
            continue
        metric = task_rows[0]["metric"]
        figure, axis = plt.subplots(figsize=(7.2, 5.2))
        for method in sorted({row["method"] for row in task_rows}):
            method_rows = [row for row in task_rows if row["method"] == method]
            for window in windows:
                points = [
                    row
                    for row in method_rows
                    if str(row.get("lm_window_length", "")) == window
                ]
                if points:
                    axis.scatter(
                        [
                            float(row["average_distinct_ngrams_per_sentence"])
                            for row in points
                        ],
                        [float(row["score"]) for row in points],
                        color=colors[method],
                        marker=markers[window],
                        alpha=0.72,
                        s=46,
                        edgecolors="white",
                        linewidths=0.35,
                    )

        x_values = [
            float(row["average_distinct_ngrams_per_sentence"])
            for row in task_rows
        ]
        y_values = [float(row["score"]) for row in task_rows]
        x_mean = statistics.fmean(x_values)
        y_mean = statistics.fmean(y_values)
        x_variance = sum((value - x_mean) ** 2 for value in x_values)
        if x_variance > 0:
            slope = sum(
                (x - x_mean) * (y - y_mean)
                for x, y in zip(x_values, y_values)
            ) / x_variance
            intercept = y_mean - slope * x_mean
            line_x = [min(x_values), max(x_values)]
            axis.plot(
                line_x,
                [intercept + slope * value for value in line_x],
                color="#555555",
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
            )

        axis.set_xlabel("Average distinct n-grams per sentence")
        axis.set_ylabel(DISPLAY_METRIC[metric])
        axis.grid(alpha=0.22)
        correlation = correlation_by_task.get(task)
        if correlation and correlation.get("pearson_r") is not None:
            note = (
                f"Pearson $r$ = {correlation['pearson_r']:.3f}; "
                f"$n$ = {correlation['observations']}"
            )
            axis.text(
                0.02,
                0.98,
                note,
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "#cccccc",
                    "alpha": 0.88,
                },
            )

        method_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                color=colors[method],
                label=method,
                markersize=6,
            )
            for method in sorted({row["method"] for row in task_rows})
        ]
        task_windows = [
            window
            for window in windows
            if any(
                str(row.get("lm_window_length", "")) == window
                for row in task_rows
            )
        ]
        window_handles = [
            Line2D(
                [0],
                [0],
                marker=markers[window],
                linestyle="none",
                markerfacecolor="#777777",
                markeredgecolor="white",
                color="#777777",
                label=f"window={window}",
                markersize=6,
            )
            for window in task_windows
        ]
        handles = method_handles + window_handles
        axis.legend(
            handles=handles,
            ncol=min(5, max(1, len(handles))),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            frameon=False,
            fontsize=8,
        )
        figure.subplots_adjust(bottom=0.27)
        save_figure(figure, output_dir / output_name)
        plt.close(figure)


def write_average_distinct_latex_table(filename: Path) -> None:
    filename.write_text(
        r"""\documentclass{article}

\usepackage[a4paper,margin=12mm]{geometry}
\usepackage{float}
\usepackage{graphicx}

\pagestyle{empty}
\setlength{\tabcolsep}{3pt}

\begin{document}

\begin{table}[H]
  \centering
  \begin{tabular}{cc}
    \includegraphics[
      width=0.47\textwidth,
      height=0.34\textheight,
      keepaspectratio
    ]{avg-intrinsic.pdf}
    &
    \includegraphics[
      width=0.47\textwidth,
      height=0.34\textheight,
      keepaspectratio
    ]{avg-ner.pdf}
    \\
    Intrinsic & NER
    \\
    \includegraphics[
      width=0.47\textwidth,
      height=0.34\textheight,
      keepaspectratio
    ]{avg-pos.pdf}
    &
    \includegraphics[
      width=0.47\textwidth,
      height=0.34\textheight,
      keepaspectratio
    ]{avg-sentiment.pdf}
    \\
    POS & Sentiment
  \end{tabular}
  \caption{Average distinct n-grams per sentence and task performance.
  Intrinsic uses Accuracy; NER, POS, and Sentiment use F-measure.}
  \label{tab:avg}
\end{table}

\end{document}
""",
        encoding="utf-8",
    )


def write_average_distinct_3x2_latex_table(filename: Path) -> None:
    filename.write_text(
        r"""\documentclass{article}

\usepackage[a4paper,landscape,margin=12mm]{geometry}
\usepackage{float}
\usepackage{graphicx}

\pagestyle{empty}
\setlength{\tabcolsep}{3pt}

\begin{document}

\begin{table}[H]
  \centering
  \begin{tabular}{ccc}
    \includegraphics[
      width=0.315\textwidth,
      height=0.36\textheight,
      keepaspectratio
    ]{avg-intrinsic.pdf}
    &
    \includegraphics[
      width=0.315\textwidth,
      height=0.36\textheight,
      keepaspectratio
    ]{avg-morphology.pdf}
    &
    \includegraphics[
      width=0.315\textwidth,
      height=0.36\textheight,
      keepaspectratio
    ]{avg-ner.pdf}
    \\
    Intrinsic & Morphology & NER
    \\
    \includegraphics[
      width=0.315\textwidth,
      height=0.36\textheight,
      keepaspectratio
    ]{avg-pos.pdf}
    &
    \includegraphics[
      width=0.315\textwidth,
      height=0.36\textheight,
      keepaspectratio
    ]{avg-sentiment.pdf}
    &
    {}
    \\
    POS & Sentiment & {}
  \end{tabular}
  \caption{Average distinct n-grams per sentence and task performance.
  Intrinsic uses Accuracy; Morphology, NER, POS, and Sentiment use
  F-measure.}
  \label{tab:avg}
\end{table}

\end{document}
""",
        encoding="utf-8",
    )


def latex_escape(value: Any) -> str:
    text = str(value if value is not None else "")
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def variant_sort_key(row: dict[str, Any]) -> tuple[str, str, int]:
    variant = str(row.get("variant_id", ""))
    variant_number = (
        int(variant[1:])
        if variant.startswith("v") and variant[1:].isdigit()
        else sys.maxsize
    )
    return row["task"], row["method"], variant_number


def write_all_findings_latex_table(
    filename: Path,
    rows: Sequence[dict[str, Any]],
) -> None:
    body = []
    for row in sorted(rows, key=variant_sort_key):
        task = row["task"]
        metric = "accuracy" if task == "analogy" else "f_score"
        score = row.get(metric)
        metric_status = (
            "legacy"
            if task in {"ner", "pos", "sentiment"}
            and (
                row.get("precision") is None
                or row.get("recall") is None
            )
            else "current"
        )
        cells = [
            task.upper(),
            row["method"],
            row.get("variant_id", ""),
            row.get("lm_window_length", ""),
            row.get("lm_slide_length", ""),
            row.get("lm_top_split", ""),
            row.get("lm_skip", ""),
            row.get("lm_stem_length", ""),
            row.get("lm_prune", ""),
            row.get("lm_likelihood_weight", ""),
            row.get("lm_prior_weight", ""),
            row.get("lm_length_penalty", ""),
            row.get("sentencepiece_model_type", ""),
            row.get("sentencepiece_vocab_size", ""),
            row.get("sentencepiece_character_coverage", ""),
            DISPLAY_METRIC[metric],
            f"{float(score):.6f}" if score is not None else "—",
            metric_status,
        ]
        body.append(" & ".join(latex_escape(cell) for cell in cells) + r" \\")

    preamble = r"""\documentclass{article}

\usepackage[a4paper,landscape,margin=8mm]{geometry}
\usepackage{array}
\usepackage{booktabs}
\usepackage{xltabular}

\pagestyle{plain}
\setlength{\tabcolsep}{2pt}
\renewcommand{\arraystretch}{1}
\newcolumntype{Y}{>{\raggedright\arraybackslash\hspace{0pt}}X}

\begin{document}
\scriptsize

\begin{xltabular}{\linewidth}{@{}lYcrrrrrrrrYlrrlrl@{}}
\caption{Complete ablation findings with selected task metrics and LM
parameters. Analogy is reported with Accuracy; Morphology, NER, POS, and
Sentiment are reported with F-measure. A legacy status marks sequence-labeling
artifacts without accompanying Precision and Recall fields.}
\label{tab:all-findings}\\
\toprule
Task & Method & ID & Win. & Slide & Top & Skip & Stem & Prune &
$w_{\mathrm{like}}$ & $w_{\mathrm{prior}}$ & Penalty & SP type & SP vocab &
SP cov. & Metric & Score & Status \\
\midrule
\endfirsthead

\multicolumn{18}{c}{\tablename\ \thetable{} --- continued} \\
\toprule
Task & Method & ID & Win. & Slide & Top & Skip & Stem & Prune &
$w_{\mathrm{like}}$ & $w_{\mathrm{prior}}$ & Penalty & SP type & SP vocab &
SP cov. & Metric & Score & Status \\
\midrule
\endhead

\midrule
\multicolumn{18}{r}{Continued on next page} \\
\endfoot

\bottomrule
\endlastfoot
"""
    ending = r"""
\end{xltabular}

\end{document}
"""
    filename.write_text(
        preamble + "\n".join(body) + ending,
        encoding="utf-8",
    )


def write_density_latex_table(filename: Path) -> None:
    filename.write_text(
        r"""\documentclass{article}

\usepackage[a4paper,landscape,margin=12mm]{geometry}
\usepackage{array}
\usepackage{graphicx}

\pagestyle{empty}
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.15}

\newcommand{\plotcell}[2]{%
  \begin{minipage}[t]{0.315\textwidth}
    \centering
    \includegraphics[
      width=\linewidth,
      height=0.37\textheight,
      keepaspectratio
    ]{#1}

    \small #2
  \end{minipage}%
}

\begin{document}

\begin{center}
  \centering
  \begin{tabular}{@{}ccc@{}}
    \plotcell{lm_graph_density_analogy.pdf}{(a) Analogy---Accuracy}
    &
    \plotcell{lm_graph_density_morphology.pdf}{(b) Morphology---F-measure}
    &
    \plotcell{lm_graph_density_ner.pdf}{(c) NER---F-measure}
    \\
    \plotcell{lm_graph_density_pos.pdf}{(d) POS---F-measure}
    &
    \plotcell{lm_graph_density_sentiment.pdf}{(e) Sentiment---F-measure}
    &
    {}
  \end{tabular}
\end{center}

\end{document}
""",
        encoding="utf-8",
    )


def markdown_table(headers: Sequence[str], body: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def write_findings(
    filename: Path,
    rows: Sequence[dict[str, Any]],
    source_files: Sequence[Path],
    methods: Sequence[dict[str, Any]],
    best: Sequence[dict[str, Any]],
    effects: Sequence[dict[str, Any]],
) -> None:
    tasks = sorted({row["task"] for row in rows})
    lines = [
        "# Reviewer 1 ablation findings",
        "",
        (
            f"Loaded **{len(rows)} variants** from **{len(source_files)} experiment "
            "index CSV files**. Scores missing from index CSVs were read from each "
            "row's `result_filename` artifact."
        ),
        "",
        "## Metric coverage",
        "",
    ]
    coverage_body = []
    for task in tasks:
        task_rows = [row for row in rows if row["task"] == task]
        coverage_body.append(
            [
                task.upper(),
                str(len(task_rows)),
                *[
                    f"{sum(row.get(metric) is not None for row in task_rows)}/{len(task_rows)}"
                    for metric in METRICS
                ],
            ]
        )
    lines.append(
        markdown_table(
            ["Task", "Variants", *[DISPLAY_METRIC[metric] for metric in METRICS]],
            coverage_body,
        )
    )
    lines.extend(["", "## Best observed variants", ""])
    best_body = [
        [
            row["task"].upper(),
            DISPLAY_METRIC[row["metric"]],
            row["method"],
            str(row["variant_id"]),
            f"{row['value']:.6g}",
        ]
        for row in best
    ]
    lines.append(
        markdown_table(
            ["Task", "Metric", "Method", "Variant", "Score"], best_body
        )
    )

    lines.extend(["", "## Method ranking by mean F-Score", ""])
    ranking_body = []
    for task in tasks:
        if task == "analogy":
            continue
        task_methods = [
            row
            for row in methods
            if row["task"] == task and row.get("f_score_mean") is not None
        ]
        for rank, row in enumerate(
            sorted(task_methods, key=lambda item: item["f_score_mean"], reverse=True),
            1,
        ):
            ranking_body.append(
                [
                    task.upper(),
                    str(rank),
                    row["method"],
                    f"{row['f_score_mean']:.6g}",
                    f"{row['f_score_std']:.6g}",
                    str(row["f_score_n"]),
                ]
            )
    lines.append(
        markdown_table(
            ["Task", "Rank", "Method", "Mean", "SD", "N"], ranking_body
        )
    )

    word_rows = sorted(
        (row for row in rows if row["method"] == "lm-word"),
        key=lambda row: row["task"],
    )
    if word_rows:
        lines.extend(["", "## `lm-word` baseline", ""])
        word_body = []
        for row in word_rows:
            word_body.append(
                [
                    row["task"].upper(),
                    (
                        f"{row['accuracy']:.6g}"
                        if row.get("accuracy") is not None
                        else "—"
                    ),
                    (
                        "Not compared"
                        if row["task"] == "analogy"
                        else (
                            f"{row['f_score']:.6g}"
                            if row.get("f_score") is not None
                            else "—"
                        )
                    ),
                ]
            )
        lines.append(
            markdown_table(["Task", "Accuracy", "F-Score"], word_body)
        )
        lines.extend(
            [
                "",
                (
                    "`lm-word` contributes one baseline variant per task. Its "
                    "zero standard deviations and method means therefore describe "
                    "single observations, not stability across configurations."
                ),
            ]
        )

    lines.extend(["", "## Strongest descriptive parameter effects", ""])
    sensitivity_body = []
    for task in tasks:
        task_effects = [row for row in effects if row["task"] == task]
        aggregate: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in task_effects:
            aggregate[(row["metric"], row["parameter"])].append(
                row["relative_effect"]
            )
        ranked = sorted(
            (
                (statistics.median(scores), metric, parameter)
                for (metric, parameter), scores in aggregate.items()
            ),
            reverse=True,
        )[:3]
        for effect, metric, parameter in ranked:
            sensitivity_body.append(
                [
                    task.upper(),
                    DISPLAY_METRIC[metric],
                    parameter,
                    f"{effect:.3f}",
                ]
            )
    lines.append(
        markdown_table(
            ["Task", "Metric", "Parameter", "Relative effect"], sensitivity_body
        )
    )

    outliers = []
    for metric in METRICS:
        scores = values(rows, metric)
        invalid = [score for score in scores if score < 0.0 or score > 1.0]
        if invalid:
            outliers.append(
                f"- `{metric}` has {len(invalid)} values outside [0, 1] "
                f"(range {min(invalid):.6g}–{max(invalid):.6g})."
            )
    lines.extend(["", "## Interpretation notes", ""])
    if outliers:
        lines.extend(outliers)
        if any(row["task"] == "analogy" for row in rows):
            lines.append(
                "- The unbounded Analogy `F1-MEASURE` values are legacy similarity "
                "outputs, not interpretable classification F-Scores; use the "
                "Analogy Accuracy column or a separately named similarity measure."
            )
    else:
        lines.append("- All available classification metrics lie in [0, 1].")
    legacy_extrinsic = [
        row
        for row in rows
        if row["task"] in {"ner", "pos", "sentiment"}
        and row.get("f_score") is not None
        and (row.get("precision") is None or row.get("recall") is None)
    ]
    if legacy_extrinsic:
        lines.append(
            f"- {len(legacy_extrinsic)} extrinsic rows have F-Score but no "
            "Precision/Recall. These legacy scores should not be assumed to use "
            "the current macro-F1 implementation without rerunning evaluation."
        )
    lines.extend(
        [
            (
                "- Missing precision and recall indicate legacy result XML files that "
                "predate those fields; they are not imputed."
            ),
            (
                "- Parameter effects are descriptive grouped-mean ranges, not causal "
                "effects or significance tests."
            ),
            (
                "- Results use the project's fixed train/test evaluations. No "
                "cross-validation or paired t-test is performed by this script."
            ),
            (
                "- Analogy F-Score is retained only in the joined raw table; it is "
                "excluded from score comparisons, rankings, and figures."
            ),
        ]
    )
    filename.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir / "plots").resolve()
    if not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, source_files = load_results(input_dir)
    if not rows:
        print(f"No ablation index rows found in {input_dir}", file=sys.stderr)
        return 2

    method_rows = summarize_methods(rows)
    best_rows = best_variants(rows)
    selected_best_rows = selected_metric_best_methods(rows)
    effect_rows = parameter_effects(rows)
    selected_effect_rows = selected_sensitivity_effects(effect_rows)
    try:
        density_rows = prepare_graph_density_rows(
            rows, input_dir, args.density_sample_size
        )
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    density_correlation_rows = graph_density_correlations(density_rows)
    average_distinct_correlation_rows = average_distinct_ngram_correlations(
        density_rows
    )
    status_rows = status_summary(rows)

    original_fields = []
    for row in rows:
        for field in row:
            if field not in original_fields:
                original_fields.append(field)
    write_csv(output_dir / "combined_results.csv", rows, original_fields)

    method_fields = ["task", "method", "variants"]
    for metric in METRICS:
        method_fields.extend(
            [
                f"{metric}_n",
                f"{metric}_mean",
                f"{metric}_std",
                f"{metric}_min",
                f"{metric}_max",
            ]
        )
    write_csv(output_dir / "method_summary.csv", method_rows, method_fields)
    write_csv(
        output_dir / "best_variants.csv",
        best_rows,
        [
            "task",
            "metric",
            "value",
            "method",
            "variant_id",
            *PARAMETERS,
            "result_filename",
        ],
    )
    write_csv(
        output_dir / "best_method_performance.csv",
        selected_best_rows,
        [
            "task",
            "metric",
            "method",
            "variant_id",
            "score",
            "score_status",
            "best_comparable_variant_id",
            "best_comparable_score",
            "result_filename",
        ],
    )
    write_csv(
        output_dir / "parameter_effects.csv",
        effect_rows,
        [
            "task",
            "method",
            "metric",
            "parameter",
            "levels",
            "observations",
            "effect_range",
            "relative_effect",
            "best_level",
            "best_level_mean",
        ],
    )
    write_csv(
        output_dir / "parameter_sensitivity_selected.csv",
        selected_effect_rows,
        [
            "task",
            "method",
            "metric",
            "parameter",
            "levels",
            "observations",
            "effect_range",
            "relative_effect",
            "best_level",
            "best_level_mean",
        ],
    )
    write_csv(
        output_dir / "lm_graph_density.csv",
        density_rows,
        [
            "task",
            "method",
            "variant_id",
            "lm_window_length",
            "metric",
            "score",
            "graph_density",
            "density_basis",
            "sampled_sentences",
            "distinct_partitions",
            "total_partitions",
            "total_edges",
            "average_sentence_partitions",
            "average_distinct_ngrams_per_sentence",
            "included_in_plot",
            "metric_status",
            "corpus_filename",
            "resolved_corpus_file",
        ],
    )
    write_csv(
        output_dir / "lm_graph_density_correlations.csv",
        density_correlation_rows,
        [
            "task",
            "metric",
            "observations",
            "pearson_r_density",
            "pearson_r_log10_density",
            "density_min",
            "density_max",
            "score_min",
            "score_max",
            "density_basis",
        ],
    )
    write_csv(
        output_dir / "average_distinct_ngrams.csv",
        [
            row
            for row in density_rows
            if row["task"]
            in {"analogy", "morphology", "ner", "pos", "sentiment"}
        ],
        [
            "task",
            "method",
            "variant_id",
            "lm_window_length",
            "metric",
            "score",
            "average_distinct_ngrams_per_sentence",
            "sampled_sentences",
            "included_in_plot",
            "metric_status",
            "corpus_filename",
            "resolved_corpus_file",
        ],
    )
    write_csv(
        output_dir / "average_distinct_ngrams_correlations.csv",
        average_distinct_correlation_rows,
        [
            "task",
            "metric",
            "observations",
            "pearson_r",
            "average_distinct_ngrams_min",
            "average_distinct_ngrams_max",
            "score_min",
            "score_max",
        ],
    )
    write_density_latex_table(output_dir / "lm_graph_density_3x2_table.tex")
    write_average_distinct_latex_table(
        output_dir / "average_distinct_ngrams_2x2_table.tex"
    )
    write_average_distinct_3x2_latex_table(
        output_dir / "average_distinct_ngrams_3x2_table.tex"
    )
    write_all_findings_latex_table(
        output_dir / "all_findings_selected_metrics.tex", rows
    )
    write_csv(
        output_dir / "status_summary.csv",
        status_rows,
        ["task", "status", "count"],
    )
    write_findings(
        output_dir / "findings.md",
        rows,
        source_files,
        method_rows,
        best_rows,
        selected_effect_rows,
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "Analysis tables were written, but plots require matplotlib. "
            "Install it with: python3 -m pip install -r scripts/plot-requirements.txt",
            file=sys.stderr,
        )
        return 1

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    plot_coverage(plt, rows, output_dir)
    plot_task_distributions(plt, rows, output_dir)
    plot_accuracy_vs_fscore(plt, rows, output_dir)
    plot_best_method_heatmaps(plt, rows, output_dir)
    plot_parameter_sensitivity(plt, selected_effect_rows, output_dir)
    plot_lm_graph_density(
        plt, density_rows, density_correlation_rows, output_dir
    )
    plot_lm_graph_density_vs_fscore(
        plt, density_rows, density_correlation_rows, output_dir
    )
    plot_average_distinct_ngrams(
        plt, density_rows, average_distinct_correlation_rows, output_dir
    )
    print(f"Wrote analysis and plots to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
