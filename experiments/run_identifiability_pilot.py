"""Run the frozen two-dimensional noiseless identifiability pilot."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import scipy

from local_transferability.identifiability_metrics import compute_identifiability_metrics
from local_transferability.quadratic_estimator import fit_complete_quadratic
from local_transferability.sampling import sample_design


RATIOS = (1, 4, 16)
ANGLES = (0.0, np.pi / 8.0, np.pi / 4.0)
DESIGNS = ("sobol", "uniform", "trajectory")
SAMPLE_SIZES = (3, 5, 6, 8, 12, 20)
SEED_COUNT = 50
PROBES = np.array(
    [
        [-0.91, -0.37], [-0.73, 0.58], [-0.44, 0.19], [-0.21, -0.82],
        [0.07, 0.31], [0.29, -0.54], [0.48, 0.77], [0.66, -0.13],
        [0.84, 0.42], [0.95, -0.69],
    ], dtype=float,
)


def rotation(angle: float) -> np.ndarray:
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def curvature(ratio: int, angle: float) -> np.ndarray:
    q = rotation(angle)
    return q @ np.diag([1.0, float(ratio)]) @ q.T


def model_values(points: np.ndarray, hessian: np.ndarray) -> np.ndarray:
    return 0.5 * np.einsum("ni,ij,nj->n", points, hessian, points)


def run_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ratio in RATIOS:
        for angle_index, angle in enumerate(ANGLES):
            truth = curvature(ratio, angle)
            truth_probe = model_values(PROBES, truth)
            for design in DESIGNS:
                for n in SAMPLE_SIZES:
                    for seed in range(SEED_COUNT):
                        paired_seed = seed + 10_000 * angle_index + 100_000 * ratio
                        points = sample_design(design, n, paired_seed, truth)
                        values = model_values(points, truth)
                        estimate = fit_complete_quadratic(points, values)
                        row: dict[str, object] = {
                            "ratio": ratio,
                            "angle": float(angle),
                            "angle_index": angle_index,
                            "design": design,
                            "n": n,
                            "seed": seed,
                            "paired_seed": paired_seed,
                            "identifiable": int(estimate.identifiable),
                            "abstain_reason": estimate.abstain_reason or "",
                            "rank": estimate.rank,
                            "condition_number": estimate.condition_number,
                            "minimum_singular_value": float(estimate.singular_values[-1]),
                            "maximum_singular_value": float(estimate.singular_values[0]),
                            "hessian_relative_error": "",
                            "magnitude_relative_error": "",
                            "spectrum_l1_error": "",
                            "log_condition_error": "",
                            "pairwise_order_accuracy": "",
                            "tie_fraction": "",
                            "estimated_magnitude": "",
                            "estimated_major_spectrum_share": "",
                            "signed_spectrum_shape_error": "",
                            "pairwise_coverage": "",
                            "estimated_min_eigenvalue": "",
                            "is_spd": "",
                            "inertia_mismatch": "",
                            "negative_eigenvalue_fraction": "",
                        }
                        if estimate.identifiable and estimate.hessian is not None:
                            predicted_probe = model_values(PROBES, estimate.hessian)
                            abs_eigenvalues = np.abs(np.linalg.eigvalsh(estimate.hessian))
                            row["estimated_magnitude"] = float(
                                np.linalg.norm(estimate.hessian, ord="fro") / np.sqrt(2.0)
                            )
                            row["estimated_major_spectrum_share"] = float(
                                np.max(abs_eigenvalues) / np.sum(abs_eigenvalues)
                            )
                            metrics = compute_identifiability_metrics(
                                truth, estimate.hessian, truth_probe, predicted_probe
                            )
                            for name, value in vars(metrics).items():
                                row[name] = value
                        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q)) if values else float("nan")


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[int, int, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["ratio"]), int(row["angle_index"]), str(row["design"]), int(row["n"]))].append(row)
    summaries: list[dict[str, object]] = []
    metric_names = (
        "hessian_relative_error", "magnitude_relative_error", "spectrum_l1_error",
        "log_condition_error", "pairwise_order_accuracy", "tie_fraction",
    )
    for key, group in sorted(groups.items()):
        valid = [row for row in group if int(row["identifiable"]) == 1]
        summary: dict[str, object] = {
            "ratio": key[0], "angle_index": key[1], "design": key[2], "n": key[3],
            "runs": len(group), "identifiable_runs": len(valid),
            "identifiable_rate": len(valid) / len(group),
            "condition_median": percentile([float(row["condition_number"]) for row in valid], 0.5),
            "condition_p90": percentile([float(row["condition_number"]) for row in valid], 0.9),
        }
        for metric in metric_names:
            values = [float(row[metric]) for row in valid]
            summary[f"{metric}_median"] = percentile(values, 0.5)
            summary[f"{metric}_p90"] = percentile(values, 0.9)
            summary[f"{metric}_max"] = max(values) if values else float("nan")
        summaries.append(summary)
    return summaries


def plot_results(rows: list[dict[str, object]], output: Path) -> None:
    colors = {"sobol": "#0072B2", "uniform": "#009E73", "trajectory": "#D55E00"}
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for design in DESIGNS:
        rates = []
        for n in SAMPLE_SIZES:
            selected = [r for r in rows if r["design"] == design and r["n"] == n]
            rates.append(np.mean([int(r["identifiable"]) for r in selected]))
        axis.plot(SAMPLE_SIZES, rates, marker="o", label=design, color=colors[design])
    axis.set(xlabel="Sample size n", ylabel="Identifiable rate", ylim=(-0.03, 1.03))
    axis.legend(); axis.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(output / "figure1_identifiable_rate.png", dpi=180); plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4.5))
    for design in DESIGNS:
        medians = []
        p90s = []
        for n in SAMPLE_SIZES:
            values = [float(r["spectrum_l1_error"]) for r in rows if r["design"] == design and r["n"] == n and r["identifiable"] == 1]
            medians.append(percentile(values, 0.5)); p90s.append(percentile(values, 0.9))
        axis.plot(SAMPLE_SIZES, medians, marker="o", label=f"{design} median", color=colors[design])
        axis.plot(SAMPLE_SIZES, p90s, linestyle="--", alpha=0.65, color=colors[design])
    axis.set(xlabel="Sample size n", ylabel="Normalized-spectrum L1 error", yscale="log")
    axis.legend(fontsize=8); axis.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(output / "figure2_spectrum_error.png", dpi=180); plt.close(fig)

    valid = [r for r in rows if r["identifiable"] == 1]
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for design in DESIGNS:
        selected = [r for r in valid if r["design"] == design]
        axis.scatter([r["condition_number"] for r in selected], [r["hessian_relative_error"] for r in selected], s=7, alpha=0.25, label=design, color=colors[design])
    axis.set(xlabel="Design-matrix condition number", ylabel="Hessian relative error", xscale="log", yscale="log")
    axis.legend(); axis.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(output / "figure3_error_vs_condition.png", dpi=180); plt.close(fig)

    # Estimated descriptor variation across samplers vs between distinct spectra.
    ns = [6, 8, 12, 20]
    within, between = [], []
    for n in ns:
        cell_medians: dict[tuple[int, int, str], float] = {}
        for ratio in RATIOS:
            for angle_index in range(len(ANGLES)):
                for design in DESIGNS:
                    vals = [float(r["estimated_major_spectrum_share"]) for r in valid if r["n"] == n and r["ratio"] == ratio and r["angle_index"] == angle_index and r["design"] == design]
                    cell_medians[(ratio, angle_index, design)] = percentile(vals, 0.5)
        within.append(float(np.mean([np.var([cell_medians[(ratio, angle_index, d)] for d in DESIGNS]) for ratio in RATIOS for angle_index in range(len(ANGLES))])))
        design_means = {d: [cell_medians[(r, a, d)] for r in RATIOS for a in range(len(ANGLES))] for d in DESIGNS}
        between.append(float(np.mean([np.var(design_means[d]) for d in DESIGNS])))
    x = np.arange(len(ns)); width = 0.36
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(x - width / 2, within, width, label="same landscape across samplers")
    axis.bar(x + width / 2, between, width, label="different spectra within sampler")
    axis.set_xticks(x, ns); axis.set(xlabel="Sample size n", ylabel="Variance of estimated major spectrum share")
    axis.legend(fontsize=8); axis.grid(axis="y", alpha=0.25); fig.tight_layout()
    fig.savefig(output / "figure4_sampler_variance.png", dpi=180); plt.close(fig)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output
    raw_path = output / "identifiability_pilot.csv"
    if raw_path.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite {raw_path}; pass --overwrite explicitly")
    output.mkdir(parents=True, exist_ok=True)
    rows = run_rows()
    summaries = summarize(rows)
    write_csv(raw_path, rows)
    write_csv(output / "identifiability_summary.csv", summaries)
    plot_results(rows, output)
    manifest = {
        "study": "2d-noiseless-oracle-representation-identifiability-pilot",
        "git_commit_before_run": git_commit(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "seed_count": SEED_COUNT,
        "ratios": RATIOS,
        "angles": ANGLES,
        "designs": DESIGNS,
        "sample_sizes": SAMPLE_SIZES,
        "row_count": len(rows),
        "command": " ".join(sys.argv),
    }
    (output / "identifiability_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "summaries": len(summaries), "output": str(output)}))


if __name__ == "__main__":
    main()
