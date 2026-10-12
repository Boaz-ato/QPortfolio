"""Benchmark visualization and result persistence for ParetoQ.

All plot functions accept ``list[Any]`` (duck-typed ``SolverBenchmarkResult``)
to avoid circular imports.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # non-interactive backend for saving


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def _color(idx: int) -> str:
    return _COLORS[idx % len(_COLORS)]


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------


def plot_hypervolume_vs_time(results: list[Any], title: str = "Hypervolume vs Time") -> Any:
    """Line plot of hypervolume over elapsed time for each solver.

    Uses all history events that have a recorded HV value plus the final
    benchmark HV, connected with straight lines to show gradual growth.

    Parameters
    ----------
    results : list of SolverBenchmarkResult
    title : plot title

    Returns
    -------
    matplotlib Figure
    """

    fig, ax = plt.subplots(figsize=(9, 5))

    for idx, bench in enumerate(results):
        color = _color(idx)
        label = bench.algorithm

        times = [0.0]
        hvs = [0.0]

        raw = bench.raw_result
        if raw is not None and hasattr(raw, "history") and raw.history:
            for event in raw.history:
                hv = event.get("hypervolume")
                t = event.get("elapsed_time", 0.0)
                if hv is not None:
                    times.append(t)
                    hvs.append(hv)

        # Append final point — ensures the line reaches the end of the run
        final_hv = (
            bench.hypervolume if bench.hypervolume is not None else (hvs[-1] if len(hvs) > 1 else 0.0)
        )
        times.append(bench.runtime)
        hvs.append(final_hv)

        ax.plot(times, hvs, color=color, label=label, linewidth=1.8, marker="o", markersize=4)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Hypervolume")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_nondominated_vs_time(
    results: list[Any], title: str = "Non-Dominated Solutions per Solver"
) -> Any:
    """Bar chart of final non-dominated solution count per solver.

    Parameters
    ----------
    results : list of SolverBenchmarkResult
    title : plot title

    Returns
    -------
    matplotlib Figure
    """

    labels = [bench.algorithm for bench in results]
    counts = [bench.n_nondominated for bench in results]
    colors = [_color(idx) for idx in range(len(results))]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8 + 1), 5))
    bars = ax.bar(labels, counts, color=colors, edgecolor="white", width=0.6)

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            str(count),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("Solver")
    ax.set_ylabel("Non-Dominated Solutions")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_pareto_front_3d(results: list[Any], title: str = "Pareto Front (3D)") -> Any:
    """One 3D scatter subplot per solver.

    Each subplot shows all feasible solutions (grey, small) behind the
    non-dominated Pareto front (solver colour, larger) — matching the
    style of the reference image.

    Objectives stored as [f1=-risk, f2=return, f3=diversification] (all-maximise).
    Axes: Negative risk (f1), Return (f2), Diversification (f3).

    Parameters
    ----------
    results : list of SolverBenchmarkResult
    title : plot title

    Returns
    -------
    matplotlib Figure
    """

    n = len(results)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)

    fig = plt.figure(figsize=(6 * ncols, 5 * nrows))
    fig.suptitle(title, fontsize=13, y=1.01)

    for idx, bench in enumerate(results):
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection="3d")
        ax.set_title(bench.algorithm, fontsize=10)

        raw = bench.raw_result
        has_any = raw is not None and hasattr(raw, "solutions") and raw.solutions
        has_nd = (
            raw is not None and hasattr(raw, "nondominated_solutions") and raw.nondominated_solutions
        )

        if not has_any and not has_nd:
            ax.text2D(0.5, 0.5, "No solutions", ha="center", va="center", transform=ax.transAxes)
        else:
            # All feasible solutions — grey background
            if has_any:
                all_objs = np.array([s.objectives for s in raw.solutions])
                ax.scatter(
                    all_objs[:, 0],
                    all_objs[:, 1],
                    all_objs[:, 2],
                    c="#888888",
                    s=45,
                    alpha=0.75,
                    depthshade=True,
                    edgecolors="#555555",
                    linewidths=0.3,
                    label="All feasible portfolios",
                )

            # Non-dominated (Pareto front) — solver colour, larger
            if has_nd:
                nd_objs = np.array([s.objectives for s in raw.nondominated_solutions])
                ax.scatter(
                    nd_objs[:, 0],
                    nd_objs[:, 1],
                    nd_objs[:, 2],
                    c=_color(idx),
                    s=60,
                    alpha=0.95,
                    edgecolors="k",
                    linewidths=0.3,
                    depthshade=True,
                    label="Pareto front",
                )

            ax.legend(loc="upper left", fontsize=6, markerscale=0.8)

        ax.set_xlabel("Neg. risk (f₁)", fontsize=8, labelpad=4)
        ax.set_ylabel("Return (f₂)", fontsize=8, labelpad=4)
        ax.set_zlabel("Diversif. (f₃)", fontsize=8, labelpad=4)
        ax.tick_params(labelsize=7)

    fig.tight_layout()
    return fig


def plot_runtime(results: list[Any], title: str = "Solver Runtime") -> Any:
    """Horizontal bar chart of total runtime per solver (sorted descending).

    Parameters
    ----------
    results : list of SolverBenchmarkResult
    title : plot title

    Returns
    -------
    matplotlib Figure
    """

    sorted_results = sorted(results, key=lambda b: b.runtime, reverse=True)
    labels = [b.algorithm for b in sorted_results]
    runtimes = [b.runtime for b in sorted_results]
    colors = [_color(results.index(b)) for b in sorted_results]

    fig, ax = plt.subplots(figsize=(9, max(3, len(labels) * 0.5 + 1)))
    bars = ax.barh(labels, runtimes, color=colors, edgecolor="white", height=0.6)

    for bar, rt in zip(bars, runtimes):
        ax.text(
            bar.get_width() + max(runtimes) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{rt:.2f}s",
            va="center",
            fontsize=8,
        )

    ax.set_xlabel("Runtime (s)")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------


def save_benchmark_results(
    results: list[Any],
    problem: Any,
    output_dir: str | Path | None = None,
    run_name: str | None = None,
) -> Path:
    """Save all plots and data to a uniquely named results directory.

    Parameters
    ----------
    results : list of SolverBenchmarkResult
    problem : PortfolioProblem
    output_dir : root directory for results (default: ``Path("results")``)
    run_name : subdirectory name (default: ``run_<datetime>_<problem_id[:8]>``)

    Returns
    -------
    Path to the created run directory
    """
    if output_dir is None:
        output_dir = Path("results")
    output_dir = Path(output_dir)

    if run_name is None:
        dt_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        pid = getattr(problem, "problem_id", "unknown")[:8]
        run_name = f"run_{dt_str}_{pid}"

    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── CSV summary ──────────────────────────────────────────────────────────
    csv_path = run_dir / "summary_table.csv"
    csv_fields = [
        "algorithm",
        "status",
        "n_nondominated",
        "n_feasible",
        "n_evaluations",
        "hypervolume",
        "normalized_hv",
        "runtime_s",
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields)
        writer.writeheader()
        for bench in results:
            writer.writerow(
                {
                    "algorithm": bench.algorithm,
                    "status": bench.status,
                    "n_nondominated": bench.n_nondominated,
                    "n_feasible": bench.n_feasible,
                    "n_evaluations": bench.n_evaluations,
                    "hypervolume": bench.hypervolume,
                    "normalized_hv": bench.normalized_hypervolume,
                    "runtime_s": bench.runtime,
                }
            )

    # ── JSON detail ──────────────────────────────────────────────────────────
    json_data: dict[str, Any] = {}
    for bench in results:
        raw = bench.raw_result
        history = []
        if raw is not None and hasattr(raw, "history") and raw.history:
            history = raw.history

        quantum_meta = None
        if raw is not None and hasattr(raw, "quantum"):
            quantum_meta = raw.quantum

        json_data[bench.algorithm] = {
            "algorithm": bench.algorithm,
            "status": bench.status,
            "n_nondominated": bench.n_nondominated,
            "n_feasible": bench.n_feasible,
            "n_evaluations": bench.n_evaluations,
            "hypervolume": bench.hypervolume,
            "normalized_hv": bench.normalized_hypervolume,
            "runtime_s": bench.runtime,
            "history": history,
            "quantum": quantum_meta,
        }

    json_path = run_dir / "solver_results.json"
    with open(json_path, "w") as fh:
        json.dump(json_data, fh, indent=2, default=_json_default)

    # ── Plots ────────────────────────────────────────────────────────────────

    fig = plot_hypervolume_vs_time(results)
    fig.savefig(run_dir / "plot_hv_vs_time.png", dpi=150)
    _close_fig(fig)

    fig = plot_nondominated_vs_time(results)
    fig.savefig(run_dir / "plot_nd_vs_time.png", dpi=150)
    _close_fig(fig)

    fig = plot_pareto_front_3d(results)
    fig.savefig(run_dir / "plot_pareto_front_3d.png", dpi=150)
    _close_fig(fig)

    fig = plot_runtime(results)
    fig.savefig(run_dir / "plot_runtime.png", dpi=150)
    _close_fig(fig)

    return run_dir


def _close_fig(fig: Any) -> None:
    plt.close(fig)


def _json_default(obj: Any) -> Any:
    """JSON serializer for numpy types and other non-serializable objects."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
