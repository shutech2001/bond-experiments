from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.lines import Line2D

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
matplotlib.use("Agg")


def set_plot_style() -> None:
    """Set the plot style for visualizations"""
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 24,
            "axes.labelsize": 28,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 24,
            "mathtext.fontset": "stix",
            "text.usetex": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def math_label(expr: str) -> str:
    """Generate a LaTeX-formatted math label

    Args:
        expr (str): The expression to label

    Returns:
        str: The LaTeX-formatted math label
    """
    return rf"$${expr}$$"


def save_legend_pdf(
    path: Path,
    handles: List[Line2D],
    labels: List[str],
    ncol: int = 5,
    fontsize: float = 24.0,
) -> None:
    """Save a legend as a PDF file

    Args:
        path (Path): The path to save the legend
        handles (List[Line2D]): The handles for the legend
        labels (List[str]): The labels for the legend
        ncol (int, optional): The number of columns for the legend. Defaults to 5.
        fontsize (float, optional): The font size for the legend. Defaults to 24.0.
    """
    if len(handles) == 0:
        return
    ncol_eff = max(1, min(ncol, len(labels)))
    height = max(2.2, 1.2 + 0.85 * math.ceil(len(labels) / ncol_eff))
    width = max(16.0, 1.8 + 3.2 * ncol_eff)
    fig_leg, ax_leg = plt.subplots(figsize=(width, height))
    ax_leg.axis("off")
    ax_leg.legend(handles, labels, loc="center", ncol=ncol_eff, frameon=False, fontsize=fontsize)
    fig_leg.tight_layout()
    fig_leg.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig_leg)


def build_method_styles(method_order: List[str]) -> Dict[str, Dict[str, Any]]:
    """Build the method styles for the visualizations

    Args:
        method_order (List[str]): The order of the methods

    Returns:
        Dict[str, Dict[str, Any]]: The method styles
    """
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "8", "p", "1"]
    linestyles: List[Any] = [
        "-",
        "--",
        "-.",
        ":",
        (0, (5, 1)),
        (0, (3, 1, 1, 1)),
        (0, (1, 1)),
    ]
    colors = list(colormaps["tab20"].colors) + list(colormaps["Set2"].colors)  # type: ignore
    marker_style_pairs = [(m, ls) for ls in linestyles for m in markers]

    style_map: Dict[str, Dict[str, Any]] = {}
    for idx, method in enumerate(method_order):
        marker, linestyle = marker_style_pairs[idx % len(marker_style_pairs)]
        style_map[method] = {
            "marker": marker,
            "linestyle": linestyle,
            "color": colors[idx % len(colors)],
            "linewidth": 3.2,
            "markersize": 10.0,
        }
    if "BOND" in style_map:
        style_map["BOND"].update(
            {
                "color": "#c40000",
                "linewidth": 4.8,
                "markersize": 13.0,
                "marker": "o",
                "linestyle": "-",
            }
        )
    return style_map


def select_focus_methods(method_order: List[str]) -> List[str]:
    """Select the focus methods for the visualizations

    Args:
        method_order (List[str]): The order of the methods

    Returns:
        List[str]: The focus methods
    """
    preferred = [
        "Current-only",
        "Naive pooling",
        "Fixed lambda=0.50",
        "Power prior(lambda=0.50)",
        "Commensurate prior(tau=1.00)",
        "Robust MAP(epsilon=0.20)",
        "BOND",
    ]
    selected = [m for m in preferred if m in method_order]
    for method in method_order:
        if method not in selected and method == "BOND":
            selected.append(method)
    return selected


def save_type1_power_figure(
    outdir: Path,
    gamma_grid: List[float],
    type1_table: List[Dict[str, float]],
    power_table: List[Dict[str, float]],
    methods: List[str],
    style_map: Dict[str, Dict[str, Any]],
    alpha: float,
    filename_stem: str,
    figsize: Tuple[float, float],
) -> None:
    """Save a type-I error and power figure

    Args:
        outdir (Path): The directory to save the figure
        gamma_grid (List[float]): The gamma grid
        type1_table (List[Dict[str, float]]): The type-I error table
        power_table (List[Dict[str, float]]): The power table
        methods (List[str]): The methods
        style_map (Dict[str, Dict[str, Any]]): The style map
        alpha (float): The alpha level
        filename_stem (str): The filename stem
        figsize (Tuple[float, float]): The figure size
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    for method in methods:
        if method not in type1_table[0]:
            continue
        style = style_map[method]
        axes[0].plot(
            gamma_grid,
            [row[method] for row in type1_table],
            label=method,
            marker=style["marker"],
            linestyle=style["linestyle"],
            color=style["color"],
            linewidth=style["linewidth"],
            markersize=style["markersize"],
        )
        axes[1].plot(
            gamma_grid,
            [row[method] for row in power_table],
            label=method,
            marker=style["marker"],
            linestyle=style["linestyle"],
            color=style["color"],
            linewidth=style["linewidth"],
            markersize=style["markersize"],
        )
    axes[0].axhline(
        alpha,
        color="black",
        linestyle=(0, (4, 2)),
        alpha=0.7,
        label=math_label("\\alpha"),
        linewidth=2.6,
    )
    axes[0].set_ylabel("Type-I error")
    axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel("Power")
    axes[1].set_xlabel(math_label("\\gamma"))
    axes[1].grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    save_legend_pdf(
        outdir / f"{filename_stem}_legend.pdf",
        handles,
        labels,
        ncol=5,
        fontsize=24.0,
    )
    fig.tight_layout()
    fig.savefig(outdir / f"{filename_stem}.pdf", format="pdf")
    plt.close(fig)


def save_lambda_figure(
    outdir: Path,
    lambda_table: List[Tuple[float, float, float, float, float, float]],
    hist_arms: str,
    filename_stem: str,
) -> None:
    """Save a lambda figure

    Args:
        outdir (Path): The directory to save the figure
        lambda_table (List[Tuple[float, float, float, float, float, float]]): The lambda table
        hist_arms (str): The historical arms
        filename_stem (str): The filename stem
    """
    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    gammas = [row[0] for row in lambda_table]
    lam0_vals = [row[1] for row in lambda_table]
    lam1_vals = [row[2] for row in lambda_table]
    w0_vals = [row[3] for row in lambda_table]
    w1_vals = [row[4] for row in lambda_table]

    ax.plot(
        gammas,
        lam0_vals,
        marker="o",
        linestyle="-",
        linewidth=4.0,
        markersize=11.5,
        label=math_label("\\lambda_0^*"),
    )
    if hist_arms == "both":
        ax.plot(
            gammas,
            lam1_vals,
            marker="D",
            linestyle="--",
            linewidth=3.8,
            markersize=11.0,
            label=math_label("\\lambda_1^*"),
        )
    ax.plot(
        gammas,
        w0_vals,
        marker="s",
        linestyle="-.",
        linewidth=3.6,
        markersize=10.8,
        label=math_label("w_0(\\lambda^*)"),
    )
    if hist_arms == "both":
        ax.plot(
            gammas,
            w1_vals,
            marker="^",
            linestyle=":",
            linewidth=3.6,
            markersize=10.8,
            label=math_label("w_1(\\lambda^*)"),
        )
    ax.set_xlabel(math_label("\\gamma"))
    ax.set_ylabel("Borrowing level")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    save_legend_pdf(
        outdir / f"{filename_stem}_legend.pdf",
        handles=list(handles),
        labels=list(labels),
        ncol=5,
        fontsize=24.0,
    )
    fig.tight_layout()
    fig.savefig(outdir / f"{filename_stem}.pdf", format="pdf")
    plt.close(fig)
