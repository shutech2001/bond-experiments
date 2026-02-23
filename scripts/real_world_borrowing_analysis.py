from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm  # type: ignore

import borrowing_methods as bm
import plot_utils as pu
import sim_utils as su


def parse_optional_float_list(text: str) -> Optional[List[float]]:
    """Parse an optional list of floats

    Args:
        text (str): The text to parse

    Returns:
        Optional[List[float]]: The list of floats
    """
    cleaned = text.strip()
    if cleaned == "":
        return None
    return su.parse_float_list(cleaned)


def parse_binary_value(raw: str) -> Optional[int]:
    """Parse a binary value

    Args:
        raw (str): The raw value

    Returns:
        Optional[int]: The binary value
    """
    text = str(raw).strip()
    if text == "":
        return None
    try:
        val = float(text)
    except ValueError:
        return None
    if abs(val - 0.0) <= 1e-12:
        return 0
    if abs(val - 1.0) <= 1e-12:
        return 1
    return None


def parse_int01(raw: str) -> Optional[int]:
    """Parse a binary value

    Args:
        raw (str): The raw value

    Returns:
        Optional[int]: The binary value
    """
    text = str(raw).strip()
    if text == "":
        return None
    try:
        val = int(float(text))
    except ValueError:
        return None
    if val in (0, 1):
        return val
    return None


def build_arm_summary_from_binary(y_current: NDArray[np.float64], y_historical: NDArray[np.float64]) -> Any:
    """Build an arm summary from binary data

    Args:
        y_current (NDArray[np.float64]): The current arm data
        y_historical (NDArray[np.float64]): The historical arm data

    Returns:
        Any: The arm summary
    """
    n_c = int(y_current.size)
    n_h = int(y_historical.size)
    sum_c = float(np.sum(y_current)) if n_c > 0 else 0.0
    sum_h = float(np.sum(y_historical)) if n_h > 0 else 0.0
    mean_c = float(sum_c / n_c) if n_c > 0 else 0.0
    mean_h = float(sum_h / n_h) if n_h > 0 else 0.0
    var_c = su.clamp_prob(mean_c) * (1.0 - su.clamp_prob(mean_c)) if n_c > 0 else 0.0
    var_h = su.clamp_prob(mean_h) * (1.0 - su.clamp_prob(mean_h)) if n_h > 0 else 0.0
    return SimpleNamespace(
        n_c=n_c,
        n_h=n_h,
        mean_c=mean_c,
        mean_h=mean_h,
        var_c=float(var_c),
        var_h=float(var_h),
        sum_c=sum_c,
        sum_h=sum_h,
        succ_c=int(round(sum_c)),
        succ_h=int(round(sum_h)),
    )


def load_real_world_binary_data(
    csv_path: Path,
    source_col: str,
    arm_col: str,
    outcome_col: str,
    subject_col: str,
) -> Dict[str, Any]:
    """Load real-world binary data

    Args:
        csv_path (Path): The path to the CSV file
        source_col (str): The name of the source column
        arm_col (str): The name of the arm column
        outcome_col (str): The name of the outcome column
        subject_col (str): The name of the subject column

    Raises:
        ValueError: If the current data does not include both arms (source=1, arm in {0,1})
        ValueError: If the historical data does not include control arm (source=0, arm=0)

    Returns:
        Dict[str, Any]: The data
    """
    y_current_control: list[int] = []
    y_current_treat: list[int] = []
    y_hist_control: list[int] = []
    y_hist_treat: list[int] = []
    source_counts: Counter[str] = Counter()
    arm_counts: Counter[tuple[int, int]] = Counter()
    dropped_rows = 0
    total_rows = 0
    subjects: list[str] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            src_raw = row.get(source_col, "")
            arm_raw = row.get(arm_col, "")
            y_raw = row.get(outcome_col, "")
            src = parse_int01(src_raw)
            arm = parse_int01(arm_raw)
            y = parse_binary_value(y_raw)
            subj = str(row.get(subject_col, "")).strip()
            if src is None or arm is None or y is None:
                dropped_rows += 1
                continue
            subjects.append(subj)
            source_counts[str(src)] += 1
            arm_counts[(src, arm)] += 1
            if src == 1 and arm == 0:
                y_current_control.append(y)
            elif src == 1 and arm == 1:
                y_current_treat.append(y)
            elif src == 0 and arm == 0:
                y_hist_control.append(y)
            elif src == 0 and arm == 1:
                y_hist_treat.append(y)

    unique_subjects = len({s for s in subjects if s != ""})
    duplicate_subject_rows = max(len([s for s in subjects if s != ""]) - unique_subjects, 0)
    if len(y_current_control) == 0 or len(y_current_treat) == 0:
        raise ValueError("Current data must include both arms (source=1, arm in {0,1}).")
    if len(y_hist_control) == 0:
        raise ValueError("Historical data must include control arm (source=0, arm=0).")

    data = {
        "y_current_control": np.array(y_current_control, dtype=float),
        "y_current_treat": np.array(y_current_treat, dtype=float),
        "y_hist_control": np.array(y_hist_control, dtype=float),
        "y_hist_treat": np.array(y_hist_treat, dtype=float),
        "source_counts": source_counts,
        "arm_counts": arm_counts,
        "total_rows": total_rows,
        "dropped_rows": dropped_rows,
        "unique_subjects": unique_subjects,
        "duplicate_subject_rows": duplicate_subject_rows,
    }
    return data


def build_method_params(args: Any) -> Any:
    """Build method parameters

    Args:
        args (Any): The arguments

    Returns:
        Any: The method parameters
    """
    return SimpleNamespace(
        alpha_pool=args.alpha_pool,
        beta_prior_alpha=args.beta_prior_alpha,
        beta_prior_beta=args.beta_prior_beta,
        vague_alpha=args.vague_alpha,
        vague_beta=args.vague_beta,
        normal_prior_mean=args.normal_prior_mean,
        normal_prior_var=args.normal_prior_var,
        vague_normal_var=args.vague_normal_var,
        fixed_lambdas=su.parse_float_list(args.fixed_lambdas),
        power_prior_lambdas=su.parse_float_list(args.power_prior_lambdas),
        commensurate_taus=su.parse_float_list(args.commensurate_taus),
        map_lambdas=su.parse_float_list(args.map_lambdas),
        map_weights=parse_optional_float_list(args.map_weights),
        rmap_epsilons=su.parse_float_list(args.rmap_epsilons),
        elastic_scales=su.parse_float_list(args.elastic_scales),
        uip_m_values=su.parse_float_list(args.uip_m_values),
        leap_prior_omega=float(args.leap_prior_omega),
        leap_nex_scale=float(args.leap_nex_scale),
        mem_prior_inclusion=float(args.mem_prior_inclusion),
        mem_tau=float(args.mem_tau),
        bhmoi_sharpness=float(args.bhmoi_sharpness),
        npb_concentration=None if args.npb_concentration is None else float(args.npb_concentration),
        npb_phi=float(args.npb_phi),
        npb_temperature=float(args.npb_temperature),
        npb_sharpness=float(args.npb_sharpness),
    )


def evaluate_methods_at_rho(
    rho: float,
    theta1: float,
    alpha: float,
    lambda_grid: int,
    method_specs: List[bm.MethodSpec],
    method_params: Any,
    arm0: Any,
    arm1: Any,
) -> List[Dict[str, Any]]:
    """Evaluate methods at a given rho

    Args:
        rho (float): The rho
        theta1 (float): The theta1
        alpha (float): The alpha
        lambda_grid (int): The lambda grid
        method_specs (List[bm.MethodSpec]): The method specifications
        method_params (Any): The method parameters
        arm0 (Any): The arm 0
        arm1 (Any): The arm 1

    Returns:
        List[Dict[str, Any]]: The rows
    """
    bond_lam0, bond_lam1, bond_kappa = bm.bond_lambda_star(
        outcome="binary",
        rho0=rho,
        rho1=rho,
        theta1=theta1,
        mu_c0=arm0.mean_c,
        mu_c1=arm1.mean_c,
        var_c0=arm0.var_c,
        var_c1=arm1.var_c,
        var_h0=arm0.var_h,
        var_h1=arm1.var_h,
        n_c0=arm0.n_c,
        n_c1=arm1.n_c,
        n_h0=arm0.n_h,
        n_h1=arm1.n_h,
        grid=lambda_grid,
    )
    z95 = float(norm.ppf(0.975))
    z_alpha = float(norm.isf(alpha))
    rows: list[dict[str, Any]] = []

    for method in method_specs:
        est0 = bm.estimate_arm_for_method(
            method=method,
            arm=arm0,
            outcome="binary",
            p=method_params,
            arm_index=0,
            bond_lambda0=bond_lam0,
            bond_lambda1=bond_lam1,
        )
        est1 = bm.estimate_arm_for_method(
            method=method,
            arm=arm1,
            outcome="binary",
            p=method_params,
            arm_index=1,
            bond_lambda0=bond_lam0,
            bond_lambda1=bond_lam1,
        )

        theta_hat = float(est1.mean - est0.mean)
        theta_sd = float(math.sqrt(max(est1.var + est0.var, 1e-12)))
        ci_low = float(theta_hat - z95 * theta_sd)
        ci_high = float(theta_hat + z95 * theta_sd)
        p_one = float(norm.sf(theta_hat / theta_sd))
        p_two = float(2.0 * norm.sf(abs(theta_hat / theta_sd)))
        lower_std = float(theta_hat - z_alpha * theta_sd)

        if method.family == "bond":
            b_plus = float(
                bm.bond_bias_plus(
                    outcome="binary",
                    w0=est0.w_eff,
                    w1=est1.w_eff,
                    rho0=rho,
                    rho1=rho,
                    mu_c0=arm0.mean_c,
                    mu_c1=arm1.mean_c,
                )
            )
            theta_adj = float(theta_hat - b_plus)
            z_robust = float(theta_adj / theta_sd)
            p_robust = float(norm.sf(z_robust))
            robust_lower = float(theta_adj - z_alpha * theta_sd)
            reject_robust = int(z_robust >= z_alpha)
            if abs(theta_hat) > 1e-12:
                bias_ratio = float(b_plus / abs(theta_hat))
            else:
                bias_ratio = float("inf")
        else:
            b_plus = float("nan")
            theta_adj = float("nan")
            z_robust = float("nan")
            p_robust = float("nan")
            robust_lower = float("nan")
            reject_robust = -1
            bias_ratio = float("nan")

        row = {
            "rho": float(rho),
            "method": method.name,
            "mu0_mean": float(est0.mean),
            "mu0_sd": float(math.sqrt(max(est0.var, 1e-12))),
            "mu1_mean": float(est1.mean),
            "mu1_sd": float(math.sqrt(max(est1.var, 1e-12))),
            "theta_mean": theta_hat,
            "theta_sd": theta_sd,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "ci95_width": float(ci_high - ci_low),
            "z": float(theta_hat / theta_sd),
            "p_one_sided": p_one,
            "p_two_sided": p_two,
            "lower_one_sided_1_minus_alpha": lower_std,
            "lambda0_eff": float(est0.lambda_eff),
            "lambda1_eff": float(est1.lambda_eff),
            "w0_eff": float(est0.w_eff),
            "w1_eff": float(est1.w_eff),
            "n_hist_borrowed0": float(est0.lambda_eff * arm0.n_h),
            "n_hist_borrowed1": float(est1.lambda_eff * arm1.n_h),
            "bond_lambda0_star": float(bond_lam0),
            "bond_lambda1_star": float(bond_lam1),
            "bond_kappa": float(bond_kappa),
            "bond_b_plus": b_plus,
            "bond_theta_adjusted": theta_adj,
            "bond_z_robust": z_robust,
            "bond_p_one_sided_robust": p_robust,
            "bond_lower_one_sided_1_minus_alpha": robust_lower,
            "bond_bias_over_abs_theta": bias_ratio,
            "reject_one_sided_alpha": int((theta_hat / theta_sd) >= z_alpha),
            "reject_bond_robust_alpha": reject_robust,
        }
        rows.append(row)

    current_sd = next((float(r["theta_sd"]) for r in rows if r["method"] == "Current-only"), float("nan"))
    for row in rows:
        row["theta_sd_over_current"] = float(row["theta_sd"] / current_sd) if current_sd > 0 else float("nan")
        row["ci95_width_over_current"] = (
            float(row["ci95_width"] / (2.0 * z95 * current_sd)) if current_sd > 0 else float("nan")
        )
    return rows


def to_float_series(rows: Iterable[Dict[str, Any]], key: str) -> NDArray[np.float64]:
    """Convert a series of rows to a float series

    Args:
        rows (Iterable[Dict[str, Any]]): The rows
        key (str): The key

    Returns:
        NDArray[np.float64]: The float series
    """
    vals = []
    for row in rows:
        val = row.get(key, float("nan"))
        try:
            vals.append(float(val))
        except (TypeError, ValueError):
            vals.append(float("nan"))
    return np.array(vals, dtype=float)


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Save a list of rows to a CSV file

    Args:
        path (Path): The path to the CSV file
        rows (List[Dict[str, Any]]): The rows
    """
    if len(rows) == 0:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric_vs_rho(
    rows: List[Dict[str, Any]],
    methods: List[str],
    style_map: Dict[str, Dict[str, Any]],
    metric: str,
    ylabel: str,
    outpath: Path,
    ylog: bool = False,
) -> None:
    """Plot a metric vs rho

    Args:
        rows (List[Dict[str, Any]]): The rows
        methods (List[str]): The methods
        style_map (Dict[str, Dict[str, Any]]): The style map
        metric (str): The metric
        ylabel (str): The y-label
        outpath (Path): The output path
        ylog (bool): Whether to log the y-axis
    """
    fig, ax = plt.subplots(figsize=(11.8, 8.6))
    rho_grid = sorted({float(r["rho"]) for r in rows})
    for method in methods:
        method_rows = [r for r in rows if r["method"] == method]
        if len(method_rows) == 0:
            continue
        method_rows = sorted(method_rows, key=lambda x: float(x["rho"]))
        y = to_float_series(method_rows, metric)
        style = style_map.get(method, {})
        ax.plot(
            rho_grid,
            y,
            label=pu.format_method_label(method),
            marker=style.get("marker", "o"),
            linestyle=style.get("linestyle", "-"),
            color=style.get("color", None),
            linewidth=style.get("linewidth", 2.8),
            markersize=style.get("markersize", 8.0),
        )
    if ylog:
        ax.set_yscale("log")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    pu.save_legend_pdf(outpath.with_name(f"{outpath.stem}_legend.pdf"), handles, labels, ncol=5, fontsize=22.0)
    fig.tight_layout()
    fig.savefig(outpath, format="pdf")
    plt.close(fig)


def plot_theta_ci_focus(
    rows: List[Dict[str, Any]],
    methods: List[str],
    style_map: Dict[str, Dict[str, Any]],
    outpath: Path,
) -> None:
    """Plot the theta CI focus

    Args:
        rows (List[Dict[str, Any]]): The rows
        methods (List[str]): The methods
        style_map (Dict[str, Dict[str, Any]]): The style map
        outpath (Path): The output path
    """
    fig, ax = plt.subplots(figsize=(12.0, 8.8))
    for method in methods:
        method_rows = sorted([r for r in rows if r["method"] == method], key=lambda x: float(x["rho"]))
        if len(method_rows) == 0:
            continue
        rho = to_float_series(method_rows, "rho")
        theta = to_float_series(method_rows, "theta_mean")
        low = to_float_series(method_rows, "ci95_low")
        high = to_float_series(method_rows, "ci95_high")
        yerr = np.vstack((theta - low, high - theta))
        style = style_map.get(method, {})
        ax.errorbar(
            rho,
            theta,
            yerr=yerr,
            label=pu.format_method_label(method),
            marker=style.get("marker", "o"),
            linestyle=style.get("linestyle", "-"),
            color=style.get("color", None),
            linewidth=style.get("linewidth", 2.8),
            markersize=style.get("markersize", 8.0),
            capsize=3.0,
        )
    ax.axhline(0.0, color="black", linestyle=(0, (4, 2)), alpha=0.7, linewidth=2.2)
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$\hat{\theta}$ with 95\% CI")
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    pu.save_legend_pdf(outpath.with_name(f"{outpath.stem}_legend.pdf"), handles, labels, ncol=4, fontsize=22.0)
    fig.tight_layout()
    fig.savefig(outpath, format="pdf")
    plt.close(fig)


def plot_bond_sensitivity(rows: List[Dict[str, Any]], outdir: Path) -> None:
    """Plot the bond sensitivity

    Args:
        rows (List[Dict[str, Any]]): The rows
        outdir (Path): The output directory
    """
    bond_rows = sorted([r for r in rows if r["method"] == "BOND"], key=lambda x: float(x["rho"]))
    if len(bond_rows) == 0:
        return
    rho = to_float_series(bond_rows, "rho")

    fig1, ax1 = plt.subplots(figsize=(10.8, 7.2))
    ax1.plot(rho, to_float_series(bond_rows, "bond_lambda0_star"), marker="o", linewidth=3.0, label=r"$\lambda_0^*$")
    ax1.plot(
        rho,
        to_float_series(bond_rows, "bond_lambda1_star"),
        marker="D",
        linewidth=3.0,
        linestyle="--",
        label=r"$\lambda_1^*$",
    )
    ax1.plot(rho, to_float_series(bond_rows, "w0_eff"), marker="s", linewidth=3.0, linestyle="-.", label=r"$w_0$")
    ax1.plot(rho, to_float_series(bond_rows, "w1_eff"), marker="^", linewidth=3.0, linestyle=":", label=r"$w_1$")
    ax1.set_xlabel(r"$\rho$")
    ax1.set_ylabel("Borrowing level")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, alpha=0.3)
    h1, l1 = ax1.get_legend_handles_labels()
    pu.save_legend_pdf(outdir / "bond_weights_vs_rho_legend.pdf", h1, l1, ncol=4, fontsize=22.0)
    fig1.tight_layout()
    fig1.savefig(outdir / "bond_weights_vs_rho.pdf", format="pdf")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(11.5, 8.0))
    ax2.plot(rho, to_float_series(bond_rows, "theta_mean"), marker="o", linewidth=3.0, label=r"$\hat{\theta}$")
    ax2.plot(
        rho, to_float_series(bond_rows, "ci95_low"), marker="v", linewidth=2.4, linestyle="--", label=r"95\% CI low"
    )
    ax2.plot(
        rho, to_float_series(bond_rows, "ci95_high"), marker="^", linewidth=2.4, linestyle="--", label=r"95\% CI high"
    )
    ax2.plot(
        rho,
        to_float_series(bond_rows, "bond_theta_adjusted"),
        marker="s",
        linewidth=3.0,
        linestyle="-.",
        label=r"$\hat{\theta}-b_+$",
    )
    ax2.axhline(0.0, color="black", linestyle=(0, (4, 2)), alpha=0.7, linewidth=2.0)
    ax2.set_xlabel(r"$\rho$")
    ax2.set_ylabel("Effect scale")
    ax2.grid(True, alpha=0.3)
    h2, l2 = ax2.get_legend_handles_labels()
    pu.save_legend_pdf(outdir / "bond_theta_ci_vs_rho_legend.pdf", h2, l2, ncol=4, fontsize=22.0)
    fig2.tight_layout()
    fig2.savefig(outdir / "bond_theta_ci_vs_rho.pdf", format="pdf")
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(11.0, 7.6))
    p_std = np.clip(to_float_series(bond_rows, "p_one_sided"), 1e-12, 1.0)
    p_rob = np.clip(to_float_series(bond_rows, "bond_p_one_sided_robust"), 1e-12, 1.0)
    ax3.plot(rho, p_std, marker="o", linewidth=3.0, label="One-sided p")
    ax3.plot(rho, p_rob, marker="s", linewidth=3.0, linestyle="--", label="Robust one-sided p")
    ax3.set_yscale("log")
    ax3.set_xlabel(r"$\rho$")
    ax3.set_ylabel("p-value (log scale)")
    ax3.grid(True, alpha=0.3)
    h3, l3 = ax3.get_legend_handles_labels()
    pu.save_legend_pdf(outdir / "bond_pvalue_vs_rho_legend.pdf", h3, l3, ncol=2, fontsize=22.0)
    fig3.tight_layout()
    fig3.savefig(outdir / "bond_pvalue_vs_rho.pdf", format="pdf")
    plt.close(fig3)


def parse_args() -> Any:
    """Parse the arguments

    Returns:
        Any: The arguments
    """
    parser = argparse.ArgumentParser(description="Real-world borrowing analysis with rho sensitivity.")
    parser.add_argument("--data", type=Path, default=Path("real-world/df_merged.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("outputs/real_world_analysis"))
    parser.add_argument("--subject-col", type=str, default="SUBJID")
    parser.add_argument("--source-col", type=str, default="DATA_SOURCE")
    parser.add_argument("--arm-col", type=str, default="ARM")
    parser.add_argument("--outcome-col", type=str, default="OUR_OUTCOME")
    parser.add_argument("--rho-grid", type=str, default="0,0.01,0.02,0.05,0.1,0.15,0.2")
    parser.add_argument("--alpha", type=float, default=0.025, help="One-sided significance level.")
    parser.add_argument("--theta1", type=float, default=0.30, help="Target alternative for BOND kappa.")
    parser.add_argument("--lambda-grid", type=int, default=401)

    parser.add_argument("--alpha-pool", type=float, default=0.1)
    parser.add_argument("--fixed-lambdas", type=str, default="0.25,0.5,0.75")
    parser.add_argument("--power-prior-lambdas", type=str, default="0.5")
    parser.add_argument("--commensurate-taus", type=str, default="1.0")
    parser.add_argument("--map-lambdas", type=str, default="0.25,1.0")
    parser.add_argument("--map-weights", type=str, default="")
    parser.add_argument("--rmap-epsilons", type=str, default="0.2")
    parser.add_argument("--elastic-scales", type=str, default="1.0")
    parser.add_argument("--uip-m-values", type=str, default="100")
    parser.add_argument("--leap-prior-omega", type=float, default=0.5)
    parser.add_argument("--leap-nex-scale", type=float, default=9.0)
    parser.add_argument("--mem-prior-inclusion", type=float, default=0.5)
    parser.add_argument("--mem-tau", type=float, default=1.0)
    parser.add_argument("--bhmoi-sharpness", type=float, default=1.0)
    parser.add_argument("--npb-concentration", type=float, default=None)
    parser.add_argument("--npb-phi", type=float, default=0.5)
    parser.add_argument("--npb-temperature", type=float, default=1.0)
    parser.add_argument("--npb-sharpness", type=float, default=1.0)
    parser.add_argument("--beta-prior-alpha", type=float, default=1.0)
    parser.add_argument("--beta-prior-beta", type=float, default=1.0)
    parser.add_argument("--vague-alpha", type=float, default=1.0)
    parser.add_argument("--vague-beta", type=float, default=1.0)
    parser.add_argument("--normal-prior-mean", type=float, default=0.0)
    parser.add_argument("--normal-prior-var", type=float, default=1e8)
    parser.add_argument("--vague-normal-var", type=float, default=1e8)
    return parser.parse_args()


def validate_args(args: Any) -> List[float]:
    """Validate the arguments

    Args:
        args (Any): The arguments

    Returns:
        List[float]: The rho grid
    """
    if not (0.0 < args.alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")
    if args.theta1 <= 0.0:
        raise ValueError("theta1 must be > 0")
    if args.lambda_grid < 11:
        raise ValueError("lambda-grid must be >= 11")
    rho_grid = sorted({float(x) for x in su.parse_float_list(args.rho_grid)})
    if len(rho_grid) == 0:
        raise ValueError("rho-grid must be non-empty")
    if any(rho < 0.0 for rho in rho_grid):
        raise ValueError("rho-grid must be >= 0")
    return rho_grid


def main() -> None:
    args = parse_args()
    rho_grid = validate_args(args)
    args.outdir.mkdir(parents=True, exist_ok=True)
    pu.set_plot_style()

    data = load_real_world_binary_data(
        csv_path=args.data,
        source_col=args.source_col,
        arm_col=args.arm_col,
        outcome_col=args.outcome_col,
        subject_col=args.subject_col,
    )
    arm0 = build_arm_summary_from_binary(data["y_current_control"], data["y_hist_control"])
    arm1 = build_arm_summary_from_binary(data["y_current_treat"], data["y_hist_treat"])
    method_params = build_method_params(args)
    method_specs = bm.build_method_specs(method_params)
    method_order = [m.name for m in method_specs]
    focus_methods = pu.select_focus_methods(method_order)
    style_map = pu.build_method_styles(method_order)

    all_rows: list[dict[str, Any]] = []
    for rho in rho_grid:
        all_rows.extend(
            evaluate_methods_at_rho(
                rho=float(rho),
                theta1=float(args.theta1),
                alpha=float(args.alpha),
                lambda_grid=int(args.lambda_grid),
                method_specs=method_specs,
                method_params=method_params,
                arm0=arm0,
                arm1=arm1,
            )
        )

    focus_rows = [r for r in all_rows if r["method"] in set(focus_methods)]
    save_csv(args.outdir / "analysis_all.csv", all_rows)
    save_csv(args.outdir / "analysis_focus.csv", focus_rows)

    summary_rows = [
        {
            "metric": "total_rows",
            "value": data["total_rows"],
        },
        {
            "metric": "dropped_rows_invalid_or_missing",
            "value": data["dropped_rows"],
        },
        {
            "metric": "unique_subjects_nonempty",
            "value": data["unique_subjects"],
        },
        {
            "metric": "duplicate_subject_rows_nonempty",
            "value": data["duplicate_subject_rows"],
        },
        {
            "metric": "n_current_control",
            "value": arm0.n_c,
        },
        {
            "metric": "n_current_treatment",
            "value": arm1.n_c,
        },
        {
            "metric": "n_historical_control",
            "value": arm0.n_h,
        },
        {
            "metric": "n_historical_treatment",
            "value": arm1.n_h,
        },
        {
            "metric": "orr_current_control",
            "value": arm0.mean_c,
        },
        {
            "metric": "orr_current_treatment",
            "value": arm1.mean_c,
        },
        {
            "metric": "orr_historical_control",
            "value": arm0.mean_h,
        },
        {
            "metric": "orr_historical_treatment",
            "value": arm1.mean_h,
        },
        {
            "metric": "control_gap_abs_|mu_h0-mu_c0|",
            "value": abs(arm0.mean_h - arm0.mean_c),
        },
    ]
    save_csv(args.outdir / "dataset_summary.csv", summary_rows)

    plot_metric_vs_rho(
        rows=all_rows,
        methods=method_order,
        style_map=style_map,
        metric="theta_mean",
        ylabel=r"$\hat{\theta}$",
        outpath=args.outdir / "theta_vs_rho_all.pdf",
    )
    plot_metric_vs_rho(
        rows=focus_rows,
        methods=focus_methods,
        style_map=style_map,
        metric="theta_mean",
        ylabel=r"$\hat{\theta}$",
        outpath=args.outdir / "theta_vs_rho_focus.pdf",
    )
    plot_metric_vs_rho(
        rows=all_rows,
        methods=method_order,
        style_map=style_map,
        metric="ci95_width",
        ylabel=r"95\% CI width",
        outpath=args.outdir / "ci_width_vs_rho_all.pdf",
    )
    plot_metric_vs_rho(
        rows=focus_rows,
        methods=focus_methods,
        style_map=style_map,
        metric="ci95_width",
        ylabel=r"95\% CI width",
        outpath=args.outdir / "ci_width_vs_rho_focus.pdf",
    )
    plot_metric_vs_rho(
        rows=all_rows,
        methods=method_order,
        style_map=style_map,
        metric="p_one_sided",
        ylabel="One-sided p-value",
        outpath=args.outdir / "pvalue_vs_rho_all.pdf",
        ylog=True,
    )
    plot_metric_vs_rho(
        rows=focus_rows,
        methods=focus_methods,
        style_map=style_map,
        metric="p_one_sided",
        ylabel="One-sided p-value",
        outpath=args.outdir / "pvalue_vs_rho_focus.pdf",
        ylog=True,
    )
    plot_metric_vs_rho(
        rows=all_rows,
        methods=method_order,
        style_map=style_map,
        metric="w0_eff",
        ylabel=r"$w_0$",
        outpath=args.outdir / "w0_vs_rho_all.pdf",
    )
    plot_metric_vs_rho(
        rows=focus_rows,
        methods=focus_methods,
        style_map=style_map,
        metric="w0_eff",
        ylabel=r"$w_0$",
        outpath=args.outdir / "w0_vs_rho_focus.pdf",
    )
    plot_metric_vs_rho(
        rows=all_rows,
        methods=method_order,
        style_map=style_map,
        metric="w1_eff",
        ylabel=r"$w_1$",
        outpath=args.outdir / "w1_vs_rho_all.pdf",
    )
    plot_metric_vs_rho(
        rows=focus_rows,
        methods=focus_methods,
        style_map=style_map,
        metric="w1_eff",
        ylabel=r"$w_1$",
        outpath=args.outdir / "w1_vs_rho_focus.pdf",
    )
    plot_metric_vs_rho(
        rows=all_rows,
        methods=method_order,
        style_map=style_map,
        metric="n_hist_borrowed0",
        ylabel="Borrowed historical n (arm 0)",
        outpath=args.outdir / "n_hist_borrowed0_vs_rho_all.pdf",
    )
    plot_metric_vs_rho(
        rows=focus_rows,
        methods=focus_methods,
        style_map=style_map,
        metric="n_hist_borrowed0",
        ylabel="Borrowed historical n (arm 0)",
        outpath=args.outdir / "n_hist_borrowed0_vs_rho_focus.pdf",
    )
    plot_theta_ci_focus(
        rows=focus_rows,
        methods=focus_methods,
        style_map=style_map,
        outpath=args.outdir / "theta_ci95_vs_rho_focus.pdf",
    )
    plot_bond_sensitivity(all_rows, args.outdir)

    print("Real-world borrowing analysis completed")
    print(f"Output directory: {args.outdir}")
    print(f"All-method table: {args.outdir / 'analysis_all.csv'}")
    print(f"Focus-method table: {args.outdir / 'analysis_focus.csv'}")
    print(f"Dataset summary: {args.outdir / 'dataset_summary.csv'}")


if __name__ == "__main__":
    main()
