from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import plot_utils as pu


SummaryKey = Tuple[int, str, str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate simulation plots from summary/lambda CSV files.")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("outputs/dro_simulation/summary.csv"),
        help="Path to summary.csv",
    )
    parser.add_argument(
        "--lambda-csv",
        type=Path,
        default=Path("outputs/dro_simulation/bond_lambda.csv"),
        help="Path to bond_lambda.csv",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory for PDF figures (default: same directory as summary CSV).",
    )
    parser.add_argument("--alpha", type=float, default=0.025, help="Nominal Type-I level for reference line.")
    return parser.parse_args()


def scenario_hist_arms(scenario: str) -> str:
    # Matches data-generation convention in dro_borrowing_simulation.py.
    return "control" if scenario == "S3" else "both"


def read_summary_rows(path: Path) -> Tuple[
    Dict[SummaryKey, Dict[float, Dict[str, Tuple[float, float]]]],
    Dict[SummaryKey, List[str]],
]:
    grouped: Dict[SummaryKey, Dict[float, Dict[str, Tuple[float, float]]]] = defaultdict(dict)
    method_order: Dict[SummaryKey, List[str]] = defaultdict(list)

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"n_historical", "outcome", "scenario", "stage", "gamma", "method", "type1", "power"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"summary CSV is missing columns: {sorted(missing)}")

        for row in reader:
            n_h = int(row["n_historical"])
            key: SummaryKey = (n_h, row["outcome"], row["scenario"], row["stage"])
            gamma = float(row["gamma"])
            method = row["method"]
            type1 = float(row["type1"])
            power = float(row["power"])
            if gamma not in grouped[key]:
                grouped[key][gamma] = {}
            grouped[key][gamma][method] = (type1, power)
            if method not in method_order[key]:
                method_order[key].append(method)

    return grouped, method_order


def read_lambda_rows(path: Path) -> Dict[SummaryKey, List[Tuple[float, float, float, float, float, float]]]:
    grouped: Dict[SummaryKey, List[Tuple[float, float, float, float, float, float]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "n_historical",
            "outcome",
            "scenario",
            "stage",
            "gamma",
            "lambda0",
            "lambda1",
            "w0",
            "w1",
            "kappa",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"bond_lambda CSV is missing columns: {sorted(missing)}")

        for row in reader:
            n_h = int(row["n_historical"])
            key: SummaryKey = (n_h, row["outcome"], row["scenario"], row["stage"])
            grouped[key].append(
                (
                    float(row["gamma"]),
                    float(row["lambda0"]),
                    float(row["lambda1"]),
                    float(row["w0"]),
                    float(row["w1"]),
                    float(row["kappa"]),
                )
            )

    for key in grouped:
        grouped[key].sort(key=lambda x: x[0])
    return grouped


def build_type1_power_tables(
    gamma_map: Dict[float, Dict[str, Tuple[float, float]]],
    methods: List[str],
) -> Tuple[List[float], List[Dict[str, float]], List[Dict[str, float]]]:
    gamma_grid = sorted(gamma_map.keys())
    type1_table: List[Dict[str, float]] = []
    power_table: List[Dict[str, float]] = []
    for gamma in gamma_grid:
        by_method = gamma_map[gamma]
        type1_row: Dict[str, float] = {}
        power_row: Dict[str, float] = {}
        for method in methods:
            if method not in by_method:
                continue
            type1_row[method] = by_method[method][0]
            power_row[method] = by_method[method][1]
        type1_table.append(type1_row)
        power_table.append(power_row)
    return gamma_grid, type1_table, power_table


def main() -> None:
    args = parse_args()
    summary_csv = args.summary_csv
    lambda_csv = args.lambda_csv

    if not summary_csv.exists():
        raise FileNotFoundError(f"summary CSV not found: {summary_csv}")
    if not lambda_csv.exists():
        raise FileNotFoundError(f"bond lambda CSV not found: {lambda_csv}")

    outdir = args.outdir if args.outdir is not None else summary_csv.parent
    outdir.mkdir(parents=True, exist_ok=True)

    pu.set_plot_style()
    summary_grouped, method_order_map = read_summary_rows(summary_csv)
    lambda_grouped = read_lambda_rows(lambda_csv)

    num_type1_power = 0
    num_lambda = 0

    for key in sorted(summary_grouped.keys()):
        n_h, outcome, scenario, stage = key
        methods = method_order_map[key]
        gamma_grid, type1_table, power_table = build_type1_power_tables(summary_grouped[key], methods)
        if len(gamma_grid) == 0:
            continue

        style_map = pu.build_method_styles(methods)
        focus_methods = pu.select_focus_methods(methods)
        stage_file = stage.replace(".", "p")

        pu.save_type1_power_figure(
            outdir=outdir,
            gamma_grid=gamma_grid,
            type1_table=type1_table,
            power_table=power_table,
            methods=methods,
            style_map=style_map,
            alpha=float(args.alpha),
            filename_stem=f"type1_power_{outcome}_{scenario}_{stage_file}_nH{n_h}",
            figsize=(12.8, 9.8),
        )
        pu.save_type1_power_figure(
            outdir=outdir,
            gamma_grid=gamma_grid,
            type1_table=type1_table,
            power_table=power_table,
            methods=focus_methods,
            style_map=style_map,
            alpha=float(args.alpha),
            filename_stem=f"type1_power_focus_{outcome}_{scenario}_{stage_file}_nH{n_h}",
            figsize=(11.8, 8.9),
        )
        num_type1_power += 2

        lambda_table = lambda_grouped.get(key, [])
        if len(lambda_table) > 0:
            pu.save_lambda_figure(
                outdir=outdir,
                lambda_table=lambda_table,
                hist_arms=scenario_hist_arms(scenario),
                filename_stem=f"lambda_{outcome}_{scenario}_{stage_file}_nH{n_h}",
            )
            num_lambda += 1

    print(f"Replotted {num_type1_power} type1/power figures and {num_lambda} lambda figures into: {outdir}")


if __name__ == "__main__":
    main()
