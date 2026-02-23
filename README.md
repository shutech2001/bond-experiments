# Experiments for BOND (Borrowing under Optimal Nonparametric Distributional robustness)
Materials for "[**Error-Controlled Borrowing from External Data Using Wasserstein Ambiguity Sets**](https://arxiv.org/abs/2602.NNNNN)".

## What is in this repo
- Simulation study code: `scripts/dro_borrowing_simulation.py`
- Borrowing method implementations: `scripts/borrowing_methods.py`
- Real-world binary analysis: `scripts/real_world_borrowing_analysis.py`
- Replot utility from CSV outputs: `scripts/replot_from_csv.py`

### Requirements and setup
```
# clone the repository
git clone git@github.com:shutech2001/bond-experiments.git

# build the environment with poetry
poetry install

# activate virtual environment
eval $(poetry env activate)

# [Option] to activate the interpreter, select the following output as the interpreter
poetry env info --path
```

### Simulation quickstart
```bash
python scripts/dro_borrowing_simulation.py
```

#### Common options
- `--n-jobs`: number of worker processes for gamma-level parallelization.
- `--stages`: `oracle`, `data`, or both (comma-separated).
- `--rho-multipliers`: multipliers for data-driven rho in `data` stage.
- `--outcomes`: `continuous,binary` (or subset).
- `--scenarios`: subset of `S0,S1,S2,S3`.
- `--n-historical-grid`: evaluate multiple historical sample sizes.
- `--bond-calibration-mode`: `replicate` (default) or `design`.

#### Examples
```bash
# data-stage only, binary S0 only, with 8 workers
python scripts/dro_borrowing_simulation.py \
  --stages data \
  --outcomes binary \
  --scenarios S0 \
  --n-jobs 8

# multiple historical sizes
python scripts/dro_borrowing_simulation.py \
  --n-historical-grid 250,500,1000
```

#### Main outputs (`--outdir`, default: `outputs/dro_simulation`)
- `summary.csv`: per-(n_historical, outcome, scenario, stage, gamma, method) Type-I and power.
- `worst_case.csv`: worst-case summary across gamma.
- `bond_lambda.csv`: BOND rho/lambda/weight/kappa per gamma.
- `type1_power_*.pdf`, `type1_power_focus_*.pdf`, `lambda_*.pdf` plus legend PDFs.

### Real-world analysis (binary)
```bash
python scripts/real_world_borrowing_analysis.py \
  --data real-world/merged_df.csv \
  --outdir outputs/real_world_analysis
```

Notes:
- The script evaluates methods across `--rho-grid` (default: `0,0.01,0.02,0.05,0.1,0.15,0.2`).
- If your merged file path/name differs, pass it via `--data`.

Main outputs:
- `analysis_all.csv`, `analysis_focus.csv`
- `dataset_summary.csv`
- `*_vs_rho*.pdf` and corresponding `*_legend.pdf`

### Replot figures from CSV
```bash
python scripts/replot_from_csv.py \
  --summary-csv outputs/dro_simulation/summary.csv \
  --lambda-csv outputs/dro_simulation/bond_lambda.csv
```

### Method-only usage
If you only want to use borrowing methods (without running full simulations), use `scripts/borrowing_methods.py` directly.

```python
import sys
from types import SimpleNamespace

sys.path.append("scripts")
import borrowing_methods as bm

# Per-arm summary input (continuous example)
arm0 = SimpleNamespace(
    n_c=100, n_h=500,
    mean_c=0.10, mean_h=0.25,
    var_c=1.00, var_h=1.05,
    sum_c=10.0, sum_h=125.0,
    succ_c=0, succ_h=0,
)
arm1 = SimpleNamespace(
    n_c=100, n_h=500,
    mean_c=0.40, mean_h=0.45,
    var_c=1.10, var_h=1.00,
    sum_c=40.0, sum_h=225.0,
    succ_c=0, succ_h=0,
)

# Hyperparameters used by method families
p = SimpleNamespace(
    alpha_pool=0.10,
    beta_prior_alpha=1.0,
    beta_prior_beta=1.0,
    vague_alpha=1.0,
    vague_beta=1.0,
    normal_prior_mean=0.0,
    normal_prior_var=1e8,
    vague_normal_var=1e8,
    map_lambdas=[0.25, 1.0],
    map_weights=[0.5, 0.5],
    leap_prior_omega=0.5,
    leap_nex_scale=9.0,
    mem_prior_inclusion=0.5,
    mem_tau=1.0,
    bhmoi_sharpness=1.0,
    npb_concentration=None,
    npb_phi=0.5,
    npb_temperature=1.0,
    npb_sharpness=1.0,
)

outcome = "continuous"

# Unified runner signature:
# runner(arm, outcome, p, param=None, arm_index=0, bond_lambda0=0.0, bond_lambda1=0.0)
m_current = bm.method_current(arm0, outcome, p)
m_power = bm.method_power_prior(arm0, outcome, p, param=0.5)
m_comm = bm.method_commensurate_prior(arm0, outcome, p, param=1.0)

# BOND: choose lambda* first, then call method_bond
lam0, lam1, _ = bm.bond_lambda_star(
    outcome=outcome,
    rho0=0.10, rho1=0.10, theta1=0.30,
    mu_c0=arm0.mean_c, mu_c1=arm1.mean_c,
    var_c0=arm0.var_c, var_c1=arm1.var_c,
    var_h0=arm0.var_h, var_h1=arm1.var_h,
    n_c0=arm0.n_c, n_c1=arm1.n_c,
    n_h0=arm0.n_h, n_h1=arm1.n_h,
    grid=401,
)
m_bond_arm0 = bm.method_bond(arm0, outcome, p, arm_index=0, bond_lambda0=lam0, bond_lambda1=lam1)
m_bond_arm1 = bm.method_bond(arm1, outcome, p, arm_index=1, bond_lambda0=lam0, bond_lambda1=lam1)

print("Current-only mean:", m_current.mean)
print("Power prior mean:", m_power.mean)
print("Commensurate mean:", m_comm.mean)
print("BOND lambdas:", lam0, lam1)
print("BOND arm means:", m_bond_arm0.mean, m_bond_arm1.mean)
```

Notes:
- Method families are defined in `METHOD_RUNNERS` in `scripts/borrowing_methods.py`.
- For dispatch from `MethodSpec`, use `estimate_arm_for_method(...)`.

## Contact
If you have any question, please feel free to contact: `tamano-shu212@g.ecc.u-tokyo.ac.jp`
