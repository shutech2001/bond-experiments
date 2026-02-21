# BOND
Materials for "[**Error-Controlled Borrowing from External Data Using Wasserstein Ambiguity Sets**](https://arxiv.org/abs/2602.NNNNN)".

## What is this repo?

### Requirements and Setup
```
# clone the repository
git clone git@github.com:shutech2001/borrowing-based-on-dro.git

# build the environment with poetry
poetry install

# activate virtual environment
eval $(poetry env activate)

# [Option] to activate the interpreter, select the following output as the interpreter.
poetry env info --path
```

## Simulation quickstart
```bash
# run a regular scenario
python scripts/dro_borrowing_simulation.py
```

Outputs:
- `outputs/dro_simulation`: 

## Method-only usage
If you only want to use borrowing methods (without running the full simulation), use `scripts/borrowing_methods.py` directly.

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

# Hyperparameters used by different methods
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
    exnex_prior_ex=0.7,
    mem_prior_inclusion=0.5,
    mem_tau=1.0,
    bhmoi_sharpness=1.0,
    npb_temperature=1.0,
    npb_sharpness=1.0,
)

outcome = "continuous"

# Any method can be called via unified runner signature:
# runner(arm, outcome, p, param=None, arm_index=0, bond_lambda0=0.0, bond_lambda1=0.0)
m_current = bm.method_current(arm0, outcome, p)
m_power = bm.method_power_prior(arm0, outcome, p, param=0.5)
m_comm = bm.method_commensurate_prior(arm0, outcome, p, param=1.0)

# BOND: first choose lambda* (arm0/control, arm1/treatment), then run method_bond
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
- Available families are in `METHOD_RUNNERS` inside `scripts/borrowing_methods.py`.
- If you already have `MethodSpec`, call `estimate_arm_for_method(...)` for dispatch by method family.

## Citation
```text
@article{kimura2026worst,
    author={Kimura, Yui and Tamano, Shu},
    journal={arXiv preprint arXiv:2602.NNNNN},
    title={Error-Controlled Borrowing from External Data Using {Wasserstein} Ambiguity Sets},
    year={2026},
}
```

## Contact

If you have any question, please feel free to contact: tamano-shu212@g.ecc.u-tokyo.ac.jp
