from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from scipy.special import betaln, expit, logsumexp  # type: ignore
from scipy.stats import norm, wasserstein_distance  # type: ignore
import matplotlib
import matplotlib.pyplot as plt

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
matplotlib.use("Agg")


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 12,
            "mathtext.fontset": "stix",
            "text.usetex": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def math_label(expr: str) -> str:
    return rf"$${expr}$$"


def z_alpha(alpha: float) -> float:
    return float(norm.isf(alpha))


def clamp_prob(p: float) -> float:
    return max(1e-6, min(1.0 - 1e-6, p))


def w_from_lambda(lam: float, n_c: int, n_h: int) -> float:
    if n_h <= 0 or lam <= 0.0:
        return 0.0
    return float((lam * n_h) / (n_c + lam * n_h))


def lambda_from_uip(m: float, n_h: int) -> float:
    if n_h <= 0:
        return 0.0
    return float(np.clip(m / n_h, 0.0, 1.0))


def var_mean_binary(mean: float, n: int) -> float:
    if n <= 0:
        return 0.0
    p = clamp_prob(mean)
    return p * (1.0 - p) / n


def var_mean_cont(var_y: float, n: int) -> float:
    if n <= 1:
        return 0.0
    return var_y / n


def wasserstein_1d(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    return float(wasserstein_distance(x, y))


def calibrate_binary_tau(
    beta0: float,
    beta: np.ndarray,
    eta: np.ndarray,
    theta_target: float,
    p_dim: int,
    mc_size: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(mc_size, p_dim))
    lin0 = beta0 + x @ beta
    lin1_base = beta0 + x @ (beta + eta)
    mu0 = float(np.mean(expit(lin0)))
    target_mu1 = mu0 + theta_target
    if not (1e-6 < target_mu1 < 1.0 - 1e-6):
        raise ValueError(
            f"Target mean difference out of feasible range for binary outcome: theta={theta_target}, mu0={mu0}"
        )

    def f(tau: float) -> float:
        mu1 = float(np.mean(expit(lin1_base + tau)))
        return mu1 - target_mu1

    lo = -12.0
    hi = 12.0
    flo = f(lo)
    fhi = f(hi)
    while flo > 0.0:
        hi = lo
        lo -= 8.0
        flo = f(lo)
        if lo < -80.0:
            break
    while fhi < 0.0:
        lo = hi
        hi += 8.0
        fhi = f(hi)
        if hi > 80.0:
            break

    for _ in range(100):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if abs(fmid) < 1e-8:
            return float(mid)
        if fmid > 0.0:
            hi = mid
        else:
            lo = mid
    return float(0.5 * (lo + hi))


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    cov_shift: bool
    effect_mod: bool
    control_drift: bool
    hist_arms: str  # "both" or "control"


SCENARIOS = {
    "S0": Scenario(
        key="S0",
        label="S0: Commensurate",
        cov_shift=False,
        effect_mod=False,
        control_drift=False,
        hist_arms="both",
    ),
    "S2": Scenario(
        key="S2",
        label="S2: Covariate shift + effect mod",
        cov_shift=True,
        effect_mod=True,
        control_drift=False,
        hist_arms="both",
    ),
    "S1": Scenario(
        key="S1",
        label="S1: Covariate shift only",
        cov_shift=True,
        effect_mod=False,
        control_drift=False,
        hist_arms="both",
    ),
    "S3": Scenario(
        key="S3",
        label="S3: Control drift (hist. control-only)",
        cov_shift=False,
        effect_mod=False,
        control_drift=True,
        hist_arms="control",
    ),
}


@dataclass(frozen=True)
class DGPParams:
    p: int
    beta: NDArray[np.float64]
    beta0_cont: float
    beta0_bin: float
    eta_effect: NDArray[np.float64]
    sigma: float
    pi: float


@dataclass(frozen=True)
class Moments:
    mu_c0: float
    mu_c1: float
    mu_h0: float
    mu_h1: float
    var_c0: float
    var_c1: float
    var_h0: float
    var_h1: float
    w1_c0: float
    w1_c1: float


@dataclass(frozen=True)
class ArmSummary:
    n_c: int
    n_h: int
    mean_c: float
    mean_h: float
    var_c: float
    var_h: float
    sum_c: float
    sum_h: float
    succ_c: int
    succ_h: int


@dataclass(frozen=True)
class PosteriorSummary:
    mean: float
    var: float
    lambda_eff: float
    w_eff: float


@dataclass(frozen=True)
class MethodSpec:
    name: str
    family: str
    param: float | None = None


@dataclass(frozen=True)
class Params:
    seed: int
    outdir: Path
    alpha: float
    theta1: float
    gamma_grid: list[float]
    n_current: int
    n_historical_grid: list[int]
    m_type1: int
    m_power: int
    outcomes: list[str]
    scenarios: list[str]
    stages: list[str]
    rho_multipliers: list[float]
    lambda_grid: int
    fixed_lambdas: list[float]
    mc_moments: int
    alpha_pool: float
    power_prior_lambdas: list[float]
    commensurate_taus: list[float]
    map_lambdas: list[float]
    map_weights: list[float] | None
    rmap_epsilons: list[float]
    elastic_scales: list[float]
    uip_m_values: list[float]
    leap_prior_omega: float
    leap_nex_scale: float
    exnex_prior_ex: float
    mem_prior_inclusion: float
    mem_tau: float
    bhmoi_sharpness: float
    npb_temperature: float
    npb_sharpness: float
    beta_prior_alpha: float
    beta_prior_beta: float
    vague_alpha: float
    vague_beta: float
    normal_prior_mean: float
    normal_prior_var: float
    vague_normal_var: float


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_optional_float_list(text: str) -> list[float] | None:
    cleaned = text.strip()
    if cleaned == "":
        return None
    return parse_float_list(cleaned)


def parse_args() -> Params:
    parser = argparse.ArgumentParser(description="DRO-based borrowing simulation (continuous + binary).")
    parser.add_argument("--seed", type=int, default=202402)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/dro_simulation"))
    parser.add_argument("--alpha", type=float, default=0.025)
    parser.add_argument("--theta1", type=float, default=0.3)
    parser.add_argument("--gamma-grid", type=str, default="0,0.25,0.5,1,2")
    parser.add_argument("--n-current", type=int, default=200)
    parser.add_argument("--n-historical", type=int, default=500)
    parser.add_argument(
        "--n-historical-grid",
        type=str,
        default="",
        help="Optional comma-separated n_H grid. If empty, uses --n-historical.",
    )
    parser.add_argument("--m-type1", type=int, default=20000)
    parser.add_argument("--m-power", type=int, default=10000)
    parser.add_argument(
        "--outcomes",
        type=str,
        default="continuous,binary",
        help="Comma-separated list: continuous,binary",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default="S0,S2,S3",
        help="Comma-separated list of scenarios: S0,S1,S2,S3",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="oracle,data",
        help="Comma-separated list: oracle,data",
    )
    parser.add_argument("--rho-multiplier", type=float, default=1.5)
    parser.add_argument(
        "--rho-multipliers",
        type=str,
        default="",
        help="Optional comma-separated multipliers for data-driven rho (e.g., 1,1.5,2).",
    )
    parser.add_argument("--lambda-grid", type=int, default=401)
    parser.add_argument("--fixed-lambdas", type=str, default="0.25,0.5,0.75")
    parser.add_argument("--mc-moments", type=int, default=20000)
    parser.add_argument("--alpha-pool", type=float, default=0.1)
    parser.add_argument(
        "--power-prior-lambdas",
        type=str,
        default="0.5",
        help="Comma-separated power prior exponents.",
    )
    parser.add_argument(
        "--commensurate-taus",
        type=str,
        default="1.0",
        help="Comma-separated commensurate precision parameters.",
    )
    parser.add_argument(
        "--map-lambdas",
        type=str,
        default="0.25,1.0",
        help="Comma-separated MAP mixture component borrowing exponents.",
    )
    parser.add_argument(
        "--map-weights",
        type=str,
        default="",
        help="Optional MAP mixture weights (comma-separated, same length as --map-lambdas).",
    )
    parser.add_argument(
        "--rmap-epsilons",
        type=str,
        default="0.2",
        help="Comma-separated robust MAP vague-component mixture weights.",
    )
    parser.add_argument(
        "--elastic-scales",
        type=str,
        default="1.0",
        help="Comma-separated scale parameters for elastic prior borrowing.",
    )
    parser.add_argument(
        "--uip-m-values",
        type=str,
        default="100",
        help="Comma-separated unit-information levels m (effective pseudo sample sizes).",
    )
    parser.add_argument(
        "--leap-prior-omega",
        type=float,
        default=0.5,
        help="Prior exchangeability probability for LEAP-like borrowing.",
    )
    parser.add_argument(
        "--leap-nex-scale",
        type=float,
        default=9.0,
        help="Variance inflation factor for LEAP non-exchangeable component (>0).",
    )
    parser.add_argument(
        "--exnex-prior-ex",
        type=float,
        default=0.7,
        help="Prior exchangeable mixture weight for EXNEX.",
    )
    parser.add_argument(
        "--mem-prior-inclusion",
        type=float,
        default=0.5,
        help="Prior source-inclusion probability for MEM-like borrowing.",
    )
    parser.add_argument(
        "--mem-tau",
        type=float,
        default=1.0,
        help="Commensurability scale for MEM-like exchangeable component (>0).",
    )
    parser.add_argument(
        "--bhmoi-sharpness",
        type=float,
        default=1.0,
        help="Sharpness exponent for BHMOI overlap index (>=0).",
    )
    parser.add_argument(
        "--npb-temperature",
        type=float,
        default=1.0,
        help="Temperature for nonparametric-Bayes similarity mapping (>0).",
    )
    parser.add_argument(
        "--npb-sharpness",
        type=float,
        default=1.0,
        help="Sharpness exponent for nonparametric-Bayes borrowing (>=0).",
    )
    parser.add_argument("--beta-prior-alpha", type=float, default=1.0)
    parser.add_argument("--beta-prior-beta", type=float, default=1.0)
    parser.add_argument("--vague-alpha", type=float, default=1.0)
    parser.add_argument("--vague-beta", type=float, default=1.0)
    parser.add_argument("--normal-prior-mean", type=float, default=0.0)
    parser.add_argument("--normal-prior-var", type=float, default=1e8)
    parser.add_argument("--vague-normal-var", type=float, default=1e8)
    args = parser.parse_args()
    n_h_grid = parse_int_list(args.n_historical_grid) if args.n_historical_grid.strip() else [args.n_historical]
    rho_mults = parse_float_list(args.rho_multipliers) if args.rho_multipliers.strip() else [args.rho_multiplier]

    return Params(
        seed=args.seed,
        outdir=args.outdir,
        alpha=args.alpha,
        theta1=args.theta1,
        gamma_grid=parse_float_list(args.gamma_grid),
        n_current=args.n_current,
        n_historical_grid=n_h_grid,
        m_type1=args.m_type1,
        m_power=args.m_power,
        outcomes=[x.strip() for x in args.outcomes.split(",") if x.strip()],
        scenarios=[x.strip() for x in args.scenarios.split(",") if x.strip()],
        stages=[x.strip() for x in args.stages.split(",") if x.strip()],
        rho_multipliers=rho_mults,
        lambda_grid=args.lambda_grid,
        fixed_lambdas=parse_float_list(args.fixed_lambdas),
        mc_moments=args.mc_moments,
        alpha_pool=args.alpha_pool,
        power_prior_lambdas=parse_float_list(args.power_prior_lambdas),
        commensurate_taus=parse_float_list(args.commensurate_taus),
        map_lambdas=parse_float_list(args.map_lambdas),
        map_weights=parse_optional_float_list(args.map_weights),
        rmap_epsilons=parse_float_list(args.rmap_epsilons),
        elastic_scales=parse_float_list(args.elastic_scales),
        uip_m_values=parse_float_list(args.uip_m_values),
        leap_prior_omega=args.leap_prior_omega,
        leap_nex_scale=args.leap_nex_scale,
        exnex_prior_ex=args.exnex_prior_ex,
        mem_prior_inclusion=args.mem_prior_inclusion,
        mem_tau=args.mem_tau,
        bhmoi_sharpness=args.bhmoi_sharpness,
        npb_temperature=args.npb_temperature,
        npb_sharpness=args.npb_sharpness,
        beta_prior_alpha=args.beta_prior_alpha,
        beta_prior_beta=args.beta_prior_beta,
        vague_alpha=args.vague_alpha,
        vague_beta=args.vague_beta,
        normal_prior_mean=args.normal_prior_mean,
        normal_prior_var=args.normal_prior_var,
        vague_normal_var=args.vague_normal_var,
    )


def validate_params(p: Params) -> None:
    if not (0.0 < p.alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")
    if p.theta1 <= 0.0:
        raise ValueError("theta1 must be > 0")
    if p.n_current <= 0:
        raise ValueError("Sample sizes must be positive")
    if any(n_h < 0 for n_h in p.n_historical_grid):
        raise ValueError("n-historical-grid values must be >= 0")
    if p.m_type1 <= 0 or p.m_power <= 0:
        raise ValueError("Simulation sizes must be positive")
    if any(c <= 0.0 for c in p.rho_multipliers):
        raise ValueError("rho multipliers must be > 0")
    if p.lambda_grid < 11:
        raise ValueError("lambda-grid must be >= 11")
    if any(lam < 0.0 or lam > 1.0 for lam in p.fixed_lambdas):
        raise ValueError("fixed-lambdas must be in [0,1]")
    if any(lam < 0.0 or lam > 1.0 for lam in p.power_prior_lambdas):
        raise ValueError("power-prior-lambdas must be in [0,1]")
    if any(tau <= 0.0 for tau in p.commensurate_taus):
        raise ValueError("commensurate-taus must be > 0")
    if len(p.map_lambdas) == 0:
        raise ValueError("map-lambdas must contain at least one value")
    if any(lam <= 0.0 for lam in p.map_lambdas):
        raise ValueError("map-lambdas must be > 0")
    if p.map_weights is not None:
        if len(p.map_weights) != len(p.map_lambdas):
            raise ValueError("map-weights length must match map-lambdas length")
        if any(w <= 0.0 for w in p.map_weights):
            raise ValueError("map-weights must be positive")
    if any(eps < 0.0 or eps >= 1.0 for eps in p.rmap_epsilons):
        raise ValueError("rmap-epsilons must be in [0,1)")
    if any(scale <= 0.0 for scale in p.elastic_scales):
        raise ValueError("elastic-scales must be > 0")
    if any(m <= 0.0 for m in p.uip_m_values):
        raise ValueError("uip-m-values must be > 0")
    if not (0.0 < p.leap_prior_omega < 1.0):
        raise ValueError("leap-prior-omega must be in (0,1)")
    if p.leap_nex_scale <= 0.0:
        raise ValueError("leap-nex-scale must be > 0")
    if not (0.0 < p.exnex_prior_ex < 1.0):
        raise ValueError("exnex-prior-ex must be in (0,1)")
    if not (0.0 < p.mem_prior_inclusion < 1.0):
        raise ValueError("mem-prior-inclusion must be in (0,1)")
    if p.mem_tau <= 0.0:
        raise ValueError("mem-tau must be > 0")
    if p.bhmoi_sharpness < 0.0:
        raise ValueError("bhmoi-sharpness must be >= 0")
    if p.npb_temperature <= 0.0:
        raise ValueError("npb-temperature must be > 0")
    if p.npb_sharpness < 0.0:
        raise ValueError("npb-sharpness must be >= 0")
    if p.beta_prior_alpha <= 0.0 or p.beta_prior_beta <= 0.0:
        raise ValueError("beta prior hyperparameters must be > 0")
    if p.vague_alpha <= 0.0 or p.vague_beta <= 0.0:
        raise ValueError("vague beta prior hyperparameters must be > 0")
    if p.normal_prior_var <= 0.0 or p.vague_normal_var <= 0.0:
        raise ValueError("normal prior variances must be > 0")


def scenario_params(scenario: Scenario, p: DGPParams) -> tuple[np.ndarray, float, float, str]:
    eta = p.eta_effect if scenario.effect_mod else np.zeros_like(p.eta_effect)
    mean_shift = np.ones(p.p) if scenario.cov_shift else np.zeros(p.p)
    return eta, mean_shift, (1.0 if scenario.control_drift else 0.0), scenario.hist_arms


def simulate_trial(
    rng: np.random.Generator,
    n: int,
    mean_shift: np.ndarray,
    beta0: float,
    beta: np.ndarray,
    eta: np.ndarray,
    tau: float,
    sigma: float,
    pi: float,
    u0: float,
    u1: float,
    outcome: str,
    hist_arms: str,
) -> tuple[np.ndarray, np.ndarray]:
    x = rng.normal(loc=mean_shift, scale=1.0, size=(n, beta.size))
    if hist_arms == "both":
        n_t = int(round(n * pi))
        n_t = min(max(n_t, 0), n)
        a = np.concatenate(
            [
                np.zeros(n - n_t, dtype=int),
                np.ones(n_t, dtype=int),
            ]
        )
        rng.shuffle(a)
    else:
        a = np.zeros(n, dtype=int)
    lin = beta0 + x @ beta + tau * a + (x * a[:, None]) @ eta + u0 + (u1 - u0) * a
    if outcome == "continuous":
        y = lin + rng.normal(0.0, sigma, size=n)
    else:
        prob = expit(lin)
        y = rng.binomial(1, prob, size=n)
    return a, y


def normalize_weights(weights: list[float] | None, n: int) -> np.ndarray:
    if weights is None:
        return np.full(n, 1.0 / n, dtype=float)
    arr = np.array(weights, dtype=float)
    total = float(np.sum(arr))
    if total <= 0.0:
        raise ValueError("Mixture weights must sum to a positive value.")
    return arr / total


def safe_log(x: float) -> float:
    if x <= 0.0:
        return -np.inf
    return float(np.log(x))


def clip_unit(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def logit(x: float) -> float:
    x = float(np.clip(x, 1e-8, 1.0 - 1e-8))
    return float(np.log(x / (1.0 - x)))


def logistic(x: float) -> float:
    return float(expit(x))


def beta_log_norm(a: float, b: float) -> float:
    return float(betaln(a, b))


def beta_binomial_log_marginal(success: int, failure: int, a: float, b: float) -> float:
    return beta_log_norm(a + success, b + failure) - beta_log_norm(a, b)


def normal_logpdf(x: float, mean: float, var: float) -> float:
    var = max(var, 1e-12)
    return float(norm.logpdf(x, loc=mean, scale=math.sqrt(var)))


def estimate_sigma2_continuous(arm: ArmSummary) -> float:
    if arm.n_c > 1 and arm.n_h > 1:
        num = (arm.n_c - 1) * arm.var_c + (arm.n_h - 1) * arm.var_h
        den = arm.n_c + arm.n_h - 2
        return max(num / max(den, 1), 1e-8)
    if arm.n_c > 1:
        return max(arm.var_c, 1e-8)
    if arm.n_h > 1:
        return max(arm.var_h, 1e-8)
    return 1.0


def build_arm_summary(y_c_arm: np.ndarray, y_h_arm: np.ndarray, outcome: str) -> ArmSummary:
    n_c = int(y_c_arm.size)
    n_h = int(y_h_arm.size)
    sum_c = float(np.sum(y_c_arm)) if n_c > 0 else 0.0
    sum_h = float(np.sum(y_h_arm)) if n_h > 0 else 0.0
    mean_c = float(sum_c / n_c) if n_c > 0 else 0.0
    mean_h = float(sum_h / n_h) if n_h > 0 else 0.0
    var_c = float(np.var(y_c_arm, ddof=1)) if n_c > 1 else 0.0
    var_h = float(np.var(y_h_arm, ddof=1)) if n_h > 1 else 0.0
    if outcome == "binary":
        var_c = clamp_prob(mean_c) * (1.0 - clamp_prob(mean_c)) if n_c > 0 else 0.0
        var_h = clamp_prob(mean_h) * (1.0 - clamp_prob(mean_h)) if n_h > 0 else 0.0
    succ_c = int(round(sum_c)) if outcome == "binary" else 0
    succ_h = int(round(sum_h)) if outcome == "binary" else 0
    return ArmSummary(
        n_c=n_c,
        n_h=n_h,
        mean_c=mean_c,
        mean_h=mean_h,
        var_c=var_c,
        var_h=var_h,
        sum_c=sum_c,
        sum_h=sum_h,
        succ_c=succ_c,
        succ_h=succ_h,
    )


def pooled_estimate(arm: ArmSummary, lam: float, outcome: str) -> PosteriorSummary:
    lam = float(np.clip(lam, 0.0, 1.0))
    w = w_from_lambda(lam, arm.n_c, arm.n_h)
    mean = (1.0 - w) * arm.mean_c + w * arm.mean_h
    if outcome == "binary":
        var = (1.0 - w) ** 2 * var_mean_binary(arm.mean_c, arm.n_c)
        if arm.n_h > 0:
            var += w**2 * var_mean_binary(arm.mean_h, arm.n_h)
    else:
        var = (1.0 - w) ** 2 * var_mean_cont(arm.var_c, arm.n_c)
        if arm.n_h > 0:
            var += w**2 * var_mean_cont(arm.var_h, arm.n_h)
    return PosteriorSummary(mean=float(mean), var=max(float(var), 1e-12), lambda_eff=lam, w_eff=w)


def posterior_power_prior(arm: ArmSummary, outcome: str, lam: float, p: Params) -> PosteriorSummary:
    lam = float(np.clip(lam, 0.0, 1.0))
    if arm.n_h <= 0:
        return pooled_estimate(arm, 0.0, outcome)

    if outcome == "binary":
        s_c = arm.succ_c
        f_c = arm.n_c - s_c
        s_h = arm.succ_h
        f_h = arm.n_h - s_h
        a_post = p.beta_prior_alpha + s_c + lam * s_h
        b_post = p.beta_prior_beta + f_c + lam * f_h
        total = a_post + b_post
        mean = a_post / total
        var = (a_post * b_post) / (total**2 * (total + 1.0))
        return PosteriorSummary(
            mean=float(mean),
            var=max(float(var), 1e-12),
            lambda_eff=lam,
            w_eff=w_from_lambda(lam, arm.n_c, arm.n_h),
        )

    sigma2 = estimate_sigma2_continuous(arm)
    prec0 = 1.0 / p.normal_prior_var
    post_prec = prec0 + arm.n_c / sigma2 + lam * arm.n_h / sigma2
    post_mean = (
        prec0 * p.normal_prior_mean + arm.n_c * arm.mean_c / sigma2 + lam * arm.n_h * arm.mean_h / sigma2
    ) / post_prec
    post_var = 1.0 / post_prec
    return PosteriorSummary(
        mean=float(post_mean),
        var=max(float(post_var), 1e-12),
        lambda_eff=lam,
        w_eff=w_from_lambda(lam, arm.n_c, arm.n_h),
    )


def posterior_commensurate(arm: ArmSummary, outcome: str, tau: float) -> PosteriorSummary:
    tau = max(float(tau), 1e-12)
    if arm.n_h <= 0:
        return pooled_estimate(arm, 0.0, outcome)

    if outcome == "continuous":
        sigma2 = estimate_sigma2_continuous(arm)
        prior_var = sigma2 / arm.n_h + 1.0 / tau
        prior_prec = 1.0 / prior_var
        like_prec = arm.n_c / sigma2
        post_prec = like_prec + prior_prec
        post_mean = (like_prec * arm.mean_c + prior_prec * arm.mean_h) / post_prec
        post_var = 1.0 / post_prec
        lam_eff = 1.0 / (1.0 + arm.n_h / (sigma2 * tau))
    else:
        like_var = var_mean_binary(arm.mean_c, arm.n_c)
        prior_var = var_mean_binary(arm.mean_h, arm.n_h) + 1.0 / tau
        like_prec = 1.0 / max(like_var, 1e-12)
        prior_prec = 1.0 / max(prior_var, 1e-12)
        post_prec = like_prec + prior_prec
        post_mean = (like_prec * arm.mean_c + prior_prec * arm.mean_h) / post_prec
        post_mean = clamp_prob(float(post_mean))
        post_var = 1.0 / post_prec
        sigma2_h = clamp_prob(arm.mean_h) * (1.0 - clamp_prob(arm.mean_h))
        lam_eff = 1.0 / (1.0 + arm.n_h / (max(sigma2_h, 1e-8) * tau))
        lam_eff = float(np.clip(lam_eff, 0.0, 1.0))

    return PosteriorSummary(
        mean=float(post_mean),
        var=max(float(post_var), 1e-12),
        lambda_eff=float(lam_eff),
        w_eff=w_from_lambda(float(lam_eff), arm.n_c, arm.n_h),
    )


def posterior_map(arm: ArmSummary, outcome: str, p: Params) -> PosteriorSummary:
    if arm.n_h <= 0:
        return pooled_estimate(arm, 0.0, outcome)

    lambdas = np.array(p.map_lambdas, dtype=float)
    base_weights = normalize_weights(p.map_weights, lambdas.size)

    if outcome == "binary":
        s_c = arm.succ_c
        f_c = arm.n_c - s_c
        s_h = arm.succ_h
        f_h = arm.n_h - s_h
        mean_components: list[float] = []
        var_components: list[float] = []
        log_post_weights: list[float] = []
        w_components: list[float] = []
        for idx, lam in enumerate(lambdas):
            a_prior = p.beta_prior_alpha + lam * s_h
            b_prior = p.beta_prior_beta + lam * f_h
            a_post = a_prior + s_c
            b_post = b_prior + f_c
            total = a_post + b_post
            mean_k = a_post / total
            var_k = (a_post * b_post) / (total**2 * (total + 1.0))
            log_ml = beta_log_norm(a_post, b_post) - beta_log_norm(a_prior, b_prior)
            log_post_weights.append(float(safe_log(float(base_weights[idx])) + log_ml))
            mean_components.append(float(mean_k))
            var_components.append(float(var_k))
            w_components.append(w_from_lambda(float(lam), arm.n_c, arm.n_h))
    else:
        sigma2 = estimate_sigma2_continuous(arm)
        like_var = sigma2 / arm.n_c
        mean_components = []
        var_components = []
        log_post_weights = []
        w_components = []
        for idx, lam in enumerate(lambdas):
            prior_mean = arm.mean_h
            prior_var = sigma2 / (lam * arm.n_h)
            prior_prec = 1.0 / prior_var
            like_prec = 1.0 / like_var
            post_var = 1.0 / (like_prec + prior_prec)
            post_mean = post_var * (like_prec * arm.mean_c + prior_prec * prior_mean)
            log_ml = normal_logpdf(arm.mean_c, prior_mean, like_var + prior_var)
            log_post_weights.append(float(safe_log(float(base_weights[idx])) + log_ml))
            mean_components.append(float(post_mean))
            var_components.append(float(post_var))
            w_components.append(w_from_lambda(float(lam), arm.n_c, arm.n_h))

    log_weights = np.array(log_post_weights, dtype=float)
    post_weights = np.exp(log_weights - float(logsumexp(log_weights)))
    means = np.array(mean_components, dtype=float)
    variances = np.array(var_components, dtype=float)
    w_vals = np.array(w_components, dtype=float)
    lam_vals = lambdas
    post_mean = float(np.sum(post_weights * means))
    post_var = float(np.sum(post_weights * (variances + means**2)) - post_mean**2)
    lambda_eff = float(np.sum(post_weights * lam_vals))
    w_eff = float(np.sum(post_weights * w_vals))
    return PosteriorSummary(
        mean=post_mean,
        var=max(post_var, 1e-12),
        lambda_eff=lambda_eff,
        w_eff=w_eff,
    )


def posterior_rmap(arm: ArmSummary, outcome: str, p: Params, epsilon: float) -> PosteriorSummary:
    epsilon = float(np.clip(epsilon, 0.0, 0.999999))
    if arm.n_h <= 0:
        return pooled_estimate(arm, 0.0, outcome)

    lambdas = np.array(p.map_lambdas, dtype=float)
    base_weights = normalize_weights(p.map_weights, lambdas.size)
    map_weights = (1.0 - epsilon) * base_weights

    if outcome == "binary":
        s_c = arm.succ_c
        f_c = arm.n_c - s_c
        s_h = arm.succ_h
        f_h = arm.n_h - s_h

        mean_components: list[float] = []
        var_components: list[float] = []
        log_post_weights: list[float] = []
        lambda_components: list[float] = []
        w_components: list[float] = []

        for idx, lam in enumerate(lambdas):
            a_prior = p.beta_prior_alpha + lam * s_h
            b_prior = p.beta_prior_beta + lam * f_h
            a_post = a_prior + s_c
            b_post = b_prior + f_c
            total = a_post + b_post
            mean_k = a_post / total
            var_k = (a_post * b_post) / (total**2 * (total + 1.0))
            log_ml = beta_log_norm(a_post, b_post) - beta_log_norm(a_prior, b_prior)
            log_post_weights.append(float(safe_log(float(map_weights[idx])) + log_ml))
            mean_components.append(float(mean_k))
            var_components.append(float(var_k))
            lambda_components.append(float(lam))
            w_components.append(w_from_lambda(float(lam), arm.n_c, arm.n_h))

        a_v = p.vague_alpha
        b_v = p.vague_beta
        a_post_v = a_v + s_c
        b_post_v = b_v + f_c
        total_v = a_post_v + b_post_v
        mean_v = a_post_v / total_v
        var_v = (a_post_v * b_post_v) / (total_v**2 * (total_v + 1.0))
        log_ml_v = beta_log_norm(a_post_v, b_post_v) - beta_log_norm(a_v, b_v)
        log_post_weights.append(float(safe_log(epsilon) + log_ml_v))
        mean_components.append(float(mean_v))
        var_components.append(float(var_v))
        lambda_components.append(0.0)
        w_components.append(0.0)
    else:
        sigma2 = estimate_sigma2_continuous(arm)
        like_var = sigma2 / arm.n_c

        mean_components = []
        var_components = []
        log_post_weights = []
        lambda_components = []
        w_components = []

        for idx, lam in enumerate(lambdas):
            prior_mean = arm.mean_h
            prior_var = sigma2 / (lam * arm.n_h)
            prior_prec = 1.0 / prior_var
            like_prec = 1.0 / like_var
            post_var = 1.0 / (like_prec + prior_prec)
            post_mean = post_var * (like_prec * arm.mean_c + prior_prec * prior_mean)
            log_ml = normal_logpdf(arm.mean_c, prior_mean, like_var + prior_var)
            log_post_weights.append(float(safe_log(float(map_weights[idx])) + log_ml))
            mean_components.append(float(post_mean))
            var_components.append(float(post_var))
            lambda_components.append(float(lam))
            w_components.append(w_from_lambda(float(lam), arm.n_c, arm.n_h))

        prior_mean_v = p.normal_prior_mean
        prior_var_v = p.vague_normal_var
        prior_prec_v = 1.0 / prior_var_v
        like_prec = 1.0 / like_var
        post_var_v = 1.0 / (like_prec + prior_prec_v)
        post_mean_v = post_var_v * (like_prec * arm.mean_c + prior_prec_v * prior_mean_v)
        log_ml_v = normal_logpdf(arm.mean_c, prior_mean_v, like_var + prior_var_v)
        log_post_weights.append(float(safe_log(epsilon) + log_ml_v))
        mean_components.append(float(post_mean_v))
        var_components.append(float(post_var_v))
        lambda_components.append(0.0)
        w_components.append(0.0)

    log_weights = np.array(log_post_weights, dtype=float)
    post_weights = np.exp(log_weights - float(logsumexp(log_weights)))
    means = np.array(mean_components, dtype=float)
    variances = np.array(var_components, dtype=float)
    lambda_vals = np.array(lambda_components, dtype=float)
    w_vals = np.array(w_components, dtype=float)
    post_mean = float(np.sum(post_weights * means))
    post_var = float(np.sum(post_weights * (variances + means**2)) - post_mean**2)
    lambda_eff = float(np.sum(post_weights * lambda_vals))
    w_eff = float(np.sum(post_weights * w_vals))
    return PosteriorSummary(
        mean=post_mean,
        var=max(post_var, 1e-12),
        lambda_eff=lambda_eff,
        w_eff=w_eff,
    )


def posterior_elastic(arm: ArmSummary, outcome: str, scale: float) -> PosteriorSummary:
    if arm.n_h <= 0:
        return pooled_estimate(arm, 0.0, outcome)

    if outcome == "binary":
        se2 = var_mean_binary(arm.mean_c, arm.n_c) + var_mean_binary(arm.mean_h, arm.n_h)
    else:
        se2 = var_mean_cont(arm.var_c, arm.n_c) + var_mean_cont(arm.var_h, arm.n_h)
    if se2 <= 0.0:
        return pooled_estimate(arm, 0.0, outcome)

    z = (arm.mean_c - arm.mean_h) / np.sqrt(se2)
    # Larger discrepancy implies lower borrowing.
    g = float(np.exp(-0.5 * scale * (float(z) ** 2)))
    lam_eff = float(np.clip(g, 0.0, 1.0))
    return pooled_estimate(arm, lam_eff, outcome)


def posterior_uip(arm: ArmSummary, outcome: str, m_units: float) -> PosteriorSummary:
    if arm.n_h <= 0:
        return pooled_estimate(arm, 0.0, outcome)
    lam_eff = lambda_from_uip(m_units, arm.n_h)
    return pooled_estimate(arm, lam_eff, outcome)


def posterior_leap(arm: ArmSummary, outcome: str, p: Params) -> PosteriorSummary:
    if arm.n_h <= 0:
        return pooled_estimate(arm, 0.0, outcome)

    if outcome == "binary":
        se2 = var_mean_binary(arm.mean_c, arm.n_c) + var_mean_binary(arm.mean_h, arm.n_h)
    else:
        se2 = var_mean_cont(arm.var_c, arm.n_c) + var_mean_cont(arm.var_h, arm.n_h)
    if se2 <= 0.0:
        return pooled_estimate(arm, 0.0, outcome)

    diff = arm.mean_c - arm.mean_h
    var_ex = max(se2, 1e-12)
    var_nex = max((1.0 + p.leap_nex_scale) * se2, var_ex + 1e-12)
    log_bf_ex = normal_logpdf(diff, 0.0, var_ex) - normal_logpdf(diff, 0.0, var_nex)
    post_logit = logit(p.leap_prior_omega) + log_bf_ex
    lam_eff = clip_unit(logistic(post_logit))
    return pooled_estimate(arm, lam_eff, outcome)


def posterior_exnex(arm: ArmSummary, outcome: str, p: Params) -> PosteriorSummary:
    if arm.n_h <= 0:
        return pooled_estimate(arm, 0.0, outcome)

    omega_ex = float(np.clip(p.exnex_prior_ex, 1e-8, 1.0 - 1e-8))
    post_ex = posterior_power_prior(arm, outcome, lam=1.0, p=p)

    if outcome == "binary":
        s_c = arm.succ_c
        f_c = arm.n_c - s_c
        s_h = arm.succ_h
        f_h = arm.n_h - s_h

        a_ex_prior = p.beta_prior_alpha + s_h
        b_ex_prior = p.beta_prior_beta + f_h
        log_ml_ex = beta_binomial_log_marginal(s_c, f_c, a_ex_prior, b_ex_prior)

        a_nex = p.vague_alpha
        b_nex = p.vague_beta
        a_post_nex = a_nex + s_c
        b_post_nex = b_nex + f_c
        total_nex = a_post_nex + b_post_nex
        mean_nex = a_post_nex / total_nex
        var_nex = (a_post_nex * b_post_nex) / (total_nex**2 * (total_nex + 1.0))
        post_nex = PosteriorSummary(
            mean=float(mean_nex),
            var=max(float(var_nex), 1e-12),
            lambda_eff=0.0,
            w_eff=0.0,
        )
        log_ml_nex = beta_binomial_log_marginal(s_c, f_c, a_nex, b_nex)
    else:
        sigma2 = estimate_sigma2_continuous(arm)
        like_var = max(sigma2 / max(arm.n_c, 1), 1e-12)

        prior_mean_ex = arm.mean_h
        prior_var_ex = max(sigma2 / max(arm.n_h, 1), 1e-12)
        log_ml_ex = normal_logpdf(arm.mean_c, prior_mean_ex, like_var + prior_var_ex)

        prior_mean_nex = p.normal_prior_mean
        prior_var_nex = p.vague_normal_var
        prior_prec_nex = 1.0 / max(prior_var_nex, 1e-12)
        like_prec = 1.0 / like_var
        post_var_nex = 1.0 / (like_prec + prior_prec_nex)
        post_mean_nex = post_var_nex * (like_prec * arm.mean_c + prior_prec_nex * prior_mean_nex)
        post_nex = PosteriorSummary(
            mean=float(post_mean_nex),
            var=max(float(post_var_nex), 1e-12),
            lambda_eff=0.0,
            w_eff=0.0,
        )
        log_ml_nex = normal_logpdf(arm.mean_c, prior_mean_nex, like_var + prior_var_nex)

    log_w_ex = safe_log(omega_ex) + log_ml_ex
    log_w_nex = safe_log(1.0 - omega_ex) + log_ml_nex
    log_ws = np.array([log_w_ex, log_w_nex], dtype=float)
    if not np.isfinite(log_ws).any():
        weights = np.array([0.5, 0.5], dtype=float)
    else:
        weights = np.exp(log_ws - float(logsumexp(log_ws)))

    mean = float(weights[0] * post_ex.mean + weights[1] * post_nex.mean)
    var = float(weights[0] * (post_ex.var + post_ex.mean**2) + weights[1] * (post_nex.var + post_nex.mean**2) - mean**2)
    lambda_eff = float(weights[0] * post_ex.lambda_eff + weights[1] * post_nex.lambda_eff)
    w_eff = float(weights[0] * post_ex.w_eff + weights[1] * post_nex.w_eff)
    return PosteriorSummary(mean=mean, var=max(var, 1e-12), lambda_eff=lambda_eff, w_eff=w_eff)


def posterior_mem(arm: ArmSummary, outcome: str, p: Params) -> PosteriorSummary:
    if arm.n_h <= 0:
        return pooled_estimate(arm, 0.0, outcome)

    pi_in = float(np.clip(p.mem_prior_inclusion, 1e-8, 1.0 - 1e-8))
    post_out = pooled_estimate(arm, 0.0, outcome)

    if outcome == "binary":
        s_c = arm.succ_c
        f_c = arm.n_c - s_c
        s_h = arm.succ_h
        f_h = arm.n_h - s_h
        lam_in = float(np.clip(p.mem_tau / (1.0 + p.mem_tau), 0.0, 1.0))
        post_in = posterior_power_prior(arm, outcome, lam=lam_in, p=p)

        a_in = p.beta_prior_alpha + lam_in * s_h
        b_in = p.beta_prior_beta + lam_in * f_h
        log_ml_in = beta_binomial_log_marginal(s_c, f_c, a_in, b_in)
        log_ml_out = beta_binomial_log_marginal(s_c, f_c, p.vague_alpha, p.vague_beta)
    else:
        post_in = posterior_commensurate(arm, outcome, tau=p.mem_tau)
        sigma2 = estimate_sigma2_continuous(arm)
        like_var = max(sigma2 / max(arm.n_c, 1), 1e-12)
        prior_var_in = max(sigma2 / max(arm.n_h, 1) + 1.0 / p.mem_tau, 1e-12)
        log_ml_in = normal_logpdf(arm.mean_c, arm.mean_h, like_var + prior_var_in)
        log_ml_out = normal_logpdf(arm.mean_c, p.normal_prior_mean, like_var + p.vague_normal_var)

    log_w_in = safe_log(pi_in) + log_ml_in
    log_w_out = safe_log(1.0 - pi_in) + log_ml_out
    log_ws = np.array([log_w_in, log_w_out], dtype=float)
    if not np.isfinite(log_ws).any():
        weights = np.array([0.5, 0.5], dtype=float)
    else:
        weights = np.exp(log_ws - float(logsumexp(log_ws)))

    mean = float(weights[0] * post_in.mean + weights[1] * post_out.mean)
    var = float(weights[0] * (post_in.var + post_in.mean**2) + weights[1] * (post_out.var + post_out.mean**2) - mean**2)
    lambda_eff = float(weights[0] * post_in.lambda_eff + weights[1] * post_out.lambda_eff)
    w_eff = float(weights[0] * post_in.w_eff + weights[1] * post_out.w_eff)
    return PosteriorSummary(mean=mean, var=max(var, 1e-12), lambda_eff=lambda_eff, w_eff=w_eff)


def posterior_bhmoi(arm: ArmSummary, outcome: str, p: Params) -> PosteriorSummary:
    if arm.n_h <= 0:
        return pooled_estimate(arm, 0.0, outcome)

    if outcome == "continuous":
        sd_c = math.sqrt(max(arm.var_c, 1e-12))
        sd_h = math.sqrt(max(arm.var_h, 1e-12))
        denom = max(sd_c**2 + sd_h**2, 1e-12)
        oci = math.sqrt(max(2.0 * sd_c * sd_h / denom, 0.0))
        oci *= math.exp(-((arm.mean_c - arm.mean_h) ** 2) / (4.0 * denom))
        oci = clip_unit(oci)
        obi = math.exp(-abs(math.log((sd_c + 1e-8) / (sd_h + 1e-8))))
    else:
        p_c = clamp_prob(arm.mean_c)
        p_h = clamp_prob(arm.mean_h)
        oci = math.sqrt(p_c * p_h) + math.sqrt((1.0 - p_c) * (1.0 - p_h))
        oci = clip_unit(oci)
        v_c = p_c * (1.0 - p_c)
        v_h = p_h * (1.0 - p_h)
        obi = 2.0 * min(v_c, v_h) / max(v_c + v_h, 1e-8)
        obi = clip_unit(obi)

    overlap = clip_unit(math.sqrt(max(oci * obi, 0.0)))
    lam_eff = clip_unit(overlap**p.bhmoi_sharpness)
    return pooled_estimate(arm, lam_eff, outcome)


def posterior_nonpara_bayes(arm: ArmSummary, outcome: str, p: Params) -> PosteriorSummary:
    if arm.n_h <= 0:
        return pooled_estimate(arm, 0.0, outcome)

    if outcome == "continuous":
        sd_c = math.sqrt(max(arm.var_c, 1e-12))
        sd_h = math.sqrt(max(arm.var_h, 1e-12))
        w1_proxy = abs(arm.mean_c - arm.mean_h) + math.sqrt(2.0 / math.pi) * abs(sd_c - sd_h)
        scale = max(sd_c + sd_h, 1e-8)
        sbi = math.exp(-p.npb_temperature * w1_proxy / scale)
    else:
        p_c = clamp_prob(arm.mean_c)
        p_h = clamp_prob(arm.mean_h)
        bhatta = math.sqrt(p_c * p_h) + math.sqrt((1.0 - p_c) * (1.0 - p_h))
        var_gap = abs(p_c * (1.0 - p_c) - p_h * (1.0 - p_h))
        sbi = bhatta * math.exp(-p.npb_temperature * var_gap)
    sbi = clip_unit(sbi)
    lam_eff = clip_unit(sbi**p.npb_sharpness)
    return pooled_estimate(arm, lam_eff, outcome)


def estimate_moments_continuous(
    scenario: Scenario,
    p: DGPParams,
    gamma: float,
    tau: float,
) -> Moments:
    eta, mean_shift_base, drift_on, _ = scenario_params(scenario, p)
    mean_c = np.zeros(p.p)
    mean_h = gamma * mean_shift_base

    u_h0 = gamma * drift_on
    u_h1 = 0.0

    beta = p.beta
    beta0 = p.beta0_cont

    mu_c0 = beta0 + beta @ mean_c
    mu_c1 = beta0 + beta @ mean_c + tau + eta @ mean_c
    mu_h0 = beta0 + beta @ mean_h + u_h0
    mu_h1 = beta0 + beta @ mean_h + tau + eta @ mean_h + u_h1

    var_c0 = float(np.dot(beta, beta) + p.sigma**2)
    var_c1 = float(np.dot(beta + eta, beta + eta) + p.sigma**2)
    var_h0 = var_c0
    var_h1 = var_c1

    w1_c0 = 0.0
    w1_c1 = 0.0
    return Moments(mu_c0, mu_c1, mu_h0, mu_h1, var_c0, var_c1, var_h0, var_h1, w1_c0, w1_c1)


def estimate_moments_binary(
    scenario: Scenario,
    p: DGPParams,
    gamma: float,
    tau: float,
    mc_size: int,
    seed: int,
) -> Moments:
    eta, mean_shift_base, drift_on, _ = scenario_params(scenario, p)
    rng = np.random.default_rng(seed)

    def sample_group(mean_shift: np.ndarray, a_val: int, u0: float, u1: float) -> tuple[float, float]:
        x = rng.normal(loc=mean_shift, scale=1.0, size=(mc_size, p.p))
        a = np.full(mc_size, a_val)
        lin = p.beta0_bin + x @ p.beta + tau * a + (x * a[:, None]) @ eta + u0 + (u1 - u0) * a
        prob = expit(lin)
        y = rng.binomial(1, prob)
        mean = float(np.mean(y))
        var = float(np.var(y, ddof=1)) if mc_size > 1 else 0.0
        return mean, var

    mean_c = np.zeros(p.p)
    mean_h = gamma * mean_shift_base

    u_h0 = gamma * drift_on
    u_h1 = 0.0

    mu_c0, var_c0 = sample_group(mean_c, 0, 0.0, 0.0)
    mu_c1, var_c1 = sample_group(mean_c, 1, 0.0, 0.0)
    mu_h0, var_h0 = sample_group(mean_h, 0, u_h0, u_h1)
    mu_h1, var_h1 = sample_group(mean_h, 1, u_h0, u_h1)

    w1_c0 = abs(mu_h0 - mu_c0)
    w1_c1 = abs(mu_h1 - mu_c1)
    return Moments(mu_c0, mu_c1, mu_h0, mu_h1, var_c0, var_c1, var_h0, var_h1, w1_c0, w1_c1)


def estimate_w1_continuous(
    scenario: Scenario,
    p: DGPParams,
    gamma: float,
    tau: float,
    mc_size: int,
    seed: int,
) -> tuple[float, float]:
    eta, mean_shift_base, drift_on, _ = scenario_params(scenario, p)
    rng = np.random.default_rng(seed)

    def sample_y(mean_shift: np.ndarray, a_val: int, u0: float, u1: float) -> np.ndarray:
        x = rng.normal(loc=mean_shift, scale=1.0, size=(mc_size, p.p))
        a = np.full(mc_size, a_val)
        lin = p.beta0_cont + x @ p.beta + tau * a + (x * a[:, None]) @ eta + u0 + (u1 - u0) * a
        y = lin + rng.normal(0.0, p.sigma, size=mc_size)
        return y

    mean_c = np.zeros(p.p)
    mean_h = gamma * mean_shift_base

    u_h0 = gamma * drift_on
    u_h1 = 0.0

    y_c0 = sample_y(mean_c, 0, 0.0, 0.0)
    y_h0 = sample_y(mean_h, 0, u_h0, u_h1)
    y_c1 = sample_y(mean_c, 1, 0.0, 0.0)
    y_h1 = sample_y(mean_h, 1, u_h0, u_h1)

    w1_c0 = wasserstein_1d(y_c0, y_h0)
    w1_c1 = wasserstein_1d(y_c1, y_h1)
    return w1_c0, w1_c1


def dro_shift_bounds(outcome: str, rho: float, mu_c: float) -> tuple[float, float]:
    if outcome == "continuous":
        return float(rho), float(-rho)
    mu = clamp_prob(mu_c)
    delta_plus = min(float(rho), 1.0 - mu)
    delta_minus = -min(float(rho), mu)
    return float(delta_plus), float(delta_minus)


def dro_bias_plus(
    outcome: str,
    w0: float,
    w1: float,
    rho0: float,
    rho1: float,
    mu_c0: float,
    mu_c1: float,
) -> float:
    delta1_plus, _ = dro_shift_bounds(outcome, rho1, mu_c1)
    _, delta0_minus = dro_shift_bounds(outcome, rho0, mu_c0)
    return float(w1 * delta1_plus - w0 * delta0_minus)


def dro_kappa(
    outcome: str,
    theta1: float,
    w0: float,
    w1: float,
    rho0: float,
    rho1: float,
    mu_c0: float,
    mu_c1: float,
    s: float,
) -> float:
    delta1_plus, delta1_minus = dro_shift_bounds(outcome, rho1, mu_c1)
    delta0_plus, delta0_minus = dro_shift_bounds(outcome, rho0, mu_c0)
    numerator = theta1 - w1 * (delta1_plus - delta1_minus) - w0 * (delta0_plus - delta0_minus)
    return float(numerator / s)


def dro_lambda_star(
    outcome: str,
    rho0: float,
    rho1: float,
    theta1: float,
    mu_c0: float,
    mu_c1: float,
    var_c0: float,
    var_c1: float,
    var_h0: float,
    var_h1: float,
    n_c0: int,
    n_c1: int,
    n_h0: int,
    n_h1: int,
    grid: int,
) -> tuple[float, float, float]:
    lam_grid = np.linspace(0.0, 1.0, grid)
    lam0_grid = lam_grid if n_h0 > 0 else np.array([0.0])
    lam1_grid = lam_grid if n_h1 > 0 else np.array([0.0])
    best_lam0 = 0.0
    best_lam1 = 0.0
    best_kappa = -np.inf
    for lam0 in lam0_grid:
        w0 = w_from_lambda(float(lam0), n_c0, n_h0)
        for lam1 in lam1_grid:
            w1 = w_from_lambda(float(lam1), n_c1, n_h1)
            s2 = (
                (1.0 - w1) ** 2 * var_c1 / max(n_c1, 1)
                + w1**2 * (var_h1 / max(n_h1, 1) if n_h1 > 0 else 0.0)
                + (1.0 - w0) ** 2 * var_c0 / max(n_c0, 1)
                + w0**2 * (var_h0 / max(n_h0, 1) if n_h0 > 0 else 0.0)
            )
            if s2 <= 0.0:
                continue
            s = float(np.sqrt(s2))
            kappa = dro_kappa(outcome, theta1, w0, w1, rho0, rho1, mu_c0, mu_c1, s)
            if kappa > best_kappa:
                best_kappa = kappa
                best_lam0 = float(lam0)
                best_lam1 = float(lam1)
    return best_lam0, best_lam1, float(best_kappa)


def build_method_specs(p: Params) -> list[MethodSpec]:
    specs = [
        MethodSpec(name="Current-only", family="current"),
        MethodSpec(name="Naive pooling", family="naive"),
    ]
    for lam in p.fixed_lambdas:
        specs.append(MethodSpec(name=f"Fixed lambda={lam:.2f}", family="fixed", param=float(lam)))
    for lam in p.power_prior_lambdas:
        specs.append(
            MethodSpec(
                name=f"Power prior(lambda={lam:.2f})",
                family="power_prior",
                param=float(lam),
            )
        )
    for tau in p.commensurate_taus:
        specs.append(
            MethodSpec(
                name=f"Commensurate prior(tau={tau:.2f})",
                family="commensurate",
                param=float(tau),
            )
        )
    specs.append(MethodSpec(name="MAP prior", family="map"))
    for eps in p.rmap_epsilons:
        specs.append(
            MethodSpec(
                name=f"Robust MAP(epsilon={eps:.2f})",
                family="robust_map",
                param=float(eps),
            )
        )
    for scale in p.elastic_scales:
        specs.append(
            MethodSpec(
                name=f"Elastic prior(scale={scale:.2f})",
                family="elastic",
                param=float(scale),
            )
        )
    for m_units in p.uip_m_values:
        specs.append(
            MethodSpec(
                name=f"Unit-Info prior(m={m_units:.0f})",
                family="uip",
                param=float(m_units),
            )
        )
    specs.append(MethodSpec(name="LEAP", family="leap"))
    specs.append(MethodSpec(name="EXNEX", family="exnex"))
    specs.append(MethodSpec(name="MEM", family="mem"))
    specs.append(MethodSpec(name="BHMOI", family="bhmoi"))
    specs.append(MethodSpec(name="Nonpara Bayes", family="npb"))
    specs.append(MethodSpec(name="Test-then-pool", family="test_then_pool"))
    specs.append(MethodSpec(name="DRO-opt", family="dro"))
    return specs


def build_method_styles(method_order: list[str]) -> dict[str, dict[str, object]]:
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "8", "p", "1"]
    linestyles: list[object] = [
        "-",
        "--",
        "-.",
        ":",
        (0, (5, 1)),
        (0, (3, 1, 1, 1)),
        (0, (1, 1)),
    ]
    colors = list(plt.cm.tab20.colors) + list(plt.cm.Set2.colors)
    marker_style_pairs = [(m, ls) for ls in linestyles for m in markers]

    style_map: dict[str, dict[str, object]] = {}
    for idx, method in enumerate(method_order):
        marker, linestyle = marker_style_pairs[idx % len(marker_style_pairs)]
        style_map[method] = {
            "marker": marker,
            "linestyle": linestyle,
            "color": colors[idx % len(colors)],
            "linewidth": 1.8,
            "markersize": 5.5,
        }
    if "DRO-opt" in style_map:
        style_map["DRO-opt"].update(
            {
                "color": "#c40000",
                "linewidth": 2.8,
                "markersize": 7.0,
                "marker": "o",
                "linestyle": "-",
            }
        )
    return style_map


def select_focus_methods(method_order: list[str]) -> list[str]:
    preferred = [
        "Current-only",
        "Naive pooling",
        "Fixed lambda=0.50",
        "Power prior(lambda=0.50)",
        "Commensurate prior(tau=1.00)",
        "Robust MAP(epsilon=0.20)",
        "DRO-opt",
    ]
    selected = [m for m in preferred if m in method_order]
    for m in method_order:
        if m not in selected and m == "DRO-opt":
            selected.append(m)
    return selected


def estimate_arm_for_method(
    method: MethodSpec,
    arm: ArmSummary,
    outcome: str,
    p: Params,
    arm_index: int,
    dro_lambda0: float,
    dro_lambda1: float,
) -> PosteriorSummary:
    if method.family == "current":
        return pooled_estimate(arm, 0.0, outcome)
    if method.family == "naive":
        return pooled_estimate(arm, 1.0, outcome)
    if method.family == "fixed":
        assert method.param is not None
        return pooled_estimate(arm, method.param, outcome)
    if method.family == "power_prior":
        assert method.param is not None
        return posterior_power_prior(arm, outcome, method.param, p)
    if method.family == "commensurate":
        assert method.param is not None
        return posterior_commensurate(arm, outcome, method.param)
    if method.family == "map":
        return posterior_map(arm, outcome, p)
    if method.family == "robust_map":
        assert method.param is not None
        return posterior_rmap(arm, outcome, p, method.param)
    if method.family == "elastic":
        assert method.param is not None
        return posterior_elastic(arm, outcome, method.param)
    if method.family == "uip":
        assert method.param is not None
        return posterior_uip(arm, outcome, method.param)
    if method.family == "leap":
        return posterior_leap(arm, outcome, p)
    if method.family == "exnex":
        return posterior_exnex(arm, outcome, p)
    if method.family == "mem":
        return posterior_mem(arm, outcome, p)
    if method.family == "bhmoi":
        return posterior_bhmoi(arm, outcome, p)
    if method.family == "npb":
        return posterior_nonpara_bayes(arm, outcome, p)
    if method.family == "test_then_pool":
        if arm.n_h <= 0:
            return pooled_estimate(arm, 0.0, outcome)
        if outcome == "binary":
            se2 = var_mean_binary(arm.mean_c, arm.n_c) + var_mean_binary(arm.mean_h, arm.n_h)
        else:
            se2 = var_mean_cont(arm.var_c, arm.n_c) + var_mean_cont(arm.var_h, arm.n_h)
        if se2 <= 0.0:
            return pooled_estimate(arm, 0.0, outcome)
        z = (arm.mean_c - arm.mean_h) / np.sqrt(se2)
        pval = float(2.0 * norm.sf(abs(float(z))))
        lam = 1.0 if pval > p.alpha_pool else 0.0
        return pooled_estimate(arm, lam, outcome)
    if method.family == "dro":
        lam = dro_lambda0 if arm_index == 0 else dro_lambda1
        return pooled_estimate(arm, lam, outcome)
    raise ValueError(f"Unknown method family: {method.family}")


def simulate_rates(
    rng: np.random.Generator,
    outcome: str,
    scenario: Scenario,
    gamma: float,
    n_current: int,
    n_historical: int,
    tau_effect: float,
    m: int,
    alpha: float,
    rho0: float,
    rho1: float,
    dro_lambda0: float,
    dro_lambda1: float,
    method_specs: list[MethodSpec],
    p: Params,
) -> dict[str, float]:
    dgp = DGPParams(
        p=2,
        beta=np.array([0.5, 0.5]),
        beta0_cont=0.0,
        beta0_bin=-1.0,
        eta_effect=np.array([0.3, 0.3]),
        sigma=1.0,
        pi=0.5,
    )
    eta, mean_shift_base, drift_on, hist_arms = scenario_params(scenario, dgp)
    mean_c = np.zeros(dgp.p)
    mean_h = gamma * mean_shift_base

    z = z_alpha(alpha)

    methods = [method.name for method in method_specs]
    counts = {name: 0 for name in methods}

    if m <= 0:
        return {method: 0.0 for method in methods}

    for _ in range(m):
        a_c, y_c = simulate_trial(
            rng,
            n_current,
            mean_c,
            dgp.beta0_cont if outcome == "continuous" else dgp.beta0_bin,
            dgp.beta,
            eta,
            tau_effect,
            dgp.sigma,
            dgp.pi,
            0.0,
            0.0,
            outcome,
            "both",
        )
        a_h, y_h = simulate_trial(
            rng,
            n_historical,
            mean_h,
            dgp.beta0_cont if outcome == "continuous" else dgp.beta0_bin,
            dgp.beta,
            eta,
            tau_effect,
            dgp.sigma,
            dgp.pi,
            gamma * drift_on,
            0.0,
            outcome,
            hist_arms,
        )

        y_c0 = y_c[a_c == 0]
        y_c1 = y_c[a_c == 1]
        y_h0 = y_h[a_h == 0]
        y_h1 = y_h[a_h == 1]

        arm0 = build_arm_summary(y_c0, y_h0, outcome)
        arm1 = build_arm_summary(y_c1, y_h1, outcome)

        for method in method_specs:
            est0 = estimate_arm_for_method(
                method=method,
                arm=arm0,
                outcome=outcome,
                p=p,
                arm_index=0,
                dro_lambda0=dro_lambda0,
                dro_lambda1=dro_lambda1,
            )
            est1 = estimate_arm_for_method(
                method=method,
                arm=arm1,
                outcome=outcome,
                p=p,
                arm_index=1,
                dro_lambda0=dro_lambda0,
                dro_lambda1=dro_lambda1,
            )

            theta_hat = est1.mean - est0.mean
            s2 = est1.var + est0.var
            if s2 <= 0.0:
                continue
            s = np.sqrt(s2)

            if method.family == "dro":
                b_plus = dro_bias_plus(
                    outcome=outcome,
                    w0=est0.w_eff,
                    w1=est1.w_eff,
                    rho0=rho0,
                    rho1=rho1,
                    mu_c0=arm0.mean_c,
                    mu_c1=arm1.mean_c,
                )
                stat = (theta_hat - b_plus) / s
            else:
                stat = theta_hat / s

            if stat >= z:
                counts[method.name] += 1

    return {method: counts[method] / m for method in methods}


def run_simulation(p: Params) -> None:
    set_plot_style()
    p.outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(p.seed)

    dgp = DGPParams(
        p=2,
        beta=np.array([0.5, 0.5]),
        beta0_cont=0.0,
        beta0_bin=-1.0,
        eta_effect=np.array([0.3, 0.3]),
        sigma=1.0,
        pi=0.5,
    )

    results_rows: list[str] = []
    worst_rows: list[str] = []
    lambda_rows: list[str] = []
    consistency_warnings: list[str] = []
    method_specs = build_method_specs(p)
    method_order = [method.name for method in method_specs]
    style_map = build_method_styles(method_order)
    focus_methods = select_focus_methods(method_order)

    for n_historical in p.n_historical_grid:
        for outcome in p.outcomes:
            for scen_key in p.scenarios:
                scenario = SCENARIOS[scen_key]
                eta_scenario, _, _, _ = scenario_params(scenario, dgp)

                if outcome == "binary":
                    tau_null = calibrate_binary_tau(
                        beta0=dgp.beta0_bin,
                        beta=dgp.beta,
                        eta=eta_scenario,
                        theta_target=0.0,
                        p_dim=dgp.p,
                        mc_size=max(40000, p.mc_moments),
                        seed=p.seed + 9000 + 17 * n_historical + len(scen_key),
                    )
                    tau_alt = calibrate_binary_tau(
                        beta0=dgp.beta0_bin,
                        beta=dgp.beta,
                        eta=eta_scenario,
                        theta_target=p.theta1,
                        p_dim=dgp.p,
                        mc_size=max(40000, p.mc_moments),
                        seed=p.seed + 9500 + 17 * n_historical + len(scen_key),
                    )
                else:
                    tau_null = 0.0
                    tau_alt = p.theta1

                # Precompute design moments for each gamma.
                design_moments_null: dict[float, Moments] = {}
                design_moments_alt: dict[float, Moments] = {}
                w1_map_null: dict[float, tuple[float, float]] = {}

                for gamma in p.gamma_grid:
                    seed = p.seed + int(1000 * gamma) + (0 if outcome == "continuous" else 7)
                    if outcome == "continuous":
                        moments_null = estimate_moments_continuous(scenario, dgp, gamma, tau=tau_null)
                        moments_alt = estimate_moments_continuous(scenario, dgp, gamma, tau=tau_alt)
                        w1_c0, w1_c1 = estimate_w1_continuous(
                            scenario, dgp, gamma, tau=tau_null, mc_size=p.mc_moments, seed=seed
                        )
                    else:
                        moments_null = estimate_moments_binary(
                            scenario,
                            dgp,
                            gamma,
                            tau=tau_null,
                            mc_size=p.mc_moments,
                            seed=seed,
                        )
                        moments_alt = estimate_moments_binary(
                            scenario,
                            dgp,
                            gamma,
                            tau=tau_alt,
                            mc_size=p.mc_moments,
                            seed=seed + 101,
                        )
                        w1_c0 = abs(moments_null.mu_h0 - moments_null.mu_c0)
                        w1_c1 = abs(moments_null.mu_h1 - moments_null.mu_c1)

                    if gamma == 0.0:
                        w1_c0 = 0.0
                        w1_c1 = 0.0

                    design_moments_null[gamma] = moments_null
                    design_moments_alt[gamma] = moments_alt
                    w1_map_null[gamma] = (w1_c0, w1_c1)

                for stage in p.stages:
                    if stage == "oracle":
                        stage_variants: list[tuple[str, float | None]] = [("oracle", None)]
                    elif stage == "data":
                        stage_variants = [(f"data_c{c:g}", c) for c in p.rho_multipliers]
                    else:
                        raise ValueError(f"Unknown stage: {stage}")

                    for stage_name, rho_multiplier in stage_variants:
                        type1_table: list[dict[str, float]] = []
                        power_table: list[dict[str, float]] = []
                        lambda_table: list[tuple[float, float, float, float, float, float]] = []

                        for gamma in p.gamma_grid:
                            moments_null = design_moments_null[gamma]
                            moments_alt = design_moments_alt[gamma]
                            if stage_name == "oracle":
                                rho0 = abs(moments_null.mu_h0 - moments_null.mu_c0)
                                rho1 = abs(moments_null.mu_h1 - moments_null.mu_c1)
                            else:
                                w1_c0, w1_c1 = w1_map_null[gamma]
                                assert rho_multiplier is not None
                                rho0 = rho_multiplier * w1_c0
                                rho1 = rho_multiplier * w1_c1

                            n_c0 = p.n_current // 2
                            n_c1 = p.n_current - n_c0
                            if scenario.hist_arms == "both":
                                n_h0 = n_historical // 2
                                n_h1 = n_historical - n_h0
                            else:
                                n_h0 = n_historical
                                n_h1 = 0

                            dro_lam0, dro_lam1, dro_kappa_val = dro_lambda_star(
                                outcome=outcome,
                                rho0=rho0,
                                rho1=rho1,
                                theta1=p.theta1,
                                mu_c0=moments_alt.mu_c0,
                                mu_c1=moments_alt.mu_c1,
                                var_c0=moments_alt.var_c0,
                                var_c1=moments_alt.var_c1,
                                var_h0=moments_alt.var_h0,
                                var_h1=moments_alt.var_h1,
                                n_c0=n_c0,
                                n_c1=n_c1,
                                n_h0=n_h0,
                                n_h1=n_h1,
                                grid=p.lambda_grid,
                            )

                            type1 = simulate_rates(
                                rng=rng,
                                outcome=outcome,
                                scenario=scenario,
                                gamma=gamma,
                                n_current=p.n_current,
                                n_historical=n_historical,
                                tau_effect=tau_null,
                                m=p.m_type1,
                                alpha=p.alpha,
                                rho0=rho0,
                                rho1=rho1,
                                dro_lambda0=dro_lam0,
                                dro_lambda1=dro_lam1,
                                method_specs=method_specs,
                                p=p,
                            )
                            power = simulate_rates(
                                rng=rng,
                                outcome=outcome,
                                scenario=scenario,
                                gamma=gamma,
                                n_current=p.n_current,
                                n_historical=n_historical,
                                tau_effect=tau_alt,
                                m=p.m_power,
                                alpha=p.alpha,
                                rho0=rho0,
                                rho1=rho1,
                                dro_lambda0=dro_lam0,
                                dro_lambda1=dro_lam1,
                                method_specs=method_specs,
                                p=p,
                            )

                            type1_table.append(type1)
                            power_table.append(power)

                            w0 = w_from_lambda(dro_lam0, n_c0, n_h0)
                            w1 = w_from_lambda(dro_lam1, n_c1, n_h1)
                            lambda_table.append((gamma, dro_lam0, dro_lam1, w0, w1, dro_kappa_val))
                            lambda_rows.append(
                                ",".join(
                                    [
                                        str(n_historical),
                                        outcome,
                                        scenario.key,
                                        stage_name,
                                        f"{gamma:.4f}",
                                        f"{rho0:.6f}",
                                        f"{rho1:.6f}",
                                        f"{dro_lam0:.6f}",
                                        f"{dro_lam1:.6f}",
                                        f"{w0:.6f}",
                                        f"{w1:.6f}",
                                        f"{dro_kappa_val:.6f}",
                                    ]
                                )
                            )

                            for method in type1.keys():
                                results_rows.append(
                                    ",".join(
                                        [
                                            str(n_historical),
                                            outcome,
                                            scenario.key,
                                            stage_name,
                                            f"{gamma:.4f}",
                                            method,
                                            f"{type1[method]:.6f}",
                                            f"{power[method]:.6f}",
                                        ]
                                    )
                                )

                        methods = type1_table[0].keys()
                        for method in methods:
                            max_type1 = max(row[method] for row in type1_table)
                            min_power = min(row[method] for row in power_table)
                            worst_rows.append(
                                ",".join(
                                    [
                                        str(n_historical),
                                        outcome,
                                        scenario.key,
                                        stage_name,
                                        method,
                                        f"{max_type1:.6f}",
                                        f"{min_power:.6f}",
                                    ]
                                )
                            )

                        if "DRO-opt" in type1_table[0]:
                            dro_max = max(row["DRO-opt"] for row in type1_table)
                            mc_se = math.sqrt(p.alpha * (1.0 - p.alpha) / p.m_type1)
                            tolerance = 3.0 * mc_se + 0.01
                            if stage_name == "oracle" and dro_max > p.alpha + tolerance:
                                consistency_warnings.append(
                                    f"Warning: DRO size check borderline for nH={n_historical}, {outcome}, {scenario.key}, {stage_name}. "  # noqa: E501
                                    f"max Type-I={dro_max:.4f}, alpha={p.alpha:.4f}, tol={tolerance:.4f}"
                                )

                        stage_file = stage_name.replace(".", "p")
                        fig, axes = plt.subplots(2, 1, figsize=(10, 8.3), sharex=True)
                        for method in method_order:
                            if method not in type1_table[0]:
                                continue
                            style = style_map[method]
                            axes[0].plot(
                                p.gamma_grid,
                                [row[method] for row in type1_table],
                                label=method,
                                marker=style["marker"],
                                linestyle=style["linestyle"],
                                color=style["color"],
                                linewidth=style["linewidth"],
                                markersize=style["markersize"],
                            )
                            axes[1].plot(
                                p.gamma_grid,
                                [row[method] for row in power_table],
                                label=method,
                                marker=style["marker"],
                                linestyle=style["linestyle"],
                                color=style["color"],
                                linewidth=style["linewidth"],
                                markersize=style["markersize"],
                            )

                        axes[0].axhline(
                            p.alpha,
                            color="black",
                            linestyle=(0, (4, 2)),
                            alpha=0.7,
                            label=math_label("\\alpha"),
                        )
                        axes[0].set_ylabel("Type-I error")
                        axes[0].set_title(
                            f"Type-I error vs {math_label('\\\\gamma')} ({outcome}, {scenario.key}, n_H={n_historical}, {stage_name})"  # noqa: E501
                        )
                        axes[0].grid(True, alpha=0.3)
                        axes[0].legend(fontsize=8.5, ncol=2)

                        axes[1].set_ylabel("Power")
                        axes[1].set_xlabel(math_label("\\gamma"))
                        axes[1].set_title(
                            f"Power vs {math_label('\\\\gamma')} ({outcome}, {scenario.key}, n_H={n_historical}, {stage_name})"  # noqa: E501
                        )
                        axes[1].grid(True, alpha=0.3)
                        axes[1].legend(fontsize=8.5, ncol=2)

                        fig.tight_layout()
                        fig.savefig(
                            p.outdir / f"type1_power_{outcome}_{scenario.key}_{stage_file}_nH{n_historical}.pdf",
                            format="pdf",
                        )
                        plt.close(fig)

                        fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.6), sharex=True)
                        for method in focus_methods:
                            if method not in type1_table[0]:
                                continue
                            style = style_map[method]
                            axes[0].plot(
                                p.gamma_grid,
                                [row[method] for row in type1_table],
                                label=method,
                                marker=style["marker"],
                                linestyle=style["linestyle"],
                                color=style["color"],
                                linewidth=style["linewidth"],
                                markersize=style["markersize"],
                            )
                            axes[1].plot(
                                p.gamma_grid,
                                [row[method] for row in power_table],
                                label=method,
                                marker=style["marker"],
                                linestyle=style["linestyle"],
                                color=style["color"],
                                linewidth=style["linewidth"],
                                markersize=style["markersize"],
                            )
                        axes[0].axhline(
                            p.alpha,
                            color="black",
                            linestyle=(0, (4, 2)),
                            alpha=0.7,
                            label=math_label("\\alpha"),
                        )
                        axes[0].set_ylabel("Type-I error")
                        axes[0].set_title(
                            f"Focused comparison ({outcome}, {scenario.key}, n_H={n_historical}, {stage_name})"
                        )
                        axes[0].grid(True, alpha=0.3)
                        axes[0].legend(fontsize=9.5, ncol=2)
                        axes[1].set_ylabel("Power")
                        axes[1].set_xlabel(math_label("\\gamma"))
                        axes[1].grid(True, alpha=0.3)
                        axes[1].legend(fontsize=9.5, ncol=2)
                        fig.tight_layout()
                        fig.savefig(
                            p.outdir / f"type1_power_focus_{outcome}_{scenario.key}_{stage_file}_nH{n_historical}.pdf",
                            format="pdf",
                        )
                        plt.close(fig)

                        fig, ax = plt.subplots(figsize=(9, 4.8))
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
                            linewidth=2.2,
                            label=math_label("\\lambda_0^*"),
                        )
                        if scenario.hist_arms == "both":
                            ax.plot(
                                gammas,
                                lam1_vals,
                                marker="D",
                                linestyle="--",
                                linewidth=2.0,
                                label=math_label("\\lambda_1^*"),
                            )
                        ax.plot(
                            gammas,
                            w0_vals,
                            marker="s",
                            linestyle="-.",
                            linewidth=1.9,
                            label=math_label("w_0(\\lambda^*)"),
                        )
                        if scenario.hist_arms == "both":
                            ax.plot(
                                gammas,
                                w1_vals,
                                marker="^",
                                linestyle=":",
                                linewidth=1.9,
                                label=math_label("w_1(\\lambda^*)"),
                            )
                        ax.set_xlabel(math_label("\\gamma"))
                        ax.set_ylabel("Borrowing level")
                        ax.set_title(
                            f"Optimal borrowing vs {math_label('\\\\gamma')} ({outcome}, {scenario.key}, n_H={n_historical}, {stage_name})"  # noqa: E501
                        )
                        ax.set_ylim(0.0, 1.0)
                        ax.grid(True, alpha=0.3)
                        ax.legend(fontsize=10)
                        fig.tight_layout()
                        fig.savefig(
                            p.outdir / f"lambda_{outcome}_{scenario.key}_{stage_file}_nH{n_historical}.pdf",
                            format="pdf",
                        )
                        plt.close(fig)

    results_path = p.outdir / "summary.csv"
    with results_path.open("w", encoding="utf-8") as f:
        f.write("n_historical,outcome,scenario,stage,gamma,method,type1,power\n")
        f.write("\n".join(results_rows))
        f.write("\n")

    worst_path = p.outdir / "worst_case.csv"
    with worst_path.open("w", encoding="utf-8") as f:
        f.write("n_historical,outcome,scenario,stage,method,max_type1,min_power\n")
        f.write("\n".join(worst_rows))
        f.write("\n")

    lambda_path = p.outdir / "dro_lambda.csv"
    with lambda_path.open("w", encoding="utf-8") as f:
        f.write("n_historical,outcome,scenario,stage,gamma,rho0,rho1,lambda0,lambda1,w0,w1,kappa\n")
        f.write("\n".join(lambda_rows))
        f.write("\n")

    print("Simulation completed")
    print(f"Output directory: {p.outdir}")
    print(f"Summary table: {results_path}")
    print(f"Worst-case table: {worst_path}")
    print(f"DRO lambda table: {lambda_path}")
    for warning in consistency_warnings:
        print(warning)


def main() -> None:
    p = parse_args()
    validate_params(p)
    run_simulation(p)


if __name__ == "__main__":
    main()
