from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit  # type: ignore

import borrowing_methods as bm
import plot_utils as pu
import sim_utils as su


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
class BondCalibrationSummary:
    lambda0_mean: float
    lambda1_mean: float
    w0_mean: float
    w1_mean: float
    kappa_mean: float


class ProgressBar:
    """Simple terminal progress bar without external dependencies."""

    def __init__(self, total: int, enabled: bool = True, width: int = 32) -> None:
        self.total = max(int(total), 1)
        self.enabled = bool(enabled)
        self.width = max(int(width), 10)
        self.count = 0
        self._is_tty = bool(getattr(sys.stderr, "isatty", lambda: False)())
        self._last_plain_pct = -1
        if self.enabled:
            self._render("")

    def _render(self, suffix: str) -> None:
        ratio = float(self.count) / float(self.total)
        ratio = min(max(ratio, 0.0), 1.0)
        pct = int(round(ratio * 100.0))
        if self._is_tty:
            filled = int(round(ratio * self.width))
            bar = "#" * filled + "-" * (self.width - filled)
            msg = f"\r[{bar}] {self.count}/{self.total} ({pct:3d}%) {suffix}"
            print(msg, end="", file=sys.stderr, flush=True)
            return
        if pct >= self._last_plain_pct + 5 or self.count >= self.total:
            self._last_plain_pct = pct
            print(f"[{pct:3d}%] {self.count}/{self.total} {suffix}", file=sys.stderr, flush=True)

    def update(self, step: int = 1, suffix: str = "") -> None:
        if not self.enabled:
            return
        self.count = min(self.total, self.count + max(int(step), 0))
        self._render(suffix)

    def close(self) -> None:
        if self.enabled and self._is_tty:
            print(file=sys.stderr, flush=True)


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
    bond_calibration_mode: str
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
    npb_concentration: float | None
    npb_phi: float
    npb_temperature: float
    npb_sharpness: float
    beta_prior_alpha: float
    beta_prior_beta: float
    vague_alpha: float
    vague_beta: float
    normal_prior_mean: float
    normal_prior_var: float
    vague_normal_var: float
    show_progress: bool


def parse_args() -> Params:
    parser = argparse.ArgumentParser(description="DRO-based borrowing simulation (continuous + binary).")
    parser.add_argument("--seed", type=int, default=202402)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/dro_simulation"))
    parser.add_argument("--alpha", type=float, default=0.025)
    parser.add_argument("--theta1", type=float, default=0.3)
    parser.add_argument(
        "--gamma-grid",
        type=str,
        default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2",
    )
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
    parser.add_argument(
        "--bond-calibration-mode",
        type=str,
        default="replicate",
        choices=("design", "replicate"),
        help="How BOND lambdas are calibrated: once per gamma design point, or per simulated replicate.",
    )
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
        help="Concentration for LEAP prior exchangeability Beta distribution (>0).",
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
        "--npb-concentration",
        type=float,
        default=None,
        help=(
            "Concentration parameter M for nonparametric-Bayes (DPM/DDPM-inspired). "
            "If omitted, falls back to --npb-temperature for backward compatibility."
        ),
    )
    parser.add_argument(
        "--npb-phi",
        type=float,
        default=0.5,
        help="DDPM dependence parameter phi in [0,1] for nonparametric-Bayes.",
    )
    parser.add_argument(
        "--npb-temperature",
        type=float,
        default=1.0,
        help="Legacy fallback for nonparametric-Bayes concentration M (>0).",
    )
    parser.add_argument(
        "--npb-sharpness",
        type=float,
        default=1.0,
        help="Legacy parameter (kept for backward compatibility).",
    )
    parser.add_argument("--beta-prior-alpha", type=float, default=1.0)
    parser.add_argument("--beta-prior-beta", type=float, default=1.0)
    parser.add_argument("--vague-alpha", type=float, default=1.0)
    parser.add_argument("--vague-beta", type=float, default=1.0)
    parser.add_argument("--normal-prior-mean", type=float, default=0.0)
    parser.add_argument("--normal-prior-var", type=float, default=1e8)
    parser.add_argument("--vague-normal-var", type=float, default=1e8)
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar output.",
    )
    args = parser.parse_args()
    n_h_grid = su.parse_int_list(args.n_historical_grid) if args.n_historical_grid.strip() else [args.n_historical]
    rho_mults = su.parse_float_list(args.rho_multipliers) if args.rho_multipliers.strip() else [args.rho_multiplier]

    return Params(
        seed=args.seed,
        outdir=args.outdir,
        alpha=args.alpha,
        theta1=args.theta1,
        gamma_grid=su.parse_float_list(args.gamma_grid),
        n_current=args.n_current,
        n_historical_grid=n_h_grid,
        m_type1=args.m_type1,
        m_power=args.m_power,
        outcomes=[x.strip() for x in args.outcomes.split(",") if x.strip()],
        scenarios=[x.strip() for x in args.scenarios.split(",") if x.strip()],
        stages=[x.strip() for x in args.stages.split(",") if x.strip()],
        rho_multipliers=rho_mults,
        lambda_grid=args.lambda_grid,
        bond_calibration_mode=args.bond_calibration_mode,
        fixed_lambdas=su.parse_float_list(args.fixed_lambdas),
        mc_moments=args.mc_moments,
        alpha_pool=args.alpha_pool,
        power_prior_lambdas=su.parse_float_list(args.power_prior_lambdas),
        commensurate_taus=su.parse_float_list(args.commensurate_taus),
        map_lambdas=su.parse_float_list(args.map_lambdas),
        map_weights=su.parse_optional_float_list(args.map_weights),
        rmap_epsilons=su.parse_float_list(args.rmap_epsilons),
        elastic_scales=su.parse_float_list(args.elastic_scales),
        uip_m_values=su.parse_float_list(args.uip_m_values),
        leap_prior_omega=args.leap_prior_omega,
        leap_nex_scale=args.leap_nex_scale,
        exnex_prior_ex=args.exnex_prior_ex,
        mem_prior_inclusion=args.mem_prior_inclusion,
        mem_tau=args.mem_tau,
        bhmoi_sharpness=args.bhmoi_sharpness,
        npb_concentration=args.npb_concentration,
        npb_phi=args.npb_phi,
        npb_temperature=args.npb_temperature,
        npb_sharpness=args.npb_sharpness,
        beta_prior_alpha=args.beta_prior_alpha,
        beta_prior_beta=args.beta_prior_beta,
        vague_alpha=args.vague_alpha,
        vague_beta=args.vague_beta,
        normal_prior_mean=args.normal_prior_mean,
        normal_prior_var=args.normal_prior_var,
        vague_normal_var=args.vague_normal_var,
        show_progress=not bool(args.no_progress),
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
    if p.bond_calibration_mode not in {"design", "replicate"}:
        raise ValueError("bond-calibration-mode must be either 'design' or 'replicate'")
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
    if p.npb_concentration is not None and p.npb_concentration <= 0.0:
        raise ValueError("npb-concentration must be > 0 when provided")
    if not (0.0 <= p.npb_phi <= 1.0):
        raise ValueError("npb-phi must be in [0,1]")
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


def scenario_params(scenario: Scenario, p: DGPParams) -> Tuple[NDArray[np.float64], float, float, str]:
    """Get the scenario parameters

    Args:
        scenario (Scenario): The scenario
        p (DGPParams): The DGP parameters

    Returns:
        Tuple[NDArray[np.float64], float, float, str]: The scenario parameters
    """
    eta = p.eta_effect if scenario.effect_mod else np.zeros_like(p.eta_effect)
    mean_shift = np.ones(p.p) if scenario.cov_shift else np.zeros(p.p)
    return eta, mean_shift, (1.0 if scenario.control_drift else 0.0), scenario.hist_arms


def simulate_trial(
    rng: np.random.Generator,
    n: int,
    mean_shift: NDArray[np.float64],
    beta0: float,
    beta: NDArray[np.float64],
    eta: NDArray[np.float64],
    tau: float,
    sigma: float,
    pi: float,
    u0: float,
    u1: float,
    outcome: str,
    hist_arms: str,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Simulate a trial

    Args:
        rng (np.random.Generator): The random number generator
        n (int): The sample size
        mean_shift (NDArray[np.float64]): The mean shift
        beta0 (float): The intercept
        beta (NDArray[np.float64]): The beta coefficients
        eta (NDArray[np.float64]): The eta coefficients
        tau (float): The tau
        sigma (float): The sigma
        pi (float): The pi
        u0 (float): The u0
        u1 (float): The u1
        outcome (str): The outcome
        hist_arms (str): The historical arms

    Returns:
        Tuple[NDArray[np.float64], NDArray[np.float64]]: The simulated trial
    """
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


def build_arm_summary(y_c_arm: NDArray[np.float64], y_h_arm: NDArray[np.float64], outcome: str) -> ArmSummary:
    """Build an arm summary

    Args:
        y_c_arm (NDArray[np.float64]): The control arm
        y_h_arm (NDArray[np.float64]): The historical arm
        outcome (str): The outcome

    Returns:
        ArmSummary: The arm summary
    """
    n_c = int(y_c_arm.size)
    n_h = int(y_h_arm.size)
    sum_c = float(np.sum(y_c_arm)) if n_c > 0 else 0.0
    sum_h = float(np.sum(y_h_arm)) if n_h > 0 else 0.0
    mean_c = float(sum_c / n_c) if n_c > 0 else 0.0
    mean_h = float(sum_h / n_h) if n_h > 0 else 0.0
    var_c = float(np.var(y_c_arm, ddof=1)) if n_c > 1 else 0.0
    var_h = float(np.var(y_h_arm, ddof=1)) if n_h > 1 else 0.0
    if outcome == "binary":
        var_c = su.clamp_prob(mean_c) * (1.0 - su.clamp_prob(mean_c)) if n_c > 0 else 0.0
        var_h = su.clamp_prob(mean_h) * (1.0 - su.clamp_prob(mean_h)) if n_h > 0 else 0.0
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


def estimate_moments_continuous(
    scenario: Scenario,
    p: DGPParams,
    gamma: float,
    tau: float,
) -> Moments:
    """Estimate the moments

    Args:
        scenario (Scenario): The scenario
        p (DGPParams): The DGP parameters
        gamma (float): The gamma
        tau (float): The tau

    Returns:
        Moments: The moments
    """
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
    """Estimate the moments

    Args:
        scenario (Scenario): The scenario
        p (DGPParams): The DGP parameters
        gamma (float): The gamma
        tau (float): The tau
        mc_size (int): The Monte Carlo size
        seed (int): The seed

    Returns:
        Moments: The moments
    """
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
) -> Tuple[float, float]:
    """Estimate the w1

    Args:
        scenario (Scenario): The scenario
        p (DGPParams): The DGP parameters
        gamma (float): The gamma
        tau (float): The tau
        mc_size (int): The Monte Carlo size
        seed (int): The seed

    Returns:
        Tuple[float, float]: The w1
    """
    eta, mean_shift_base, drift_on, _ = scenario_params(scenario, p)
    rng = np.random.default_rng(seed)

    def sample_y(mean_shift: NDArray[np.float64], a_val: int, u0: float, u1: float) -> NDArray[np.float64]:
        """Sample the y

        Args:
            mean_shift (NDArray[np.float64]): The mean shift
            a_val (int): The a value
            u0 (float): The u0
            u1 (float): The u1

        Returns:
            NDArray[np.float64]: The y
        """
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

    w1_c0 = su.wasserstein_1d(y_c0, y_h0)
    w1_c1 = su.wasserstein_1d(y_c1, y_h1)
    return w1_c0, w1_c1


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
    bond_lambda0: float,
    bond_lambda1: float,
    method_specs: list[bm.MethodSpec],
    p: Params,
) -> tuple[Dict[str, float], BondCalibrationSummary]:
    """Simulate the rates

    Args:
        rng (np.random.Generator): The random number generator
        outcome (str): The outcome
        scenario (Scenario): The scenario
        gamma (float): The gamma
        n_current (int): The current sample size
        n_historical (int): The historical sample size
        tau_effect (float): The tau effect
        m (int): The number of simulations
        alpha (float): The alpha level
        rho0 (float): The rho0
        rho1 (float): The rho1
        bond_lambda0 (float): The bond lambda0
        bond_lambda1 (float): The bond lambda1
        method_specs (list[bm.MethodSpec]): The method specifications
        p (Params): The parameters

    Returns:
        tuple[Dict[str, float], BondCalibrationSummary]: The rates and BOND calibration summary
    """
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

    z = su.z_alpha(alpha)

    methods = [method.name for method in method_specs]
    counts = {name: 0 for name in methods}

    if m <= 0:
        return {method: 0.0 for method in methods}, BondCalibrationSummary(
            lambda0_mean=0.0,
            lambda1_mean=0.0,
            w0_mean=0.0,
            w1_mean=0.0,
            kappa_mean=float("nan"),
        )

    lam0_sum = 0.0
    lam1_sum = 0.0
    w0_sum = 0.0
    w1_sum = 0.0
    kappa_sum = 0.0
    kappa_count = 0

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

        if p.bond_calibration_mode == "replicate":
            bond_lambda0_rep, bond_lambda1_rep, bond_kappa_rep = bm.bond_lambda_star(
                outcome=outcome,
                rho0=rho0,
                rho1=rho1,
                theta1=p.theta1,
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
                grid=p.lambda_grid,
            )
        else:
            bond_lambda0_rep = bond_lambda0
            bond_lambda1_rep = bond_lambda1
            w0_tmp = bm.w_from_lambda(bond_lambda0_rep, arm0.n_c, arm0.n_h)
            w1_tmp = bm.w_from_lambda(bond_lambda1_rep, arm1.n_c, arm1.n_h)
            s2_tmp = (
                (1.0 - w1_tmp) ** 2 * (arm1.var_c / max(arm1.n_c, 1))
                + w1_tmp**2 * (arm1.var_h / max(arm1.n_h, 1) if arm1.n_h > 0 else 0.0)
                + (1.0 - w0_tmp) ** 2 * (arm0.var_c / max(arm0.n_c, 1))
                + w0_tmp**2 * (arm0.var_h / max(arm0.n_h, 1) if arm0.n_h > 0 else 0.0)
            )
            if s2_tmp > 0.0:
                bond_kappa_rep = bm.bond_kappa(
                    outcome=outcome,
                    theta1=p.theta1,
                    w0=w0_tmp,
                    w1=w1_tmp,
                    rho0=rho0,
                    rho1=rho1,
                    mu_c0=arm0.mean_c,
                    mu_c1=arm1.mean_c,
                    s=float(math.sqrt(s2_tmp)),
                )
            else:
                bond_kappa_rep = float("nan")

        bond_w0_rep = bm.w_from_lambda(bond_lambda0_rep, arm0.n_c, arm0.n_h)
        bond_w1_rep = bm.w_from_lambda(bond_lambda1_rep, arm1.n_c, arm1.n_h)
        lam0_sum += float(bond_lambda0_rep)
        lam1_sum += float(bond_lambda1_rep)
        w0_sum += float(bond_w0_rep)
        w1_sum += float(bond_w1_rep)
        if np.isfinite(bond_kappa_rep):
            kappa_sum += float(bond_kappa_rep)
            kappa_count += 1

        for method in method_specs:
            est0 = bm.estimate_arm_for_method(
                method=method,
                arm=arm0,
                outcome=outcome,
                p=p,
                arm_index=0,
                bond_lambda0=bond_lambda0_rep,
                bond_lambda1=bond_lambda1_rep,
            )
            est1 = bm.estimate_arm_for_method(
                method=method,
                arm=arm1,
                outcome=outcome,
                p=p,
                arm_index=1,
                bond_lambda0=bond_lambda0_rep,
                bond_lambda1=bond_lambda1_rep,
            )

            theta_hat = est1.mean - est0.mean
            s2 = est1.var + est0.var
            if s2 <= 0.0:
                continue
            s = np.sqrt(s2)

            if method.family == "bond":
                b_plus = bm.bond_bias_plus(
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

    kappa_mean = float(kappa_sum / kappa_count) if kappa_count > 0 else float("nan")
    summary = BondCalibrationSummary(
        lambda0_mean=float(lam0_sum / m),
        lambda1_mean=float(lam1_sum / m),
        w0_mean=float(w0_sum / m),
        w1_mean=float(w1_sum / m),
        kappa_mean=kappa_mean,
    )
    return {method: counts[method] / m for method in methods}, summary


def run_simulation(p: Params) -> None:
    """Run the simulation

    Args:
        p (Params): The parameters
    """
    pu.set_plot_style()
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
    method_specs = bm.build_method_specs(p)
    method_order = [method.name for method in method_specs]
    style_map = pu.build_method_styles(method_order)
    focus_methods = pu.select_focus_methods(method_order)

    def stage_variant_count(stage: str) -> int:
        if stage == "oracle":
            return 1
        if stage == "data":
            return len(p.rho_multipliers)
        raise ValueError(f"Unknown stage: {stage}")

    total_steps = 0
    for _ in p.n_historical_grid:
        for _ in p.outcomes:
            for _ in p.scenarios:
                for stage in p.stages:
                    total_steps += stage_variant_count(stage) * len(p.gamma_grid)
    progress = ProgressBar(total=total_steps, enabled=p.show_progress)

    try:
        for n_historical in p.n_historical_grid:
            for outcome in p.outcomes:
                for scen_key in p.scenarios:
                    scenario = SCENARIOS[scen_key]
                    eta_scenario, _, _, _ = scenario_params(scenario, dgp)

                    if outcome == "binary":
                        tau_null = su.calibrate_binary_tau(
                            beta0=dgp.beta0_bin,
                            beta=dgp.beta,
                            eta=eta_scenario,
                            theta_target=0.0,
                            p_dim=dgp.p,
                            mc_size=max(40000, p.mc_moments),
                            seed=p.seed + 9000 + 17 * n_historical + len(scen_key),
                        )
                        tau_alt = su.calibrate_binary_tau(
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
                    w1_map_null: dict[float, tuple[float, float]] = {}

                    for gamma in p.gamma_grid:
                        seed = p.seed + int(1000 * gamma) + (0 if outcome == "continuous" else 7)
                        if outcome == "continuous":
                            moments_null = estimate_moments_continuous(scenario, dgp, gamma, tau=tau_null)
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
                            w1_c0 = abs(moments_null.mu_h0 - moments_null.mu_c0)
                            w1_c1 = abs(moments_null.mu_h1 - moments_null.mu_c1)

                        if gamma == 0.0:
                            w1_c0 = 0.0
                            w1_c1 = 0.0

                        design_moments_null[gamma] = moments_null
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

                            n_c0 = p.n_current // 2
                            n_c1 = p.n_current - n_c0
                            if scenario.hist_arms == "both":
                                n_h0 = n_historical // 2
                                n_h1 = n_historical - n_h0
                            else:
                                n_h0 = n_historical
                                n_h1 = 0

                            for gamma in p.gamma_grid:
                                moments_null = design_moments_null[gamma]
                                if stage_name == "oracle":
                                    rho0 = abs(moments_null.mu_h0 - moments_null.mu_c0)
                                    rho1 = abs(moments_null.mu_h1 - moments_null.mu_c1)
                                else:
                                    w1_c0, w1_c1 = w1_map_null[gamma]
                                    assert rho_multiplier is not None
                                    rho0 = rho_multiplier * w1_c0
                                    rho1 = rho_multiplier * w1_c1

                                if p.bond_calibration_mode == "design":
                                    bond_lam0, bond_lam1, bond_kappa_val = bm.bond_lambda_star(
                                        outcome=outcome,
                                        rho0=rho0,
                                        rho1=rho1,
                                        theta1=p.theta1,
                                        mu_c0=moments_null.mu_c0,
                                        mu_c1=moments_null.mu_c1,
                                        var_c0=moments_null.var_c0,
                                        var_c1=moments_null.var_c1,
                                        var_h0=moments_null.var_h0,
                                        var_h1=moments_null.var_h1,
                                        n_c0=n_c0,
                                        n_c1=n_c1,
                                        n_h0=n_h0,
                                        n_h1=n_h1,
                                        grid=p.lambda_grid,
                                    )
                                else:
                                    bond_lam0 = 0.0
                                    bond_lam1 = 0.0
                                    bond_kappa_val = float("nan")

                                type1, type1_bond = simulate_rates(
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
                                    bond_lambda0=bond_lam0,
                                    bond_lambda1=bond_lam1,
                                    method_specs=method_specs,
                                    p=p,
                                )
                                power, _ = simulate_rates(
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
                                    bond_lambda0=bond_lam0,
                                    bond_lambda1=bond_lam1,
                                    method_specs=method_specs,
                                    p=p,
                                )

                                type1_table.append(type1)
                                power_table.append(power)

                                if p.bond_calibration_mode == "replicate":
                                    bond_lam0 = type1_bond.lambda0_mean
                                    bond_lam1 = type1_bond.lambda1_mean
                                    w0 = type1_bond.w0_mean
                                    w1 = type1_bond.w1_mean
                                    bond_kappa_val = type1_bond.kappa_mean
                                else:
                                    w0 = bm.w_from_lambda(bond_lam0, n_c0, n_h0)
                                    w1 = bm.w_from_lambda(bond_lam1, n_c1, n_h1)

                                lambda_table.append((gamma, bond_lam0, bond_lam1, w0, w1, bond_kappa_val))
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
                                            f"{bond_lam0:.6f}",
                                            f"{bond_lam1:.6f}",
                                            f"{w0:.6f}",
                                            f"{w1:.6f}",
                                            f"{bond_kappa_val:.6f}",
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
                                progress.update(
                                    1,
                                    (
                                        f"nH={n_historical} | {outcome} | {scenario.key} | "
                                        f"{stage_name} | gamma={gamma:g}"
                                    ),
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

                            if "BOND" in type1_table[0]:
                                bond_max = max(row["BOND"] for row in type1_table)
                                mc_se = math.sqrt(p.alpha * (1.0 - p.alpha) / p.m_type1)
                                tolerance = 3.0 * mc_se + 0.01
                                if stage_name == "oracle" and bond_max > p.alpha + tolerance:
                                    consistency_warnings.append(
                                        f"Warning: DRO size check borderline for nH={n_historical}, {outcome}, {scenario.key}, {stage_name}. "  # noqa: E501
                                        f"max Type-I={bond_max:.4f}, alpha={p.alpha:.4f}, tol={tolerance:.4f}"
                                    )

                            stage_file = stage_name.replace(".", "p")
                            pu.save_type1_power_figure(
                                outdir=p.outdir,
                                gamma_grid=p.gamma_grid,
                                type1_table=type1_table,
                                power_table=power_table,
                                methods=method_order,
                                style_map=style_map,
                                alpha=p.alpha,
                                filename_stem=f"type1_power_{outcome}_{scenario.key}_{stage_file}_nH{n_historical}",
                                figsize=(12.8, 9.8),
                            )

                            pu.save_type1_power_figure(
                                outdir=p.outdir,
                                gamma_grid=p.gamma_grid,
                                type1_table=type1_table,
                                power_table=power_table,
                                methods=focus_methods,
                                style_map=style_map,
                                alpha=p.alpha,
                                filename_stem=f"type1_power_focus_{outcome}_{scenario.key}_{stage_file}_nH{n_historical}",
                                figsize=(11.8, 8.9),
                            )

                            pu.save_lambda_figure(
                                outdir=p.outdir,
                                lambda_table=lambda_table,
                                hist_arms=scenario.hist_arms,
                                filename_stem=f"lambda_{outcome}_{scenario.key}_{stage_file}_nH{n_historical}",
                            )
    finally:
        progress.close()

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

    lambda_path = p.outdir / "bond_lambda.csv"
    with lambda_path.open("w", encoding="utf-8") as f:
        f.write("n_historical,outcome,scenario,stage,gamma,rho0,rho1,lambda0,lambda1,w0,w1,kappa\n")
        f.write("\n".join(lambda_rows))
        f.write("\n")

    print("Simulation completed")
    print(f"Output directory: {p.outdir}")
    print(f"Summary table: {results_path}")
    print(f"Worst-case table: {worst_path}")
    print(f"BOND lambda table: {lambda_path}")
    for warning in consistency_warnings:
        print(warning)


def main() -> None:
    """Main function

    Args:
        p (Params): The parameters
    """
    p = parse_args()
    validate_params(p)
    run_simulation(p)


if __name__ == "__main__":
    main()
