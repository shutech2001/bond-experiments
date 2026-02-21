from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.special import betaln, expit, logit, logsumexp  # type: ignore
from scipy.stats import norm  # type: ignore


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
    param: Optional[float] = None


def w_from_lambda(lam: float, n_c: int, n_h: int) -> float:
    """Calculate the weight from the lambda

    Args:
        lam (float): The lambda
        n_c (int): The number of control arm
        n_h (int): The number of historical arm

    Returns:
        float: The weight
    """
    if n_h <= 0 or lam <= 0.0:
        return 0.0
    return float((lam * n_h) / (n_c + lam * n_h))


def lambda_from_uip(m: float, n_h: int) -> float:
    """Calculate the lambda from the UIP

    Args:
        m (float): The m
        n_h (int): The number of historical arm

    Returns:
        float: The lambda
    """
    if n_h <= 0:
        return 0.0
    return float(np.clip(m / n_h, 0.0, 1.0))


def clamp_prob(p: float) -> float:
    """Clamp the probability

    Args:
        p (float): The probability

    Returns:
        float: The clamped probability
    """
    return max(1e-6, min(1.0 - 1e-6, p))


def var_mean_binary(mean: float, n: int) -> float:
    """Calculate the variance of the mean for binary outcome

    Args:
        mean (float): The mean
        n (int): The number of samples

    Returns:
        float: The variance of the mean
    """
    if n <= 0:
        return 0.0
    p = clamp_prob(mean)
    return p * (1.0 - p) / n


def var_mean_cont(var_y: float, n: int) -> float:
    """Calculate the variance of the mean for continuous outcome

    Args:
        var_y (float): The variance of the outcome
        n (int): The number of samples

    Returns:
        float: The variance of the mean
    """
    if n <= 1:
        return 0.0
    return var_y / n


def normalize_weights(weights: Optional[List[float]], n: int) -> NDArray[np.float64]:
    """Normalize the weights

    Args:
        weights (Optional[List[float]]): The weights
        n (int): The number of samples

    Returns:
        NDArray[np.float64]: The normalized weights
    """
    if weights is None:
        return np.full(n, 1.0 / n, dtype=float)
    arr = np.array(weights, dtype=float)
    total = float(np.sum(arr))
    if total <= 0.0:
        raise ValueError("Mixture weights must sum to a positive value.")
    return arr / total


def beta_binomial_log_marginal(success: int, failure: int, a: float, b: float) -> float:
    """Calculate the beta binomial log marginal

    Args:
        success (int): The success
        failure (int): The failure
        a (float): The a
        b (float): The b

    Returns:
        float: The beta binomial log marginal
    """
    return float(betaln(a + success, b + failure) - betaln(a, b))


def normal_logpdf(x: float, mean: float, var: float) -> float:
    """Calculate the normal log pdf

    Args:
        x (float): The value
        mean (float): The mean
        var (float): The variance

    Returns:
        float: The normal log pdf
    """
    var = max(var, 1e-12)
    return float(norm.logpdf(x, loc=mean, scale=math.sqrt(var)))


def estimate_sigma2_continuous(arm: Any) -> float:
    """Estimate the sigma2 for continuous outcome

    Args:
        arm (Any): The arm

    Returns:
        float: The sigma2
    """
    if arm.n_c > 1 and arm.n_h > 1:
        num = (arm.n_c - 1) * arm.var_c + (arm.n_h - 1) * arm.var_h
        den = arm.n_c + arm.n_h - 2
        return max(num / max(den, 1), 1e-8)
    if arm.n_c > 1:
        return max(arm.var_c, 1e-8)
    if arm.n_h > 1:
        return max(arm.var_h, 1e-8)
    return 1.0


def pooled_estimate(arm: Any, lam: float, outcome: str) -> PosteriorSummary:
    """Calculate the pooled estimate

    Args:
        arm (Any): The arm
        lam (float): The lambda
        outcome (str): The outcome

    Returns:
        PosteriorSummary: The posterior summary
    """
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


def posterior_power_prior(arm: Any, outcome: str, lam: float, p: Any) -> PosteriorSummary:
    """Calculate the posterior power prior

    Args:
        arm (Any): The arm
        outcome (str): The outcome
        lam (float): The lambda
        p (Any): The parameters

    Returns:
        PosteriorSummary: The posterior summary
    """
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


def posterior_commensurate(arm: Any, outcome: str, tau: float) -> PosteriorSummary:
    """Calculate the posterior commensurate

    Args:
        arm (Any): The arm
        outcome (str): The outcome
        tau (float): The tau

    Returns:
        PosteriorSummary: The posterior summary
    """
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


def posterior_map(arm: Any, outcome: str, p: Any) -> PosteriorSummary:
    """Calculate the posterior map

    Args:
        arm (Any): The arm
        outcome (str): The outcome
        p (Any): The parameters

    Returns:
        PosteriorSummary: The posterior summary
    """
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
            log_ml = float(betaln(a_post, b_post) - betaln(a_prior, b_prior))
            log_post_weights.append(float(np.log(float(base_weights[idx])) + log_ml))
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
            log_post_weights.append(float(np.log(float(base_weights[idx])) + log_ml))
            mean_components.append(float(post_mean))
            var_components.append(float(post_var))
            w_components.append(w_from_lambda(float(lam), arm.n_c, arm.n_h))

    log_weights = np.array(log_post_weights, dtype=float)
    post_weights = np.exp(log_weights - logsumexp(log_weights))
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


def posterior_rmap(arm: Any, outcome: str, p: Any, epsilon: float) -> PosteriorSummary:
    """Calculate the posterior robust map

    Args:
        arm (Any): The arm
        outcome (str): The outcome
        p (Any): The parameters
        epsilon (float): The epsilon

    Returns:
        PosteriorSummary: The posterior summary
    """
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
            log_ml = float(betaln(a_post, b_post) - betaln(a_prior, b_prior))
            log_post_weights.append(float(np.log(float(map_weights[idx])) + log_ml))
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
        log_ml_v = float(betaln(a_post_v, b_post_v) - betaln(a_v, b_v))
        log_post_weights.append(float(np.log(epsilon) + log_ml_v))
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
            log_post_weights.append(float(np.log(float(map_weights[idx])) + log_ml))
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
        log_post_weights.append(float(np.log(epsilon) + log_ml_v))
        mean_components.append(float(post_mean_v))
        var_components.append(float(post_var_v))
        lambda_components.append(0.0)
        w_components.append(0.0)

    log_weights = np.array(log_post_weights, dtype=float)
    post_weights = np.exp(log_weights - logsumexp(log_weights))
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


def posterior_elastic(arm: Any, outcome: str, scale: float) -> PosteriorSummary:
    """Calculate the posterior elastic

    Args:
        arm (Any): The arm
        outcome (str): The outcome
        scale (float): The scale

    Returns:
        PosteriorSummary: The posterior summary
    """
    if arm.n_h <= 0:
        return pooled_estimate(arm, 0.0, outcome)

    if outcome == "binary":
        se2 = var_mean_binary(arm.mean_c, arm.n_c) + var_mean_binary(arm.mean_h, arm.n_h)
    else:
        se2 = var_mean_cont(arm.var_c, arm.n_c) + var_mean_cont(arm.var_h, arm.n_h)
    if se2 <= 0.0:
        return pooled_estimate(arm, 0.0, outcome)

    z = (arm.mean_c - arm.mean_h) / np.sqrt(se2)
    g = float(np.exp(-0.5 * scale * (float(z) ** 2)))
    lam_eff = float(np.clip(g, 0.0, 1.0))
    return pooled_estimate(arm, lam_eff, outcome)


def posterior_uip(arm: Any, outcome: str, m_units: float) -> PosteriorSummary:
    """Calculate the posterior unit-info prior

    Args:
        arm (Any): The arm
        outcome (str): The outcome
        m_units (float): The m units

    Returns:
        PosteriorSummary: The posterior summary
    """
    if arm.n_h <= 0:
        return pooled_estimate(arm, 0.0, outcome)
    lam_eff = lambda_from_uip(m_units, arm.n_h)
    return pooled_estimate(arm, lam_eff, outcome)


def posterior_leap(arm: Any, outcome: str, p: Any) -> PosteriorSummary:
    """Calculate the posterior leap

    Args:
        arm (Any): The arm
        outcome (str): The outcome
        p (Any): The parameters

    Returns:
        PosteriorSummary: The posterior summary
    """
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
    post_logit = float(logit(p.leap_prior_omega)) + log_bf_ex
    lam_eff = float(np.clip(float(expit(post_logit)), 0.0, 1.0))
    return pooled_estimate(arm, lam_eff, outcome)


def posterior_exnex(arm: Any, outcome: str, p: Any) -> PosteriorSummary:
    """Calculate the posterior exnex

    Args:
        arm (Any): The arm
        outcome (str): The outcome
        p (Any): The parameters

    Returns:
        PosteriorSummary: The posterior summary
    """
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

    log_w_ex = float(np.log(omega_ex)) + log_ml_ex
    log_w_nex = float(np.log(1.0 - omega_ex)) + log_ml_nex
    log_ws = np.array([log_w_ex, log_w_nex], dtype=float)
    if not np.isfinite(log_ws).any():
        weights = np.array([0.5, 0.5], dtype=float)
    else:
        weights = np.exp(log_ws - logsumexp(log_ws))

    mean = float(weights[0] * post_ex.mean + weights[1] * post_nex.mean)
    var = float(weights[0] * (post_ex.var + post_ex.mean**2) + weights[1] * (post_nex.var + post_nex.mean**2) - mean**2)
    lambda_eff = float(weights[0] * post_ex.lambda_eff + weights[1] * post_nex.lambda_eff)
    w_eff = float(weights[0] * post_ex.w_eff + weights[1] * post_nex.w_eff)
    return PosteriorSummary(mean=mean, var=max(var, 1e-12), lambda_eff=lambda_eff, w_eff=w_eff)


def posterior_mem(arm: Any, outcome: str, p: Any) -> PosteriorSummary:
    """Calculate the posterior mem

    Args:
        arm (Any): The arm
        outcome (str): The outcome
        p (Any): The parameters

    Returns:
        PosteriorSummary: The posterior summary
    """
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

    log_w_in = float(np.log(pi_in)) + log_ml_in
    log_w_out = float(np.log(1.0 - pi_in)) + log_ml_out
    log_ws = np.array([log_w_in, log_w_out], dtype=float)
    if not np.isfinite(log_ws).any():
        weights = np.array([0.5, 0.5], dtype=float)
    else:
        weights = np.exp(log_ws - logsumexp(log_ws))

    mean = float(weights[0] * post_in.mean + weights[1] * post_out.mean)
    var = float(weights[0] * (post_in.var + post_in.mean**2) + weights[1] * (post_out.var + post_out.mean**2) - mean**2)
    lambda_eff = float(weights[0] * post_in.lambda_eff + weights[1] * post_out.lambda_eff)
    w_eff = float(weights[0] * post_in.w_eff + weights[1] * post_out.w_eff)
    return PosteriorSummary(mean=mean, var=max(var, 1e-12), lambda_eff=lambda_eff, w_eff=w_eff)


def posterior_bhmoi(arm: Any, outcome: str, p: Any) -> PosteriorSummary:
    """Calculate the posterior bhmoi

    Args:
        arm (Any): The arm
        outcome (str): The outcome
        p (Any): The parameters

    Returns:
        PosteriorSummary: The posterior summary
    """
    if arm.n_h <= 0:
        return pooled_estimate(arm, 0.0, outcome)

    if outcome == "continuous":
        sd_c = math.sqrt(max(arm.var_c, 1e-12))
        sd_h = math.sqrt(max(arm.var_h, 1e-12))
        denom = max(sd_c**2 + sd_h**2, 1e-12)
        oci = math.sqrt(max(2.0 * sd_c * sd_h / denom, 0.0))
        oci *= math.exp(-((arm.mean_c - arm.mean_h) ** 2) / (4.0 * denom))
        oci = float(np.clip(oci, 0.0, 1.0))
        obi = math.exp(-abs(math.log((sd_c + 1e-8) / (sd_h + 1e-8))))
    else:
        p_c = clamp_prob(arm.mean_c)
        p_h = clamp_prob(arm.mean_h)
        oci = math.sqrt(p_c * p_h) + math.sqrt((1.0 - p_c) * (1.0 - p_h))
        oci = float(np.clip(oci, 0.0, 1.0))
        v_c = p_c * (1.0 - p_c)
        v_h = p_h * (1.0 - p_h)
        obi = 2.0 * min(v_c, v_h) / max(v_c + v_h, 1e-8)
        obi = float(np.clip(obi, 0.0, 1.0))

    overlap = float(np.clip(math.sqrt(max(oci * obi, 0.0)), 0.0, 1.0))
    lam_eff = float(np.clip(overlap**p.bhmoi_sharpness, 0.0, 1.0))
    return pooled_estimate(arm, lam_eff, outcome)


def posterior_nonpara_bayes(arm: Any, outcome: str, p: Any) -> PosteriorSummary:
    """Calculate the posterior nonpara bayes

    Args:
        arm (Any): The arm
        outcome (str): The outcome
        p (Any): The parameters

    Returns:
        PosteriorSummary: The posterior summary
    """
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
    sbi = float(np.clip(sbi, 0.0, 1.0))
    lam_eff = float(np.clip(sbi**p.npb_sharpness, 0.0, 1.0))
    return pooled_estimate(arm, lam_eff, outcome)


def bond_shift_bounds(outcome: str, rho: float, mu_c: float) -> tuple[float, float]:
    """Calculate the bond shift bounds

    Args:
        outcome (str): The outcome
        rho (float): The rho
        mu_c (float): The mu c

    Returns:
        tuple[float, float]: The bond shift bounds
    """
    if outcome == "continuous":
        return float(rho), float(-rho)
    mu = clamp_prob(mu_c)
    delta_plus = min(float(rho), 1.0 - mu)
    delta_minus = -min(float(rho), mu)
    return float(delta_plus), float(delta_minus)


def bond_bias_plus(
    outcome: str,
    w0: float,
    w1: float,
    rho0: float,
    rho1: float,
    mu_c0: float,
    mu_c1: float,
) -> float:
    """Calculate the bond bias plus

    Args:
        outcome (str): The outcome
        w0 (float): The w0
        w1 (float): The w1
        rho0 (float): The rho0
        rho1 (float): The rho1
        mu_c0 (float): The mu c0
        mu_c1 (float): The mu c1

    Returns:
        float: The bond bias plus
    """
    delta1_plus, _ = bond_shift_bounds(outcome, rho1, mu_c1)
    _, delta0_minus = bond_shift_bounds(outcome, rho0, mu_c0)
    return float(w1 * delta1_plus - w0 * delta0_minus)


def bond_kappa(
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
    """Calculate the bond kappa

    Args:
        outcome (str): The outcome
        theta1 (float): The theta1
        w0 (float): The w0
        w1 (float): The w1
        rho0 (float): The rho0
        rho1 (float): The rho1
        mu_c0 (float): The mu c0
        mu_c1 (float): The mu c1
        s (float): The s

    Returns:
        float: The bond kappa
    """
    delta1_plus, delta1_minus = bond_shift_bounds(outcome, rho1, mu_c1)
    delta0_plus, delta0_minus = bond_shift_bounds(outcome, rho0, mu_c0)
    numerator = theta1 - w1 * (delta1_plus - delta1_minus) - w0 * (delta0_plus - delta0_minus)
    return float(numerator / s)


def bond_lambda_star(
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
) -> Tuple[float, float, float]:
    """Calculate the bond lambda star

    Args:
        outcome (str): The outcome
        rho0 (float): The rho0
        rho1 (float): The rho1
        theta1 (float): The theta1
        mu_c0 (float): The mu c0
        var_c0 (float): The var c0
        var_c1 (float): The var c1
        var_h0 (float): The var h0
        var_h1 (float): The var h1
        n_c0 (int): The n c0
        n_c1 (int): The n c1
        n_h0 (int): The n h0
        n_h1 (int): The n h1
        grid (int): The grid

    Returns:
        Tuple[float, float, float]: The bond lambda star
    """
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
            kappa = bond_kappa(outcome, theta1, w0, w1, rho0, rho1, mu_c0, mu_c1, s)
            if kappa > best_kappa:
                best_kappa = kappa
                best_lam0 = float(lam0)
                best_lam1 = float(lam1)
    return best_lam0, best_lam1, float(best_kappa)


def build_method_specs(p: Any) -> List[MethodSpec]:
    """Build the method specs

    Args:
        p (Any): The parameters

    Returns:
        List[MethodSpec]: The method specs
    """
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
    specs.append(MethodSpec(name="BOND", family="bond"))
    return specs


def method_current(
    arm: Any,
    outcome: str,
    p: Any,
    param: Optional[float] = None,
    arm_index: int = 0,
    bond_lambda0: float = 0.0,
    bond_lambda1: float = 0.0,
) -> PosteriorSummary:
    return pooled_estimate(arm, 0.0, outcome)


def method_naive_pool(
    arm: Any,
    outcome: str,
    p: Any,
    param: Optional[float] = None,
    arm_index: int = 0,
    bond_lambda0: float = 0.0,
    bond_lambda1: float = 0.0,
) -> PosteriorSummary:
    return pooled_estimate(arm, 1.0, outcome)


def method_fixed(
    arm: Any,
    outcome: str,
    p: Any,
    param: Optional[float] = None,
    arm_index: int = 0,
    bond_lambda0: float = 0.0,
    bond_lambda1: float = 0.0,
) -> PosteriorSummary:
    if param is None:
        raise ValueError("Fixed borrowing requires param.")
    return pooled_estimate(arm, float(param), outcome)


def method_power_prior(
    arm: Any,
    outcome: str,
    p: Any,
    param: Optional[float] = None,
    arm_index: int = 0,
    bond_lambda0: float = 0.0,
    bond_lambda1: float = 0.0,
) -> PosteriorSummary:
    if param is None:
        raise ValueError("Power prior requires param.")
    return posterior_power_prior(arm, outcome, float(param), p)


def method_commensurate_prior(
    arm: Any,
    outcome: str,
    p: Any,
    param: Optional[float] = None,
    arm_index: int = 0,
    bond_lambda0: float = 0.0,
    bond_lambda1: float = 0.0,
) -> PosteriorSummary:
    if param is None:
        raise ValueError("Commensurate prior requires param.")
    return posterior_commensurate(arm, outcome, float(param))


def method_map_prior(
    arm: Any,
    outcome: str,
    p: Any,
    param: Optional[float] = None,
    arm_index: int = 0,
    bond_lambda0: float = 0.0,
    bond_lambda1: float = 0.0,
) -> PosteriorSummary:
    """Calculate the posterior map

    Args:
        arm (Any): The arm
        outcome (str): The outcome
        p (Any): The parameters
        param (Optional[float]): The parameter
        arm_index (int): The arm index
        bond_lambda0 (float): The bond lambda0
        bond_lambda1 (float): The bond lambda1

    Returns:
        PosteriorSummary: The posterior summary
    """
    return posterior_map(arm, outcome, p)


def method_robust_map(
    arm: Any,
    outcome: str,
    p: Any,
    param: Optional[float] = None,
    arm_index: int = 0,
    bond_lambda0: float = 0.0,
    bond_lambda1: float = 0.0,
) -> PosteriorSummary:
    """Calculate the posterior robust map

    Args:
        arm (Any): The arm
        outcome (str): The outcome
        p (Any): The parameters
        param (Optional[float]): The parameter
        arm_index (int): The arm index
        bond_lambda0 (float): The bond lambda0
        bond_lambda1 (float): The bond lambda1

    Returns:
        PosteriorSummary: The posterior summary
    """
    if param is None:
        raise ValueError("Robust MAP requires param.")
    return posterior_rmap(arm, outcome, p, float(param))


def method_elastic_prior(
    arm: Any,
    outcome: str,
    p: Any,
    param: Optional[float] = None,
    arm_index: int = 0,
    bond_lambda0: float = 0.0,
    bond_lambda1: float = 0.0,
) -> PosteriorSummary:
    if param is None:
        raise ValueError("Elastic prior requires param.")
    return posterior_elastic(arm, outcome, float(param))


def method_unit_info_prior(
    arm: Any,
    outcome: str,
    p: Any,
    param: Optional[float] = None,
    arm_index: int = 0,
    bond_lambda0: float = 0.0,
    bond_lambda1: float = 0.0,
) -> PosteriorSummary:
    if param is None:
        raise ValueError("Unit-information prior requires param.")
    return posterior_uip(arm, outcome, float(param))


def method_leap(
    arm: Any,
    outcome: str,
    p: Any,
    param: Optional[float] = None,
    arm_index: int = 0,
    bond_lambda0: float = 0.0,
    bond_lambda1: float = 0.0,
) -> PosteriorSummary:
    return posterior_leap(arm, outcome, p)


def method_exnex(
    arm: Any,
    outcome: str,
    p: Any,
    param: Optional[float] = None,
    arm_index: int = 0,
    bond_lambda0: float = 0.0,
    bond_lambda1: float = 0.0,
) -> PosteriorSummary:
    return posterior_exnex(arm, outcome, p)


def method_mem(
    arm: Any,
    outcome: str,
    p: Any,
    param: Optional[float] = None,
    arm_index: int = 0,
    bond_lambda0: float = 0.0,
    bond_lambda1: float = 0.0,
) -> PosteriorSummary:
    return posterior_mem(arm, outcome, p)


def method_bhmoi(
    arm: Any,
    outcome: str,
    p: Any,
    param: Optional[float] = None,
    arm_index: int = 0,
    bond_lambda0: float = 0.0,
    bond_lambda1: float = 0.0,
) -> PosteriorSummary:
    return posterior_bhmoi(arm, outcome, p)


def method_nonparametric_bayes(
    arm: Any,
    outcome: str,
    p: Any,
    param: Optional[float] = None,
    arm_index: int = 0,
    bond_lambda0: float = 0.0,
    bond_lambda1: float = 0.0,
) -> PosteriorSummary:
    return posterior_nonpara_bayes(arm, outcome, p)


def method_test_then_pool(
    arm: Any,
    outcome: str,
    p: Any,
    param: Optional[float] = None,
    arm_index: int = 0,
    bond_lambda0: float = 0.0,
    bond_lambda1: float = 0.0,
) -> PosteriorSummary:
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


def method_bond(
    arm: Any,
    outcome: str,
    p: Any,
    param: Optional[float] = None,
    arm_index: int = 0,
    bond_lambda0: float = 0.0,
    bond_lambda1: float = 0.0,
) -> PosteriorSummary:
    lam = bond_lambda0 if arm_index == 0 else bond_lambda1
    return pooled_estimate(arm, lam, outcome)


MethodRunner = Callable[[Any, str, Any, Optional[float], int, float, float], PosteriorSummary]

METHOD_RUNNERS: Dict[str, MethodRunner] = {
    "current": method_current,
    "naive": method_naive_pool,
    "fixed": method_fixed,
    "power_prior": method_power_prior,
    "commensurate": method_commensurate_prior,
    "map": method_map_prior,
    "robust_map": method_robust_map,
    "elastic": method_elastic_prior,
    "uip": method_unit_info_prior,
    "leap": method_leap,
    "exnex": method_exnex,
    "mem": method_mem,
    "bhmoi": method_bhmoi,
    "npb": method_nonparametric_bayes,
    "test_then_pool": method_test_then_pool,
    "bond": method_bond,
}


def estimate_arm_for_method(
    method: MethodSpec,
    arm: Any,
    outcome: str,
    p: Any,
    arm_index: int,
    bond_lambda0: float,
    bond_lambda1: float,
) -> PosteriorSummary:
    runner = METHOD_RUNNERS.get(method.family)
    if runner is None:
        raise ValueError(f"Unknown method family: {method.family}")
    return runner(
        arm=arm,
        outcome=outcome,
        p=p,
        param=method.param,
        arm_index=arm_index,
        bond_lambda0=bond_lambda0,
        bond_lambda1=bond_lambda1,
    )
