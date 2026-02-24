from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.special import betaln, expit, gammaln, logit, logsumexp  # type: ignore
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


def log_choose(n: int, k: int) -> float:
    """Compute log(n choose k) stably.

    Args:
        n (int): Total items.
        k (int): Chosen items.

    Returns:
        float: log binomial coefficient, or -inf if invalid.
    """
    if k < 0 or k > n:
        return float("-inf")
    return float(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1))


def normal_overlap_coefficient(mu1: float, sd1: float, mu2: float, sd2: float) -> float:
    """Calculate the overlap coefficient between two normal densities.

    Args:
        mu1 (float): Mean of the first normal distribution.
        sd1 (float): Standard deviation of the first normal distribution.
        mu2 (float): Mean of the second normal distribution.
        sd2 (float): Standard deviation of the second normal distribution.

    Returns:
        float: Overlap coefficient in [0,1].
    """
    sd1 = max(float(sd1), 1e-12)
    sd2 = max(float(sd2), 1e-12)

    # Equal-variance closed form.
    if abs(sd1 - sd2) <= 1e-12:
        z = abs(mu1 - mu2) / (2.0 * sd1)
        return float(np.clip(2.0 * norm.cdf(-z), 0.0, 1.0))

    # Unequal variances: integrate the smaller density piecewise between intersections.
    a = 1.0 / (2.0 * sd2**2) - 1.0 / (2.0 * sd1**2)
    b = mu1 / (sd1**2) - mu2 / (sd2**2)
    c = mu2**2 / (2.0 * sd2**2) - mu1**2 / (2.0 * sd1**2) - math.log(sd2 / sd1)

    disc = b * b - 4.0 * a * c
    if disc <= 0.0:
        roots = [float(-b / (2.0 * a))]
    else:
        sdisc = math.sqrt(disc)
        r1 = float((-b - sdisc) / (2.0 * a))
        r2 = float((-b + sdisc) / (2.0 * a))
        roots = sorted([r1, r2])

    bounds = [-np.inf] + roots + [np.inf]
    ovl = 0.0
    spread = max(sd1, sd2)
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if np.isneginf(lo):
            mid = hi - 10.0 * spread
        elif np.isposinf(hi):
            mid = lo + 10.0 * spread
        else:
            mid = 0.5 * (lo + hi)

        if norm.pdf(mid, loc=mu1, scale=sd1) <= norm.pdf(mid, loc=mu2, scale=sd2):
            cdf_hi = float(norm.cdf((hi - mu1) / sd1))
            cdf_lo = float(norm.cdf((lo - mu1) / sd1))
        else:
            cdf_hi = float(norm.cdf((hi - mu2) / sd2))
            cdf_lo = float(norm.cdf((lo - mu2) / sd2))
        ovl += cdf_hi - cdf_lo
    return float(np.clip(ovl, 0.0, 1.0))


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
    References:
        Ibrahim, J. G. and Chen, M.-H. (2000). Power prior distributions for regression models.
            Statistical Science, 15(1): 46-50.

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
        # Basic fixed-a0 power prior:
        #   pi(theta | D0, a0) propto L(theta | D0)^a0 * pi0(theta)
        # Posterior:
        #   pi(theta | D, D0, a0) propto L(theta | D) * L(theta | D0)^a0 * pi0(theta)
        # For Binomial-Beta, this is equivalent to discounting historical counts by a0.
        s_c = int(arm.succ_c)
        f_c = int(max(arm.n_c - arm.succ_c, 0))
        s_h = int(arm.succ_h)
        f_h = int(max(arm.n_h - arm.succ_h, 0))
        a_post = float(p.beta_prior_alpha) + s_c + lam * s_h
        b_post = float(p.beta_prior_beta) + f_c + lam * f_h
        total = a_post + b_post
        mean = a_post / total
        var = (a_post * b_post) / (total**2 * (total + 1.0))
        return PosteriorSummary(
            mean=float(mean),
            var=max(float(var), 1e-12),
            lambda_eff=lam,
            w_eff=w_from_lambda(lam, arm.n_c, arm.n_h),
        )

    # Normal summary-likelihood approximation:
    # ybar_c | theta ~ N(theta, var_c / n_c), ybar_h | theta ~ N(theta, var_h / n_h).
    # Raising historical likelihood to a0 scales historical precision by a0.
    n_c = int(max(arm.n_c, 1))
    n_h = int(max(arm.n_h, 1))
    like_var_c = float(max(var_mean_cont(float(arm.var_c), n_c), 1e-12))
    like_var_h = float(max(var_mean_cont(float(arm.var_h), n_h), 1e-12))
    prec0 = 1.0 / float(max(p.normal_prior_var, 1e-12))
    like_prec_c = 1.0 / like_var_c
    like_prec_h = lam / like_var_h
    post_prec = prec0 + like_prec_c + like_prec_h
    post_mean = (
        prec0 * float(p.normal_prior_mean) + like_prec_c * float(arm.mean_c) + like_prec_h * float(arm.mean_h)
    ) / max(post_prec, 1e-12)
    post_var = 1.0 / post_prec
    return PosteriorSummary(
        mean=float(post_mean),
        var=max(float(post_var), 1e-12),
        lambda_eff=lam,
        w_eff=w_from_lambda(lam, arm.n_c, arm.n_h),
    )


def posterior_commensurate(arm: Any, outcome: str, tau: float) -> PosteriorSummary:
    """Calculate the posterior commensurate
    References:
        Hobbs, B.P., Sargent, D.J., and Carlin, B.P. (2012). Commensurate priors for incorporating
            historical information in clinical trials using general and generalized linear models.
            Bayesian Analysis, (3):639-674.

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

    # One-parameter commensurate-prior reduction:
    # current parameter theta has prior theta | theta_0, tau ~ N(theta_0, 1/tau),
    # and theta_0 is learned from historical data.
    # Integrating theta_0 yields:
    #   theta ~ N(theta_hat_h, V_h + 1/tau),
    # where V_h is the (approx.) variance of the historical estimator.
    if outcome == "continuous":
        n_c = int(max(arm.n_c, 1))
        n_h = int(max(arm.n_h, 1))

        like_var = float(max(var_mean_cont(float(arm.var_c), n_c), 1e-12))
        hist_var = float(max(var_mean_cont(float(arm.var_h), n_h), 1e-12))
        prior_var = float(max(hist_var + 1.0 / tau, 1e-12))

        like_prec = 1.0 / like_var
        prior_prec = 1.0 / prior_var
        post_prec = like_prec + prior_prec
        post_var = 1.0 / max(post_prec, 1e-12)
        post_mean = post_var * (like_prec * float(arm.mean_c) + prior_prec * float(arm.mean_h))

        # Historical-information retention under commensurability discounting.
        # Full historical precision: 1/hist_var; discounted: 1/(hist_var + 1/tau).
        lam_eff = float(np.clip(hist_var / max(hist_var + 1.0 / tau, 1e-12), 0.0, 1.0))
    else:
        n_c = int(max(arm.n_c, 1))
        n_h = int(max(arm.n_h, 1))

        # GLM normal approximation on logit scale:
        # theta_hat ~ N(theta, V), V ~= 1 / (n p (1-p)).
        p_c = clamp_prob((float(arm.succ_c) + 0.5) / (float(n_c) + 1.0))
        p_h = clamp_prob((float(arm.succ_h) + 0.5) / (float(n_h) + 1.0))
        theta_c = float(logit(p_c))
        theta_h = float(logit(p_h))

        like_var_theta = float(max(1.0 / (float(n_c) * p_c * (1.0 - p_c)), 1e-12))
        hist_var_theta = float(max(1.0 / (float(n_h) * p_h * (1.0 - p_h)), 1e-12))
        prior_var_theta = float(max(hist_var_theta + 1.0 / tau, 1e-12))

        like_prec = 1.0 / like_var_theta
        prior_prec = 1.0 / prior_var_theta
        post_prec = like_prec + prior_prec
        post_var_theta = 1.0 / max(post_prec, 1e-12)
        post_mean_theta = post_var_theta * (like_prec * theta_c + prior_prec * theta_h)

        # Back-transform to probability scale (delta-method variance).
        post_mean = clamp_prob(float(expit(post_mean_theta)))
        grad = post_mean * (1.0 - post_mean)
        post_var = float(max(post_var_theta * grad * grad, 1e-12))

        lam_eff = float(np.clip(hist_var_theta / max(hist_var_theta + 1.0 / tau, 1e-12), 0.0, 1.0))

    return PosteriorSummary(
        mean=float(post_mean),
        var=max(float(post_var), 1e-12),
        lambda_eff=float(lam_eff),
        w_eff=w_from_lambda(float(lam_eff), arm.n_c, arm.n_h),
    )


def posterior_rmap(arm: Any, outcome: str, p: Any, epsilon: float) -> PosteriorSummary:
    """Calculate the posterior robust map
    References:
        Schmidli, H., Gsteiger, S., Royschoudhury, S., O'Hagan, A., Spiegelhalter, D., and Neuenschwander, B. (2014).
            Robust meta-analytic-predictive priors in clinical trials with historical control information. Biometrics,
            70(4):1023-1032.

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

    # Robust MAP prior (Schmidli et al.):
    #   p_HR = (1-epsilon) * p_H + epsilon * p_V,
    # where p_H is a MAP mixture approximated by conjugate components and p_V is vague.
    lambdas = np.array(p.map_lambdas, dtype=float)
    base_weights = normalize_weights(p.map_weights, lambdas.size)
    map_weights = (1.0 - epsilon) * base_weights

    if outcome == "binary":
        s_c = int(arm.succ_c)
        f_c = int(arm.n_c - arm.succ_c)
        n_h = int(max(arm.n_h, 1))
        p_h = clamp_prob(float(arm.mean_h))

        mean_components: list[float] = []
        var_components: list[float] = []
        log_post_weights: list[float] = []
        lambda_components: list[float] = []
        w_components: list[float] = []

        a0 = float(max(p.beta_prior_alpha, 1e-8))
        b0 = float(max(p.beta_prior_beta, 1e-8))

        # Informative MAP mixture components.
        for idx, lam in enumerate(lambdas):
            n0_k = float(max(lam, 1e-8) * n_h)
            a_prior = a0 + n0_k * p_h
            b_prior = b0 + n0_k * (1.0 - p_h)
            a_post = a_prior + s_c
            b_post = b_prior + f_c
            total = a_post + b_post
            mean_k = a_post / total
            var_k = (a_post * b_post) / (total**2 * (total + 1.0))
            # f_k in the paper (Table 1): Beta-binomial marginal probability.
            log_ml = beta_binomial_log_marginal(s_c, f_c, a_prior, b_prior)
            log_post_weights.append(float(np.log(float(map_weights[idx])) + log_ml))
            mean_components.append(float(mean_k))
            var_components.append(float(var_k))
            lambda_k = float(np.clip(n0_k / n_h, 0.0, 1.0))
            lambda_components.append(lambda_k)
            w_components.append(w_from_lambda(lambda_k, arm.n_c, arm.n_h))

        # Vague robustification component.
        a_v = p.vague_alpha
        b_v = p.vague_beta
        a_post_v = a_v + s_c
        b_post_v = b_v + f_c
        total_v = a_post_v + b_post_v
        mean_v = a_post_v / total_v
        var_v = (a_post_v * b_post_v) / (total_v**2 * (total_v + 1.0))
        log_ml_v = beta_binomial_log_marginal(s_c, f_c, a_v, b_v)
        log_post_weights.append(float(np.log(epsilon) + log_ml_v))
        mean_components.append(float(mean_v))
        var_components.append(float(var_v))
        lambda_components.append(0.0)
        w_components.append(0.0)
    else:
        n_c = int(max(arm.n_c, 1))
        n_h = int(max(arm.n_h, 1))
        like_var = float(max(arm.var_c / n_c, 1e-12))
        hist_var = float(max(arm.var_h, 1e-12))
        hist_mean = float(arm.mean_h)

        mean_components = []
        var_components = []
        log_post_weights = []
        lambda_components = []
        w_components = []

        # Informative MAP mixture components.
        for idx, lam in enumerate(lambdas):
            n0_k = float(max(lam, 1e-8) * n_h)
            prior_mean = hist_mean
            prior_var = float(max(hist_var / n0_k, 1e-12))
            prior_prec = 1.0 / max(prior_var, 1e-12)
            like_prec = 1.0 / like_var
            post_var = 1.0 / (like_prec + prior_prec)
            post_mean = post_var * (like_prec * arm.mean_c + prior_prec * prior_mean)
            # f_k in Table 1 for Normal prior/likelihood.
            log_ml = normal_logpdf(arm.mean_c, prior_mean, like_var + prior_var)
            log_post_weights.append(float(np.log(float(map_weights[idx])) + log_ml))
            mean_components.append(float(post_mean))
            var_components.append(float(post_var))
            lambda_k = float(np.clip(n0_k / n_h, 0.0, 1.0))
            lambda_components.append(lambda_k)
            w_components.append(w_from_lambda(lambda_k, arm.n_c, arm.n_h))

        # Vague robustification component.
        prior_mean_v = p.normal_prior_mean
        prior_var_v = float(max(p.vague_normal_var, 1e-12))
        prior_prec_v = 1.0 / max(prior_var_v, 1e-12)
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


def posterior_elastic(arm: Any, outcome: str, p: Any, scale: float) -> PosteriorSummary:
    """Calculate the posterior elastic
    References:
        Jiang, L., Nie, L., and Yuan, Y. (2023). Elastic priors to dynamically borrow information from
            historical data in clinical trials. Biometrics, 79(1):49-60.

    Args:
        arm (Any): The arm
        outcome (str): The outcome
        p (Any): The parameters
        scale (float): The scale

    Returns:
        PosteriorSummary: The posterior summary
    """
    if arm.n_h <= 0:
        return pooled_estimate(arm, 0.0, outcome)

    # Elastic function in Jiang et al.:
    # g(T) = 1 / (1 + exp(a + b * log(T))),  b > 0.
    # Keep compatibility with current interface by using:
    # - scale as b (steepness)
    # - optional p.elastic_logit_a as a (default 0.0)
    a = float(getattr(p, "elastic_logit_a", 0.0))
    b = float(max(scale, 1e-8))

    def logistic_elastic(t_stat: float) -> float:
        t_stat = float(max(t_stat, 1e-12))
        z = a + b * math.log(t_stat)
        if z >= 40.0:
            return 0.0
        if z <= -40.0:
            return 1.0
        return float(1.0 / (1.0 + math.exp(z)))

    if outcome == "binary":
        s_c = int(arm.succ_c)
        f_c = int(max(arm.n_c - arm.succ_c, 0))
        s_h = int(arm.succ_h)
        f_h = int(max(arm.n_h - arm.succ_h, 0))
        n_c = int(max(arm.n_c, 1))
        n_h = int(max(arm.n_h, 1))
        n_tot = n_c + n_h

        # Congruence measure T: Pearson chi-square statistic for 2x2 table.
        s_tot = s_c + s_h
        f_tot = f_c + f_h
        e_sc = n_c * s_tot / max(n_tot, 1)
        e_fc = n_c * f_tot / max(n_tot, 1)
        e_sh = n_h * s_tot / max(n_tot, 1)
        e_fh = n_h * f_tot / max(n_tot, 1)
        t_stat = 0.0
        for obs, exp in ((s_c, e_sc), (f_c, e_fc), (s_h, e_sh), (f_h, e_fh)):
            if exp <= 1e-12:
                if obs > 1e-12:
                    t_stat += 1e6
            else:
                t_stat += ((obs - exp) ** 2) / exp

        g = logistic_elastic(t_stat)
        lam_eff = float(np.clip(g, 0.0, 1.0))

        # Elastic prior: Beta((a0+s_h)g, (b0+f_h)g)
        # Posterior with current data is analytic.
        a0 = float(max(getattr(p, "beta_prior_alpha", 0.1), 1e-8))
        b0 = float(max(getattr(p, "beta_prior_beta", 0.1), 1e-8))
        a_prior = max((a0 + s_h) * g, 1e-8)
        b_prior = max((b0 + f_h) * g, 1e-8)
        a_post = a_prior + s_c
        b_post = b_prior + f_c
        total = a_post + b_post
        mean = a_post / total
        var = (a_post * b_post) / (total**2 * (total + 1.0))
        return PosteriorSummary(
            mean=float(mean),
            var=max(float(var), 1e-12),
            lambda_eff=lam_eff,
            w_eff=w_from_lambda(lam_eff, arm.n_c, arm.n_h),
        )

    # Continuous endpoint:
    n_c = int(max(arm.n_c, 1))
    n_h = int(max(arm.n_h, 1))
    mu_c = float(arm.mean_c)
    mu_h = float(arm.mean_h)
    s2_c = float(max(arm.var_c, 1e-12))
    s2_h = float(max(arm.var_h, 1e-12))

    # Congruence measure T: two-sample t statistic.
    if n_c > 1 and n_h > 1:
        s2_pool = ((n_c - 1) * s2_c + (n_h - 1) * s2_h) / max(n_c + n_h - 2, 1)
        se = math.sqrt(max(s2_pool * (1.0 / n_c + 1.0 / n_h), 1e-12))
    else:
        se = math.sqrt(max(s2_c / n_c + s2_h / n_h, 1e-12))
    diff = abs(mu_c - mu_h)
    t_stat = diff / max(se, 1e-12)

    g = logistic_elastic(t_stat)
    lam_eff = float(np.clip(g, 0.0, 1.0))

    # Elastic prior: N(mean_h, sigma_h^2 / (n_h g(T))) and normal-normal update.
    like_var = max(s2_c / n_c, 1e-12)
    like_prec = 1.0 / like_var
    prior_prec = max(g * n_h / max(s2_h, 1e-12), 0.0)
    post_prec = like_prec + prior_prec
    post_var = 1.0 / max(post_prec, 1e-12)
    post_mean = post_var * (like_prec * mu_c + prior_prec * mu_h)
    return PosteriorSummary(
        mean=float(post_mean),
        var=max(float(post_var), 1e-12),
        lambda_eff=lam_eff,
        w_eff=w_from_lambda(lam_eff, arm.n_c, arm.n_h),
    )


def posterior_uip(arm: Any, outcome: str, m_units: float) -> PosteriorSummary:
    """Calculate the posterior unit-info prior
    References:
        Jin, H. and Yin, G. (2021). Unit information prior for adaptive information borrowing from multiple
            historical datasets. Statistics in Medicine, 40(25):5657-5672.

    Args:
        arm (Any): The arm
        outcome (str): The outcome
        m_units (float): The m units

    Returns:
        PosteriorSummary: The posterior summary
    """
    if arm.n_h <= 0:
        return pooled_estimate(arm, 0.0, outcome)

    # UIP with a single historical dataset (K=1 reduction of Jin & Yin, 2021):
    # - Prior mean equals historical estimator.
    # - Prior precision is parameterized by M * I_U(theta_h).
    m = float(max(m_units, 1e-8))

    if outcome == "continuous":
        n_c = int(max(arm.n_c, 1))
        mu_h = float(arm.mean_h)
        s2_h = float(max(arm.var_h, 1e-12))
        s2_c = float(max(arm.var_c, 1e-12))

        # Prior: theta ~ N(mu_h, s2_h / M)
        prior_prec = m / s2_h
        # Likelihood (summary level): mean_c | theta ~ N(theta, s2_c / n_c)
        like_prec = n_c / s2_c
        post_prec = prior_prec + like_prec
        post_var = 1.0 / max(post_prec, 1e-12)
        post_mean = post_var * (prior_prec * mu_h + like_prec * float(arm.mean_c))

        # Effective borrowing summaries for reporting.
        lam_eff = lambda_from_uip(m, arm.n_h)
        w_eff = float(np.clip(prior_prec / max(post_prec, 1e-12), 0.0, 1.0))
        return PosteriorSummary(
            mean=float(post_mean),
            var=max(float(post_var), 1e-12),
            lambda_eff=lam_eff,
            w_eff=w_eff,
        )

    # Binary endpoint:
    # UIP: E(theta)=mu_h, Var(theta)=mu_h(1-mu_h)/M.
    # Matching Beta(alpha,beta): alpha+beta = M-1.
    n_c = int(max(arm.n_c, 1))
    mu_h = clamp_prob(float(arm.mean_h))
    s_c = int(arm.succ_c)
    f_c = int(max(arm.n_c - arm.succ_c, 0))

    m_eff = max(m - 1.0, 0.0)
    alpha_prior = max(mu_h * m_eff, 1e-8)
    beta_prior = max((1.0 - mu_h) * m_eff, 1e-8)
    alpha_post = alpha_prior + s_c
    beta_post = beta_prior + f_c
    total = alpha_post + beta_post
    post_mean = alpha_post / total
    post_var = (alpha_post * beta_post) / (total**2 * (total + 1.0))

    lam_eff = lambda_from_uip(m_eff, arm.n_h)
    w_eff = float(m_eff / max(m_eff + n_c, 1e-12))
    return PosteriorSummary(
        mean=float(post_mean),
        var=max(float(post_var), 1e-12),
        lambda_eff=lam_eff,
        w_eff=w_eff,
    )


def posterior_leap(arm: Any, outcome: str, p: Any) -> PosteriorSummary:
    """Calculate the posterior leap
    References:
        Alt, E.M., Chang, X., Jiang, X., Liu. Q., Mo, M., Xia, H. A., and Ibrahim, J. G. (2024).
            LEAP: The latent exchangeability prior for borrowing information from historical data.
            Biometrics, 80(3):ujae083.

    Args:
        arm (Any): The arm
        outcome (str): The outcome
        p (Any): The parameters

    Returns:
        PosteriorSummary: The posterior summary
    """
    if arm.n_h <= 0:
        return pooled_estimate(arm, 0.0, outcome)

    # LEAP here is implemented as a K=2 latent-partition model:
    # component 1 = exchangeable with current data, component 2 = non-exchangeable.
    # This is the minimal non-degenerate LEAP setup; K=1 collapses to full pooling.
    n_c = int(max(arm.n_c, 1))
    n_h = int(max(arm.n_h, 1))
    n01_max = min(n_h, n_c)  # optional SSC cap to avoid historical dominance
    if n01_max <= 0:
        return pooled_estimate(arm, 0.0, outcome)

    omega = float(np.clip(p.leap_prior_omega, 1e-8, 1.0 - 1e-8))
    kappa = float(max(p.leap_nex_scale, 1e-6))
    alpha_g = omega * kappa
    beta_g = (1.0 - omega) * kappa

    # Binary: exact partition-averaged LEAP via sufficient statistics.
    # For Bernoulli likelihood with K=2 and Beta priors, partitions sharing
    # (s01, f01) are equivalent, so we can integrate exactly without enumerating 2^n_h.
    if outcome == "binary":
        s_c = int(arm.succ_c)
        f_c = int(max(arm.n_c - arm.succ_c, 0))
        s_h = int(arm.succ_h)
        f_h = int(max(arm.n_h - arm.succ_h, 0))
        a0 = float(max(p.beta_prior_alpha, 1e-8))
        b0 = float(max(p.beta_prior_beta, 1e-8))

        means: list[float] = []
        variances: list[float] = []
        lambdas: list[float] = []
        w_components: list[float] = []
        log_weights: list[float] = []

        for s01 in range(s_h + 1):
            # n01 = s01 + f01 must satisfy cap n01 <= n01_max.
            f01_hi = min(f_h, n01_max - s01)
            if f01_hi < 0:
                continue
            for f01 in range(f01_hi + 1):
                s02 = s_h - s01
                f02 = f_h - f01
                n01 = s01 + f01

                # Number of latent allocations yielding this (s01, f01) split.
                log_mult = log_choose(s_h, s01) + log_choose(f_h, f01)

                # Integrated gamma prior term.
                log_gamma = float(betaln(alpha_g + n01, beta_g + n_h - n01) - betaln(alpha_g, beta_g))

                # Marginal likelihood terms under Beta-Bernoulli for components 1 and 2.
                log_ml_1 = float(betaln(a0 + s_c + s01, b0 + f_c + f01) - betaln(a0, b0))
                log_ml_2 = float(betaln(a0 + s02, b0 + f02) - betaln(a0, b0))

                log_weights.append(log_mult + log_gamma + log_ml_1 + log_ml_2)

                a1_post = a0 + s_c + s01
                b1_post = b0 + f_c + f01
                tot = a1_post + b1_post
                mean_1 = float(a1_post / tot)
                var_1 = float((a1_post * b1_post) / (tot * tot * (tot + 1.0)))

                lam_n01 = float(n01 / max(n_h, 1))
                means.append(mean_1)
                variances.append(var_1)
                lambdas.append(lam_n01)
                w_components.append(float(w_from_lambda(lam_n01, n_c, n_h)))

        if len(log_weights) == 0:
            return pooled_estimate(arm, 0.0, outcome)

        logw = np.array(log_weights, dtype=float)
        if not np.isfinite(logw).any():
            weights = np.full(logw.shape, 1.0 / float(logw.size), dtype=float)
        else:
            weights = np.exp(logw - logsumexp(logw))

        means_arr = np.array(means, dtype=float)
        vars_arr = np.array(variances, dtype=float)
        lam_arr = np.array(lambdas, dtype=float)
        w_arr = np.array(w_components, dtype=float)

        post_mean = float(np.sum(weights * means_arr))
        post_var = float(np.sum(weights * (vars_arr + means_arr**2)) - post_mean**2)
        lambda_eff = float(np.sum(weights * lam_arr))
        w_eff = float(np.sum(weights * w_arr))
        return PosteriorSummary(mean=post_mean, var=max(post_var, 1e-12), lambda_eff=lambda_eff, w_eff=w_eff)

    # Continuous:
    # If historical individual outcomes are available, run latent-assignment MCMC
    # (individual-level C_{0i}) for LEAP K=2.
    y_h_raw = getattr(arm, "y_h", None)
    if y_h_raw is not None:
        y_h = np.asarray(y_h_raw, dtype=float).reshape(-1)
    else:
        y_h = np.array([], dtype=float)

    if y_h.size == n_h and n_h > 0:
        rng_seed = int(getattr(p, "seed", 12345))
        rng = np.random.default_rng(rng_seed + 99173 + n_h + n_c)

        sigma2 = float(max(estimate_sigma2_continuous(arm), 1e-8))
        mu0 = float(p.normal_prior_mean)
        v0 = float(max(p.normal_prior_var, 1e-10))
        prec0 = 1.0 / v0

        n_iter = int(max(getattr(p, "leap_mcmc_iters", 5000), 500))
        burn = int(max(getattr(p, "leap_mcmc_burnin", n_iter // 2), 0))
        thin = int(max(getattr(p, "leap_mcmc_thin", 2), 1))
        if burn >= n_iter:
            burn = n_iter // 2

        mu_c = float(arm.mean_c)
        dist_to_current = np.abs(y_h - mu_c)
        init_n1 = int(np.clip(round(omega * n_h), 0, n01_max))
        c = np.zeros(n_h, dtype=np.int8)
        if init_n1 > 0:
            idx = np.argsort(dist_to_current)[:init_n1]
            c[idx] = 1

        sum_y_h = float(np.sum(y_h))
        sum_c = float(getattr(arm, "sum_c", float(arm.mean_c) * n_c))
        n1 = int(np.sum(c))
        sum1 = float(np.sum(y_h[c == 1])) if n1 > 0 else 0.0

        theta1_samples: list[float] = []
        lambda_samples: list[float] = []

        theta1 = float(mu_c)
        theta2 = float(getattr(arm, "mean_h", mu_c))
        gamma1 = float(omega)

        for t in range(n_iter):
            # Sample parameters given latent assignments.
            n2 = n_h - n1
            sum2 = sum_y_h - sum1

            prec1 = prec0 + (n_c + n1) / sigma2
            var1 = 1.0 / max(prec1, 1e-12)
            mean1 = var1 * (prec0 * mu0 + (sum_c + sum1) / sigma2)
            theta1 = float(rng.normal(mean1, math.sqrt(max(var1, 1e-12))))

            prec2 = prec0 + n2 / sigma2
            var2 = 1.0 / max(prec2, 1e-12)
            mean2 = var2 * (prec0 * mu0 + sum2 / sigma2)
            theta2 = float(rng.normal(mean2, math.sqrt(max(var2, 1e-12))))

            gamma1 = float(rng.beta(alpha_g + n1, beta_g + n_h - n1))

            # Sample latent assignments C_{0i}.
            order = rng.permutation(n_h)
            for i in order:
                yi = float(y_h[i])
                old = int(c[i])
                if old == 1:
                    n1_without = n1 - 1
                    sum1_without = sum1 - yi
                else:
                    n1_without = n1
                    sum1_without = sum1

                allow_ex = n1_without + 1 <= n01_max
                if not allow_ex:
                    new = 0
                else:
                    log_p1 = float(np.log(max(gamma1, 1e-12))) + normal_logpdf(yi, theta1, sigma2)
                    log_p0 = float(np.log(max(1.0 - gamma1, 1e-12))) + normal_logpdf(yi, theta2, sigma2)
                    m = max(log_p1, log_p0)
                    p1 = float(np.exp(log_p1 - m) / (np.exp(log_p1 - m) + np.exp(log_p0 - m)))
                    new = int(rng.random() < p1)

                if new != old:
                    if new == 1:
                        n1 = n1_without + 1
                        sum1 = sum1_without + yi
                    else:
                        n1 = n1_without
                        sum1 = sum1_without
                    c[i] = np.int8(new)

            if t >= burn and ((t - burn) % thin == 0):
                theta1_samples.append(theta1)
                lambda_samples.append(float(n1 / max(n_h, 1)))

        if len(theta1_samples) > 0:
            theta_arr = np.array(theta1_samples, dtype=float)
            lam_arr = np.array(lambda_samples, dtype=float)
            post_mean = float(np.mean(theta_arr))
            post_var = float(np.var(theta_arr, ddof=1)) if theta_arr.size > 1 else 1e-12
            lambda_eff = float(np.mean(lam_arr))
            w_eff = float(np.mean([w_from_lambda(lv, n_c, n_h) for lv in lam_arr]))
            return PosteriorSummary(
                mean=post_mean,
                var=max(post_var, 1e-12),
                lambda_eff=lambda_eff,
                w_eff=w_eff,
            )

    # Summary-only fallback (SSC-mixture approximation).
    # This branch keeps backward compatibility when individual historical outcomes
    # are unavailable to represent latent allocations C_{0i}.
    if n01_max <= 80:
        n01_vals = np.arange(0, n01_max + 1, dtype=float)
    else:
        n01_vals = np.unique(np.round(np.linspace(0.0, float(n01_max), 41))).astype(float)
        if n01_vals[0] != 0.0:
            n01_vals = np.concatenate([np.array([0.0]), n01_vals])
        if n01_vals[-1] != float(n01_max):
            n01_vals = np.concatenate([n01_vals, np.array([float(n01_max)])])

    log_prior = np.array(
        [
            float(log_choose(n_h, int(n01)) + betaln(alpha_g + n01, beta_g + n_h - n01) - betaln(alpha_g, beta_g))
            for n01 in n01_vals
        ],
        dtype=float,
    )
    log_prior -= logsumexp(log_prior)

    sigma2 = estimate_sigma2_continuous(arm)
    mu0 = float(p.normal_prior_mean)
    v0 = float(max(p.normal_prior_var, 1e-12))
    prec0 = 1.0 / v0
    like_var = float(max(sigma2 / n_c, 1e-12))
    like_prec = 1.0 / like_var

    means = []
    variances = []
    w_components = []
    log_ml = []
    for n01 in n01_vals:
        prech = float(max(n01 / max(sigma2, 1e-12), 0.0))
        prior_prec = prec0 + prech
        prior_var = 1.0 / max(prior_prec, 1e-12)
        prior_mean = (prec0 * mu0 + prech * float(arm.mean_h)) / max(prior_prec, 1e-12)

        post_var = 1.0 / (like_prec + prior_prec)
        post_mean = post_var * (like_prec * float(arm.mean_c) + prior_prec * prior_mean)
        means.append(float(post_mean))
        variances.append(float(post_var))
        lam_n01 = float(n01 / n_h)
        w_components.append(float(w_from_lambda(lam_n01, n_c, n_h)))
        log_ml.append(float(normal_logpdf(float(arm.mean_c), prior_mean, like_var + prior_var)))

    log_post = log_prior + np.array(log_ml, dtype=float)
    if not np.isfinite(log_post).any():
        weights = np.full_like(log_post, 1.0 / log_post.size, dtype=float)
    else:
        weights = np.exp(log_post - logsumexp(log_post))

    means_arr = np.array(means, dtype=float)
    vars_arr = np.array(variances, dtype=float)
    lam_arr = n01_vals / float(n_h)
    w_arr = np.array(w_components, dtype=float)

    post_mean = float(np.sum(weights * means_arr))
    post_var = float(np.sum(weights * (vars_arr + means_arr**2)) - post_mean**2)
    lambda_eff = float(np.sum(weights * lam_arr))
    w_eff = float(np.sum(weights * w_arr))
    return PosteriorSummary(mean=post_mean, var=max(post_var, 1e-12), lambda_eff=lambda_eff, w_eff=w_eff)


def posterior_mem(arm: Any, outcome: str, p: Any) -> PosteriorSummary:
    """Calculate the posterior mem
    References:
        Kaizer, A. M., Koopmeiners, J. S., and Hobbs, B. P. (2018). Bayesian hierarchical modeling based on
            multisource exchangeability. Biostatistics, 19(2):169-184.

    Args:
        arm (Any): The arm
        outcome (str): The outcome
        p (Any): The parameters

    Returns:
        PosteriorSummary: The posterior summary
    """
    if arm.n_h <= 0:
        return pooled_estimate(arm, 0.0, outcome)

    # MEM (Kaizer et al.) reduced to H=1 source:
    # - S=1: primary and supplemental are exchangeable.
    # - S=0: primary and supplemental are not exchangeable.
    # Posterior inference is a two-model BMA over {S=1, S=0}.
    pi_in = float(np.clip(p.mem_prior_inclusion, 1e-8, 1.0 - 1e-8))

    if outcome == "binary":
        s_c = int(arm.succ_c)
        f_c = int(arm.n_c - arm.succ_c)
        s_h = int(arm.succ_h)
        f_h = int(arm.n_h - arm.succ_h)

        a0 = float(max(p.beta_prior_alpha, 1e-8))
        b0 = float(max(p.beta_prior_beta, 1e-8))

        # S=1 (exchangeable): common parameter for current and historical.
        a_in = a0 + s_c + s_h
        b_in = b0 + f_c + f_h
        total_in = a_in + b_in
        mean_in = a_in / total_in
        var_in = (a_in * b_in) / (total_in**2 * (total_in + 1.0))
        post_in = PosteriorSummary(
            mean=float(mean_in),
            var=max(float(var_in), 1e-12),
            lambda_eff=1.0,
            w_eff=w_from_lambda(1.0, arm.n_c, arm.n_h),
        )

        # S=0 (non-exchangeable): primary estimated from current only.
        a_out = a0 + s_c
        b_out = b0 + f_c
        total_out = a_out + b_out
        mean_out = a_out / total_out
        var_out = (a_out * b_out) / (total_out**2 * (total_out + 1.0))
        post_out = PosteriorSummary(
            mean=float(mean_out),
            var=max(float(var_out), 1e-12),
            lambda_eff=0.0,
            w_eff=0.0,
        )

        # Integrated marginal likelihoods under each MEM.
        log_ml_in = beta_binomial_log_marginal(s_c + s_h, f_c + f_h, a0, b0)
        log_ml_out = beta_binomial_log_marginal(s_c, f_c, a0, b0) + beta_binomial_log_marginal(s_h, f_h, a0, b0)
    else:
        sigma2 = estimate_sigma2_continuous(arm)
        v_c = float(max(sigma2 / max(arm.n_c, 1), 1e-12))
        v_h = float(max(sigma2 / max(arm.n_h, 1), 1e-12))
        prec_c = 1.0 / v_c
        prec_h = 1.0 / v_h

        # S=1 (exchangeable): common mean, flat prior.
        var_in = 1.0 / (prec_c + prec_h)
        mean_in = var_in * (prec_c * arm.mean_c + prec_h * arm.mean_h)
        post_in = PosteriorSummary(
            mean=float(mean_in),
            var=max(float(var_in), 1e-12),
            lambda_eff=1.0,
            w_eff=w_from_lambda(1.0, arm.n_c, arm.n_h),
        )

        # S=0 (non-exchangeable): primary mean from current source only.
        post_out = PosteriorSummary(
            mean=float(arm.mean_c),
            var=max(float(v_c), 1e-12),
            lambda_eff=0.0,
            w_eff=0.0,
        )

        # H=1 simplification of Gaussian integrated marginal likelihood.
        diff = arm.mean_c - arm.mean_h
        log_ml_in = float(0.5 * np.log(2.0 * np.pi) - 0.5 * np.log(prec_c + prec_h) - 0.5 * (diff**2) / (v_c + v_h))
        log_ml_out = float(np.log(2.0 * np.pi) - 0.5 * np.log(prec_c * prec_h))

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
    References:
        Lu, X., and Lee, J. J. (2025). Overlapping indices for dynamic information borrowing in Bayesian hierarchical
            modeling. Journal of Computational and Graphical Statistics, pages 1-15.

    Args:
        arm (Any): The arm
        outcome (str): The outcome
        p (Any): The parameters

    Returns:
        PosteriorSummary: The posterior summary
    """
    if arm.n_h <= 0:
        return pooled_estimate(arm, 0.0, outcome)

    # Two-source reduction of BHMOI:
    # OBI for a cluster of size 2 is OVL(f_c, f_h), then alpha = s(OBI).
    # Posterior is obtained from the hierarchical model with
    #   theta_i | mu, tau ~ N(mu, 1/tau), tau ~ Gamma(alpha, beta).
    n_c = int(max(arm.n_c, 1))
    n_h = int(max(arm.n_h, 1))

    if outcome == "continuous":
        y_c = float(arm.mean_c)
        y_h = float(arm.mean_h)

        sigma2_fallback = estimate_sigma2_continuous(arm)
        if arm.n_c > 1:
            v_c = float(max(var_mean_cont(float(arm.var_c), n_c), 1e-12))
        else:
            v_c = float(max(sigma2_fallback / n_c, 1e-12))
        if arm.n_h > 1:
            v_h = float(max(var_mean_cont(float(arm.var_h), n_h), 1e-12))
        else:
            v_h = float(max(sigma2_fallback / n_h, 1e-12))
    else:
        # Approximate binomial likelihood on logit scale.
        p_c = clamp_prob((float(arm.succ_c) + 0.5) / (float(n_c) + 1.0))
        p_h = clamp_prob((float(arm.succ_h) + 0.5) / (float(n_h) + 1.0))
        y_c = float(logit(p_c))
        y_h = float(logit(p_h))
        v_c = float(max(1.0 / (float(n_c) * p_c * (1.0 - p_c)), 1e-12))
        v_h = float(max(1.0 / (float(n_h) * p_h * (1.0 - p_h)), 1e-12))

    obi = normal_overlap_coefficient(y_c, math.sqrt(v_c), y_h, math.sqrt(v_h))

    # BHMOI borrowing-strength map:
    # alpha = alpha_min + k_alpha(OBI) * (alpha_max - alpha_min),
    # k_alpha(OBI) = OBI * exp(-5 * (1-OBI)).
    damp = 5.0 * float(max(getattr(p, "bhmoi_sharpness", 1.0), 0.0))
    k_alpha = float(np.clip(obi * math.exp(-damp * (1.0 - obi)), 0.0, 1.0))
    alpha_min = float(getattr(p, "bhmoi_alpha_min", 1.0))
    alpha_max = float(getattr(p, "bhmoi_alpha_max", 200.0))
    if alpha_max <= alpha_min:
        alpha_max = alpha_min + 1.0
    alpha_tau = float(max(alpha_min + k_alpha * (alpha_max - alpha_min), 1e-8))
    beta_tau = float(max(getattr(p, "bhmoi_tau_beta", 10.0), 1e-8))

    # Hyperprior for cluster mean mu.
    mu0 = float(getattr(p, "normal_prior_mean", 0.0))
    v0 = float(max(getattr(p, "normal_prior_var", 1e8), 1e-12))

    # Numerical integration over log(tau).
    grid_size = max(int(getattr(p, "bhmoi_tau_grid_size", 401)), 11)
    log_tau_min = float(getattr(p, "bhmoi_tau_log_min", -8.0))
    log_tau_max = float(getattr(p, "bhmoi_tau_log_max", 8.0))
    if log_tau_max <= log_tau_min:
        log_tau_max = log_tau_min + 1.0

    z_grid = np.linspace(log_tau_min, log_tau_max, grid_size, dtype=float)
    tau_grid = np.exp(z_grid)

    means = np.empty_like(tau_grid)
    variances = np.empty_like(tau_grid)
    lam_grid = np.empty_like(tau_grid)
    log_post = np.empty_like(tau_grid)
    prior_const = alpha_tau * math.log(beta_tau) - math.lgamma(alpha_tau)

    for idx, tau in enumerate(tau_grid):
        s_c = v_c + 1.0 / tau
        s_h = v_h + 1.0 / tau

        # mu | y, tau
        prec_mu = 1.0 / v0 + 1.0 / s_c + 1.0 / s_h
        var_mu = 1.0 / max(prec_mu, 1e-12)
        mean_mu = var_mu * (mu0 / v0 + y_c / s_c + y_h / s_h)

        # theta_c | y, tau
        den_c = 1.0 + tau * v_c
        w_mu = (tau * v_c) / den_c
        mean_theta = (y_c + tau * v_c * mean_mu) / den_c
        var_theta = 1.0 / (1.0 / v_c + tau) + (w_mu * w_mu) * var_mu

        # p(y | tau) after integrating mu from Normal prior.
        a = s_c + v0
        d = s_h + v0
        off = v0
        det = max(a * d - off * off, 1e-12)
        inv00 = d / det
        inv11 = a / det
        inv01 = -off / det
        diff_c = y_c - mu0
        diff_h = y_h - mu0
        quad = inv00 * diff_c * diff_c + inv11 * diff_h * diff_h + 2.0 * inv01 * diff_c * diff_h
        log_ml = -0.5 * (2.0 * math.log(2.0 * math.pi) + math.log(det) + quad)

        log_prior_tau = prior_const + (alpha_tau - 1.0) * math.log(tau) - beta_tau * tau
        # Change-of-variables Jacobian for d tau = exp(z) dz.
        log_post[idx] = log_prior_tau + log_ml + z_grid[idx]
        means[idx] = mean_theta
        variances[idx] = max(var_theta, 1e-12)
        lam_grid[idx] = float(np.clip((tau * v_h) / (1.0 + tau * v_h), 0.0, 1.0))

    if not np.isfinite(log_post).any():
        return pooled_estimate(arm, 0.0, outcome)

    w = np.exp(log_post - logsumexp(log_post))
    theta_mean = float(np.sum(w * means))
    theta_var = float(np.sum(w * (variances + means * means)) - theta_mean * theta_mean)
    lam_eff = float(np.clip(np.sum(w * lam_grid), 0.0, 1.0))

    if outcome == "continuous":
        mean = theta_mean
        var = max(theta_var, 1e-12)
    else:
        mean = clamp_prob(float(expit(theta_mean)))
        grad = mean * (1.0 - mean)
        var = max(theta_var * grad * grad, 1e-12)

    return PosteriorSummary(
        mean=float(mean),
        var=float(var),
        lambda_eff=lam_eff,
        w_eff=w_from_lambda(lam_eff, arm.n_c, arm.n_h),
    )


def posterior_nonpara_bayes(arm: Any, outcome: str, p: Any) -> PosteriorSummary:
    """Calculate the posterior nonpara bayes
    References:
        Ohigashi, T., Maruo, K., Sozu, T., and Gosho, M. (2025). Nonparametric Bayesian approach for dynamic
            borrowing of historical control data. Biometrics, 81(3):ujaf118.

    Args:
        arm (Any): The arm
        outcome (str): The outcome
        p (Any): The parameters

    Returns:
        PosteriorSummary: The posterior summary
    """
    if arm.n_h <= 0:
        return pooled_estimate(arm, 0.0, outcome)

    # DPM/DDPM-inspired prior co-clustering probability for one historical source:
    #   pi0 = P(z_CC = z_H)
    #       = {2(M+1) - M*phi} / {2(M+1)^2 + M*phi}.
    # This is obtained from E[sum_c w_c^(H) w_c^(CC)] under the DDPM stick-breaking
    # construction in the manuscript, and recovers:
    #   - phi = 0 -> 1 / (1 + M)  (DPM-like)
    #   - phi = 1 -> 1 / (1 + 2M) (independent sticks)
    raw_concentration = getattr(p, "npb_concentration", None)
    if raw_concentration is None:
        raw_concentration = getattr(p, "npb_temperature", 5.0)
    concentration = float(max(raw_concentration if raw_concentration is not None else 5.0, 1e-8))
    phi = float(np.clip(getattr(p, "npb_phi", 0.5), 0.0, 1.0))
    prior_sbi_num = 2.0 * (concentration + 1.0) - concentration * phi
    prior_sbi_den = 2.0 * (concentration + 1.0) ** 2 + concentration * phi
    prior_sbi = float(np.clip(prior_sbi_num / max(prior_sbi_den, 1e-12), 1e-8, 1.0 - 1e-8))

    if outcome == "binary":
        a0 = float(max(p.beta_prior_alpha, 1e-8))
        b0 = float(max(p.beta_prior_beta, 1e-8))
        s_c = int(arm.succ_c)
        f_c = int(arm.n_c - arm.succ_c)
        s_h = int(arm.succ_h)
        f_h = int(arm.n_h - arm.succ_h)

        # Exchangeable model (same atom): common theta for current and historical.
        a_ex = a0 + s_c + s_h
        b_ex = b0 + f_c + f_h
        total_ex = a_ex + b_ex
        mean_ex = a_ex / total_ex
        var_ex = (a_ex * b_ex) / (total_ex**2 * (total_ex + 1.0))
        log_ml_ex = float(betaln(a_ex, b_ex) - betaln(a0, b0))

        # Potentially irrelevant model: separate atoms for current and historical.
        a_nex = a0 + s_c
        b_nex = b0 + f_c
        total_nex = a_nex + b_nex
        mean_nex = a_nex / total_nex
        var_nex = (a_nex * b_nex) / (total_nex**2 * (total_nex + 1.0))
        log_ml_nex = float(
            (betaln(a0 + s_c, b0 + f_c) - betaln(a0, b0)) + (betaln(a0 + s_h, b0 + f_h) - betaln(a0, b0))
        )
    else:
        sigma2 = estimate_sigma2_continuous(arm)
        v0 = float(max(p.normal_prior_var, 1e-12))
        mu0 = float(p.normal_prior_mean)
        v_c = float(max(sigma2 / max(arm.n_c, 1), 1e-12))
        v_h = float(max(sigma2 / max(arm.n_h, 1), 1e-12))

        prec0 = 1.0 / v0
        prec_c = 1.0 / v_c
        prec_h = 1.0 / v_h

        # Exchangeable model (same atom).
        var_ex = 1.0 / (prec0 + prec_c + prec_h)
        mean_ex = var_ex * (prec0 * mu0 + prec_c * arm.mean_c + prec_h * arm.mean_h)

        # Potentially irrelevant model (current-specific atom).
        var_nex = 1.0 / (prec0 + prec_c)
        mean_nex = var_nex * (prec0 * mu0 + prec_c * arm.mean_c)

        # Marginal likelihoods for model probability update.
        log_ml_h = normal_logpdf(arm.mean_h, mu0, v0 + v_h)
        var_post_h = 1.0 / (prec0 + prec_h)
        mean_post_h = var_post_h * (prec0 * mu0 + prec_h * arm.mean_h)
        log_pred_c_given_h = normal_logpdf(arm.mean_c, mean_post_h, v_c + var_post_h)
        log_ml_ex = float(log_ml_h + log_pred_c_given_h)
        log_ml_nex = float(log_ml_h + normal_logpdf(arm.mean_c, mu0, v0 + v_c))

    logit_post = float(logit(prior_sbi)) + (log_ml_ex - log_ml_nex)
    sbi = float(np.clip(expit(logit_post), 0.0, 1.0))

    mean = float(sbi * mean_ex + (1.0 - sbi) * mean_nex)
    var = float(sbi * (var_ex + mean_ex**2) + (1.0 - sbi) * (var_nex + mean_nex**2) - mean**2)
    lam_eff = sbi
    w_eff = w_from_lambda(lam_eff, arm.n_c, arm.n_h)
    return PosteriorSummary(
        mean=mean,
        var=max(var, 1e-12),
        lambda_eff=lam_eff,
        w_eff=w_eff,
    )


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
    lam_grid = np.linspace(0.0, 1.0, grid, dtype=float)
    lam0_grid = lam_grid if n_h0 > 0 else np.array([0.0], dtype=float)
    lam1_grid = lam_grid if n_h1 > 0 else np.array([0.0], dtype=float)

    if n_h0 > 0:
        denom0 = n_c0 + lam0_grid * n_h0
        w0_grid = np.divide(lam0_grid * n_h0, denom0, out=np.zeros_like(lam0_grid), where=denom0 > 0.0)
    else:
        w0_grid = np.zeros_like(lam0_grid)
    if n_h1 > 0:
        denom1 = n_c1 + lam1_grid * n_h1
        w1_grid = np.divide(lam1_grid * n_h1, denom1, out=np.zeros_like(lam1_grid), where=denom1 > 0.0)
    else:
        w1_grid = np.zeros_like(lam1_grid)

    var_c0_term = float(var_c0) / max(int(n_c0), 1)
    var_c1_term = float(var_c1) / max(int(n_c1), 1)
    var_h0_term = float(var_h0) / max(int(n_h0), 1) if n_h0 > 0 else 0.0
    var_h1_term = float(var_h1) / max(int(n_h1), 1) if n_h1 > 0 else 0.0

    s0_grid = (1.0 - w0_grid) ** 2 * var_c0_term + (w0_grid**2) * var_h0_term
    s1_grid = (1.0 - w1_grid) ** 2 * var_c1_term + (w1_grid**2) * var_h1_term
    s2_grid = s0_grid[:, None] + s1_grid[None, :]

    delta1_plus, delta1_minus = bond_shift_bounds(outcome, rho1, mu_c1)
    delta0_plus, delta0_minus = bond_shift_bounds(outcome, rho0, mu_c0)
    d1 = delta1_plus - delta1_minus
    d0 = delta0_plus - delta0_minus
    numerator = float(theta1) - d1 * w1_grid[None, :] - d0 * w0_grid[:, None]

    kappa_grid = np.full_like(s2_grid, -np.inf, dtype=float)
    valid = s2_grid > 0.0
    kappa_grid[valid] = numerator[valid] / np.sqrt(s2_grid[valid])
    finite = np.isfinite(kappa_grid)
    if not np.any(finite):
        return 0.0, 0.0, float(-np.inf)

    best_kappa = float(np.max(kappa_grid[finite]))
    best_mask = finite & np.isclose(kappa_grid, best_kappa, rtol=1e-12, atol=1e-12)
    best_idx = np.argwhere(best_mask)

    lam0_candidates = lam0_grid[best_idx[:, 0]]
    lam1_candidates = lam1_grid[best_idx[:, 1]]
    norms = lam0_candidates**2 + lam1_candidates**2
    min_norm = float(np.min(norms))
    min_norm_mask = np.isclose(norms, min_norm, rtol=1e-12, atol=1e-12)
    tie_idx = best_idx[min_norm_mask]

    # Lexicographic tie-breaking (lambda0 first, then lambda1) among min-norm maximizers.
    order = np.lexsort((tie_idx[:, 1], tie_idx[:, 0]))
    i0, i1 = tie_idx[int(order[0])]
    return float(lam0_grid[int(i0)]), float(lam1_grid[int(i1)]), best_kappa


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
    """Calculate the posterior current-only

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
    """Calculate the posterior naive pooling

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
    """Calculate the posterior fixed prior

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
    """Calculate the posterior power prior

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
    """Calculate the posterior commensurate prior

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
        raise ValueError("Commensurate prior requires param.")
    return posterior_commensurate(arm, outcome, float(param))


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
    """Calculate the posterior elastic prior

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
        raise ValueError("Elastic prior requires param.")
    return posterior_elastic(arm, outcome, p, float(param))


def method_unit_info_prior(
    arm: Any,
    outcome: str,
    p: Any,
    param: Optional[float] = None,
    arm_index: int = 0,
    bond_lambda0: float = 0.0,
    bond_lambda1: float = 0.0,
) -> PosteriorSummary:
    """Calculate the posterior unit-information prior

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
    """Calculate the posterior leap

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
    return posterior_leap(arm, outcome, p)


def method_mem(
    arm: Any,
    outcome: str,
    p: Any,
    param: Optional[float] = None,
    arm_index: int = 0,
    bond_lambda0: float = 0.0,
    bond_lambda1: float = 0.0,
) -> PosteriorSummary:
    """Calculate the posterior mem

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
    """Calculate the posterior bhmoi

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
    """Calculate the posterior nonparametric bayes

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
    """Calculate the test-then-pool estimate

    Args:
        arm (Any): The arm
        outcome (str): The outcome
        p (Any): The parameters
        param (Optional[float]): The parameter
        arm_index (int): The arm index
        bond_lambda0 (float): The bond lambda0
        bond_lambda1 (float): The bond lambda1

    Raises:
        ValueError: If the test-then-pool mode is unknown

    Returns:
        PosteriorSummary: The posterior summary
    """
    if arm.n_h <= 0:
        return pooled_estimate(arm, 0.0, outcome)

    # Two variants in Li et al. (2020):
    # - "difference" (original test-then-pool): pool if no significant difference.
    # - "equivalence" (TOST-based): pool if equivalence is established.
    mode = str(getattr(p, "ttp_mode", "difference")).strip().lower()
    alpha_ts = float(np.clip(getattr(p, "ttp_alpha_ts", getattr(p, "alpha_pool", 0.1)), 1e-12, 1.0 - 1e-12))
    alpha_eq = float(np.clip(getattr(p, "ttp_alpha_eq", 0.05), 1e-12, 1.0 - 1e-12))

    if outcome == "binary":
        n_c = int(max(arm.n_c, 1))
        n_h = int(max(arm.n_h, 1))
        p_c = clamp_prob((float(arm.succ_c) + 0.5) / (float(n_c) + 1.0))
        p_h = clamp_prob((float(arm.succ_h) + 0.5) / (float(n_h) + 1.0))
        diff = float(p_c - p_h)
        se2 = float(max(p_c * (1.0 - p_c) / n_c + p_h * (1.0 - p_h) / n_h, 1e-12))
    else:
        diff = float(arm.mean_c - arm.mean_h)
        se2 = float(max(var_mean_cont(arm.var_c, arm.n_c) + var_mean_cont(arm.var_h, arm.n_h), 1e-12))

    if se2 <= 0.0:
        return pooled_estimate(arm, 0.0, outcome)

    se = float(math.sqrt(se2))
    pool = False

    if mode in {"difference", "original", "ts", "two-sided"}:
        # H0: theta_h = theta_c vs H1: theta_h != theta_c
        # Pool when failing to reject H0.
        z = diff / se
        pval = float(2.0 * norm.sf(abs(float(z))))
        pool = bool(pval > alpha_ts)
    elif mode in {"equivalence", "eq", "tost"}:
        # H0: theta_c - theta_h >= delta or <= -delta
        # H1: -delta < theta_c - theta_h < delta
        # Pool when rejecting H0 via TOST.
        if outcome == "binary":
            delta_attr = "ttp_equiv_margin_binary"
        else:
            delta_attr = "ttp_equiv_margin"
        delta = getattr(p, delta_attr, None)
        if delta is None:
            # If delta is not specified, use the relationship from Li et al. (Eq. 8)
            # that makes original and equivalence-based pooling conditions identical.
            delta = (norm.ppf(1.0 - alpha_eq) + norm.ppf(1.0 - alpha_ts / 2.0)) * se
        delta = float(max(delta, 0.0))
        lower = -delta + norm.ppf(1.0 - alpha_eq) * se
        upper = delta - norm.ppf(1.0 - alpha_eq) * se
        pool = bool(lower < diff < upper)
    else:
        raise ValueError(f"Unknown test-then-pool mode: {mode}")

    lam = 1.0 if pool else 0.0
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
    """Calculate the BOND estimate

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
    lam = bond_lambda0 if arm_index == 0 else bond_lambda1
    return pooled_estimate(arm, lam, outcome)


MethodRunner = Callable[[Any, str, Any, Optional[float], int, float, float], PosteriorSummary]

METHOD_RUNNERS: Dict[str, MethodRunner] = {
    "current": method_current,
    "naive": method_naive_pool,
    "fixed": method_fixed,
    "power_prior": method_power_prior,
    "commensurate": method_commensurate_prior,
    "robust_map": method_robust_map,
    "elastic": method_elastic_prior,
    "uip": method_unit_info_prior,
    "leap": method_leap,
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
    """Estimate the arm for a given method

    Args:
        method (MethodSpec): The method specification
        arm (Any): The arm
        outcome (str): The outcome
        p (Any): The parameters
        arm_index (int): The arm index
        bond_lambda0 (float): The bond lambda0
        bond_lambda1 (float): The bond lambda1

    Raises:
        ValueError: If the method family is unknown

    Returns:
        PosteriorSummary: The posterior summary
    """
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
