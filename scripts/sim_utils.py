from __future__ import annotations

from typing import List, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit  # type: ignore
from scipy.stats import norm, wasserstein_distance  # type: ignore


def parse_float_list(text: str) -> List[float]:
    """Parse a comma-separated list of floats

    Args:
        text (str): The comma-separated list of floats

    Returns:
        List[float]: The list of floats
    """
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> List[int]:
    """Parse a comma-separated list of integers

    Args:
        text (str): The comma-separated list of integers

    Returns:
        List[int]: The list of integers
    """
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_optional_float_list(text: str) -> Optional[List[float]]:
    """Parse a comma-separated list of floats, returning None if the list is empty

    Args:
        text (str): The comma-separated list of floats

    Returns:
        Optional[List[float]]: The list of floats, or None if the list is empty
    """
    cleaned = text.strip()
    if cleaned == "":
        return None
    return parse_float_list(cleaned)


def z_alpha(alpha: float) -> float:
    """Compute the z-score for a given alpha level

    Args:
        alpha (float): The alpha level

    Returns:
        float: The z-score
    """
    return float(norm.isf(alpha))


def clamp_prob(p: float) -> float:
    """Clamp a probability value to the range [1e-6, 1.0 - 1e-6]

    Args:
        p (float): The probability value

    Returns:
        float: The clamped probability value
    """
    return max(1e-6, min(1.0 - 1e-6, p))


def wasserstein_1d(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Compute the Wasserstein distance between two 1D arrays

    Args:
        x (NDArray[np.float64]): The first 1D array
        y (NDArray[np.float64]): The second 1D array

    Returns:
        float: The Wasserstein distance
    """
    if x.size == 0 or y.size == 0:
        return 0.0
    return float(wasserstein_distance(x, y))


def calibrate_binary_tau(
    beta0: float,
    beta: NDArray[np.float64],
    eta: NDArray[np.float64],
    theta_target: float,
    p_dim: int,
    mc_size: int,
    seed: int,
) -> float:
    """Calibrate the tau parameter for a binary outcome

    Args:
        beta0 (float): The intercept parameter
        beta (NDArray[np.float64]): The beta parameters
        eta (NDArray[np.float64]): The eta parameters
        theta_target (float): The target mean difference
        p_dim (int): The dimension of the covariates
        mc_size (int): The number of Monte Carlo samples
        seed (int): The random seed

    Returns:
        float: The calibrated tau parameter
    """
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

    def objective(tau: float) -> float:
        """Objective function for the calibration

        Args:
            tau (float): The tau parameter

        Returns:
            float: The objective value
        """
        mu1 = float(np.mean(expit(lin1_base + tau)))
        return mu1 - target_mu1

    lo = -12.0
    hi = 12.0
    flo = objective(lo)
    fhi = objective(hi)
    while flo > 0.0:
        hi = lo
        lo -= 8.0
        flo = objective(lo)
        if lo < -80.0:
            break
    while fhi < 0.0:
        lo = hi
        hi += 8.0
        fhi = objective(hi)
        if hi > 80.0:
            break

    for _ in range(100):
        mid = 0.5 * (lo + hi)
        fmid = objective(mid)
        if abs(fmid) < 1e-8:
            return float(mid)
        if fmid > 0.0:
            hi = mid
        else:
            lo = mid
    return float(0.5 * (lo + hi))
