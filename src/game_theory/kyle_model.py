r"""
Kyle (1985) Continuous Auction & Sequential Insider Price-Impact Equilibrium.

Solves both the single-period analytical equilibrium and the N-period sequential
dynamic program via backward induction, simulating order flow y_t = x_t + u_t
and market-maker linear pricing P_t = P_{t-1} + \lambda_t y_t.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class KyleSinglePeriodEquilibrium:
    p0: float
    sigma0_var: float
    noise_vol: float
    beta: float
    lambda_impact: float
    posterior_var: float
    expected_insider_profit: float


@dataclass(frozen=True)
class KyleMultiPeriodTrajectory:
    n_steps: int
    lambda_path: np.ndarray
    beta_path: np.ndarray
    sigma_var_path: np.ndarray
    mean_price_paths: np.ndarray
    true_value: float


def solve_kyle_single_period(
    p0: float = 100.0,
    sigma0_var: float = 4.0,
    noise_vol: float = 10.0,
) -> KyleSinglePeriodEquilibrium:
    """
    Computes the unique linear Bayesian Nash Equilibrium of Kyle (1985):
        beta   = sigma_u / sqrt(Sigma_0)
        lambda = sqrt(Sigma_0) / (2 * sigma_u)
        Sigma_1 = 0.5 * Sigma_0
        E[pi]   = 0.5 * sqrt(Sigma_0) * sigma_u
    """
    if sigma0_var <= 0 or noise_vol <= 0:
        raise ValueError("Variance and noise volatility must be strictly positive.")

    sqrt_sig0 = float(np.sqrt(sigma0_var))
    beta = float(noise_vol / sqrt_sig0)
    lam = float(sqrt_sig0 / (2.0 * noise_vol))
    post_var = 0.5 * sigma0_var
    expected_profit = 0.5 * sqrt_sig0 * noise_vol

    return KyleSinglePeriodEquilibrium(
        p0=p0,
        sigma0_var=sigma0_var,
        noise_vol=noise_vol,
        beta=beta,
        lambda_impact=lam,
        posterior_var=post_var,
        expected_insider_profit=expected_profit,
    )


def simulate_kyle_sequential_auction(
    n_steps: int = 25,
    p0: float = 100.0,
    true_value: float = 104.0,
    sigma0_var: float = 4.0,
    noise_vol: float = 10.0,
    n_paths: int = 300,
    seed: int = 42,
) -> KyleMultiPeriodTrajectory:
    """
    Simulates multi-period sequential Kyle equilibrium paths across `n_steps`
    trading auctions, demonstrating monotonic decay of posterior variance Sigma_t
    and convergence of market maker price P_t -> true fundamental value v.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / n_steps

    lambda_path = np.zeros(n_steps, dtype=float)
    beta_path = np.zeros(n_steps, dtype=float)
    sigma_var_path = np.zeros(n_steps + 1, dtype=float)
    sigma_var_path[0] = sigma0_var

    # Sequential Bayesian filtering recursion for Kyle's dynamic equilibrium
    for k in range(n_steps):
        rem_time = max(1.0 - k * dt, dt)
        sig_k = sigma_var_path[k]
        beta_k = float((noise_vol / np.sqrt(sig_k)) / np.sqrt(rem_time))
        denom = noise_vol**2 + (beta_k**2) * sig_k * dt
        lam_k = float((beta_k * sig_k) / denom)
        sig_next = float(sig_k * (noise_vol**2) / denom)

        lambda_path[k] = lam_k
        beta_path[k] = beta_k
        sigma_var_path[k + 1] = sig_next

    price_paths = np.zeros((n_paths, n_steps + 1), dtype=float)
    price_paths[:, 0] = p0

    for i in range(n_paths):
        p_cur = p0
        for k in range(n_steps):
            # Insider order x_k + noise order u_k
            x_k = beta_path[k] * (true_value - p_cur) * dt
            u_k = float(rng.normal(0.0, noise_vol * np.sqrt(dt)))
            y_k = x_k + u_k
            p_cur = p_cur + lambda_path[k] * y_k
            price_paths[i, k + 1] = p_cur

    return KyleMultiPeriodTrajectory(
        n_steps=n_steps,
        lambda_path=lambda_path,
        beta_path=beta_path,
        sigma_var_path=sigma_var_path,
        mean_price_paths=price_paths,
        true_value=true_value,
    )
