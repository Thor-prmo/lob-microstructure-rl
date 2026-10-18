r"""
Kyle (1985) Continuous Auction & Sequential Insider Price-Impact Equilibrium.

Solves both the single-period linear equilibrium and the multi-period sequential
dynamic programming recursion (Kyle 1985, Theorem 2) for insider trading intensity
\beta_t, market-maker illiquidity parameter \lambda_t (Kyle's lambda), and
residual variance \Sigma_t = Var(v | y_{1:t}).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class KyleSimulationResult:
    theoretical_lambda: float
    empirical_lambda_ols: float
    theoretical_beta: float
    theoretical_insider_profit: float
    empirical_insider_profit: float
    multi_period_lambdas: np.ndarray
    multi_period_sigmas: np.ndarray


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


class KyleEquilibriumSolver:
    """
    Analytical and Monte Carlo solver for Kyle (1985) adverse selection equilibria.
    """

    def __init__(self, p0: float = 100.0, sigma_v0: float = 4.0, sigma_u: float = 10.0) -> None:
        if sigma_v0 <= 0 or sigma_u <= 0:
            raise ValueError("Variance and noise parameters must be strictly positive.")
        self.p0 = p0
        self.sigma_v0 = sigma_v0  # Prior variance \Sigma_0 of fundamental value v
        self.sigma_u = sigma_u    # Volatility \sigma_u of noise trader order flow u

    def single_period_equilibrium(self) -> Tuple[float, float, float, float]:
        r"""
        Returns (beta, lambda_kyle, sigma_1, expected_insider_profit):
            \beta = \sigma_u / \sqrt{\Sigma_0}
            \lambda = \sqrt{\Sigma_0} / (2 \sigma_u)
            \Sigma_1 = \Sigma_0 / 2
            E[\pi] = \frac{1}{2} \sqrt{\Sigma_0} \sigma_u
        """
        beta = self.sigma_u / np.sqrt(self.sigma_v0)
        lambda_kyle = np.sqrt(self.sigma_v0) / (2.0 * self.sigma_u)
        sigma_1 = 0.5 * self.sigma_v0
        expected_profit = 0.5 * np.sqrt(self.sigma_v0) * self.sigma_u
        return float(beta), float(lambda_kyle), float(sigma_1), float(expected_profit)

    def solve_sequential_multi_period(self, n_steps: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Solves the exact N-period sequential Kyle (1985) Theorem 2 recursion:
            For step k = N, N-1, ..., 1 (with terminal \alpha_N = 0):
                \frac{\lambda_k \sigma_u^2 \Delta t}{\Sigma_k} = \frac{1 - 2 \alpha_k \lambda_k}{2 \lambda_k (1 - \alpha_k \lambda_k)}
                \beta_k = \frac{\lambda_k \sigma_u^2}{\Sigma_k}
                \Sigma_{k-1} = \frac{\Sigma_k}{1 - \beta_k \lambda_k \Delta t}
                \alpha_{k-1} = \frac{1}{4 \lambda_k (1 - \alpha_k \lambda_k)}
        Uses Brent's root-finding method on terminal variance \Sigma_N \in (0, \Sigma_0)
        so that the backward-propagated initial variance matches \Sigma_0.
        """
        dt = 1.0 / n_steps
        su2_dt = (self.sigma_u ** 2) * dt

        def backward_sweep(sigma_N: float) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
            sigmas = np.zeros(n_steps + 1, dtype=float)
            lambdas = np.zeros(n_steps, dtype=float)
            betas = np.zeros(n_steps, dtype=float)
            sigmas[n_steps] = sigma_N

            alpha_k = 0.0  # Terminal boundary condition \alpha_N = 0
            for k in range(n_steps - 1, -1, -1):
                sig_k = sigmas[k + 1]
                if alpha_k <= 1e-15:
                    lam_k = np.sqrt(sig_k / (2.0 * su2_dt))
                else:
                    # Unique positive root in (0, 1 / (2 * alpha_k)) satisfying SOC 1 - 2*alpha_k*lam_k > 0
                    upper = (1.0 / (2.0 * alpha_k)) * (1.0 - 1e-11)
                    lam_k = float(
                        brentq(
                            lambda lam: (lam ** 2) * (1.0 - alpha_k * lam) * (2.0 * su2_dt / sig_k)
                            - (1.0 - 2.0 * alpha_k * lam),
                            1e-12,
                            upper,
                            xtol=1e-12,
                        )
                    )

                beta_k = (lam_k * (self.sigma_u ** 2)) / sig_k
                denom = 1.0 - beta_k * lam_k * dt
                sig_prev = sig_k / max(denom, 1e-12)
                alpha_prev = 1.0 / (4.0 * lam_k * (1.0 - alpha_k * lam_k))

                lambdas[k] = lam_k
                betas[k] = beta_k
                sigmas[k] = sig_prev
                alpha_k = alpha_prev

            return sigmas[0] - self.sigma_v0, betas, lambdas, sigmas

        sigma_N_star = float(brentq(lambda s: backward_sweep(s)[0], 1e-6, self.sigma_v0 * 0.99, xtol=1e-10))
        _, betas, lambdas, sigmas = backward_sweep(sigma_N_star)
        return betas, lambdas, sigmas

    def simulate_and_verify(self, n_trials: int = 100_000, seed: int = 42) -> KyleSimulationResult:
        rng = np.random.default_rng(seed)
        beta, lam, _, theo_profit = self.single_period_equilibrium()

        v = rng.normal(self.p0, np.sqrt(self.sigma_v0), size=n_trials)
        u = rng.normal(0.0, self.sigma_u, size=n_trials)
        x = beta * (v - self.p0)
        y = x + u

        cov_vy = float(np.cov(v, y, ddof=1)[0, 1])
        var_y = float(np.var(y, ddof=1))
        emp_lambda = cov_vy / var_y
        p_exec = self.p0 + emp_lambda * y

        insider_profits = (v - p_exec) * x
        emp_profit = float(np.mean(insider_profits))

        _, mp_lambdas, mp_sigmas = self.solve_sequential_multi_period(n_steps=15)

        return KyleSimulationResult(
            theoretical_lambda=lam,
            empirical_lambda_ols=emp_lambda,
            theoretical_beta=beta,
            theoretical_insider_profit=theo_profit,
            empirical_insider_profit=emp_profit,
            multi_period_lambdas=mp_lambdas,
            multi_period_sigmas=mp_sigmas,
        )


def solve_kyle_single_period(
    p0: float = 100.0,
    sigma0_var: float = 4.0,
    noise_vol: float = 10.0,
) -> KyleSinglePeriodEquilibrium:
    """
    Computes the unique linear Bayesian Nash Equilibrium of Kyle (1985).
    """
    solver = KyleEquilibriumSolver(p0=p0, sigma_v0=sigma0_var, sigma_u=noise_vol)
    beta, lam, post_var, profit = solver.single_period_equilibrium()
    return KyleSinglePeriodEquilibrium(
        p0=p0,
        sigma0_var=sigma0_var,
        noise_vol=noise_vol,
        beta=beta,
        lambda_impact=lam,
        posterior_var=post_var,
        expected_insider_profit=profit,
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
    Simulates multi-period sequential Kyle equilibrium paths using exact Kyle (1985) Theorem 2 recursion.
    """
    solver = KyleEquilibriumSolver(p0=p0, sigma_v0=sigma0_var, sigma_u=noise_vol)
    betas, lambdas, sigmas = solver.solve_sequential_multi_period(n_steps=n_steps)

    rng = np.random.default_rng(seed)
    dt = 1.0 / n_steps

    price_paths = np.zeros((n_paths, n_steps + 1), dtype=float)
    price_paths[:, 0] = p0

    for i in range(n_paths):
        p_cur = p0
        for k in range(n_steps):
            x_k = betas[k] * (true_value - p_cur) * dt
            u_k = float(rng.normal(0.0, noise_vol * np.sqrt(dt)))
            y_k = x_k + u_k
            p_cur = p_cur + lambdas[k] * y_k
            price_paths[i, k + 1] = p_cur

    return KyleMultiPeriodTrajectory(
        n_steps=n_steps,
        lambda_path=lambdas,
        beta_path=betas,
        sigma_var_path=sigmas,
        mean_price_paths=price_paths,
        true_value=true_value,
    )
