"""
Baseline Execution Policies (TWAP, VWAP, Passive Close) & Out-of-Sample Evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np
from .lob_env import LOBExecutionEnv


@dataclass(frozen=True)
class PolicyEvaluationSummary:
    policy_name: str
    mean_shortfall_bps: float
    std_shortfall_bps: float
    return_variance: float
    sharpe_ratio: float
    max_inventory_drawdown_bps: float
    mean_inventory_path: np.ndarray
    shortfall_distribution_bps: np.ndarray


def twap_policy(state: int, env: LOBExecutionEnv) -> int:
    """Static Time-Weighted Average Price (uniform slice at every step)."""
    return 1


def vwap_policy(state: int, env: LOBExecutionEnv) -> int:
    """Static Volume-Weighted Average Price (matches U-shaped intraday volume curve)."""
    step_vol_weight = env.volume_profile[min(env.step_idx, env.horizon_steps - 1)]
    uniform_weight = 1.0 / env.horizon_steps
    if step_vol_weight > 1.12 * uniform_weight:
        return 2
    return 1


def passive_close_policy(state: int, env: LOBExecutionEnv) -> int:
    """Places passive slices early, forcing large terminal sweep at horizon."""
    return 0


def evaluate_policy_episodes(
    env: LOBExecutionEnv,
    policy_fn: Callable[[int, LOBExecutionEnv], int],
    policy_name: str,
    n_eval_episodes: int = 500,
    base_seed: int = 50000,
) -> PolicyEvaluationSummary:
    """
    Evaluates an execution policy across `n_eval_episodes` out-of-sample trajectories.
    """
    shortfalls = np.zeros(n_eval_episodes, dtype=float)
    episode_returns = np.zeros(n_eval_episodes, dtype=float)
    episode_drawdowns = np.zeros(n_eval_episodes, dtype=float)
    inventory_paths = np.zeros((n_eval_episodes, env.horizon_steps + 1), dtype=float)

    is_static = policy_name in (
        "Static TWAP Baseline",
        "Static VWAP Baseline",
        "Passive / Terminal Sweep",
    )

    for i in range(n_eval_episodes):
        state = env.reset(seed=base_seed + i)
        inventory_paths[i, 0] = env.remaining_inv / env.initial_inventory
        done = False
        t = 0

        while not done and t < env.horizon_steps:
            action = policy_fn(state, env)
            state, _, done, _ = env.step(action, fixed_slice_mode=is_static)
            t += 1
            inventory_paths[i, t] = env.remaining_inv / env.initial_inventory

        shortfalls[i] = env.total_implementation_shortfall_bps()
        episode_returns[i] = float(np.sum(env.step_returns))
        episode_drawdowns[i] = env.total_inventory_drawdown_bps()

    mean_is = float(np.mean(shortfalls))
    std_is = float(np.std(shortfalls, ddof=1))
    ret_var = float(np.var(shortfalls * 1e-4, ddof=1))
    sharpe = float(np.mean(episode_returns) / (np.std(episode_returns, ddof=1) + 1e-12))

    # Mean cumulative drawdown across 50-block (10-week) institutional execution cycles
    cycles = episode_drawdowns.reshape(-1, 50).sum(axis=1)
    max_dd_bps = float(np.mean(cycles))

    return PolicyEvaluationSummary(
        policy_name=policy_name,
        mean_shortfall_bps=mean_is,
        std_shortfall_bps=std_is,
        return_variance=ret_var,
        sharpe_ratio=sharpe,
        max_inventory_drawdown_bps=max_dd_bps,
        mean_inventory_path=np.mean(inventory_paths, axis=0),
        shortfall_distribution_bps=shortfalls,
    )

