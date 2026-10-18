"""
Unit tests for Kyle (1985) price impact equilibrium, Primal-Dual LP strong
duality, Iterated Elimination of Strictly Dominated Strategies (IESDS),
and Auction Revenue Equivalence bounds.
"""

from __future__ import annotations

import numpy as np
from src.game_theory.kyle_model import (
    solve_kyle_single_period,
    simulate_kyle_sequential_auction,
)
from src.game_theory.auctions import (
    simulate_independent_private_value_auctions,
    simulate_walrasian_double_auction,
)
from src.game_theory.lp_equilibrium import (
    solve_zero_sum_game_lp,
    iterated_elimination_strictly_dominated,
)


def test_kyle_single_and_multi_period_invariants() -> None:
    """
    Verifies single-period Kyle (1985) analytical relations:
      1. Posterior variance Sigma_1 == 0.5 * Sigma_0
      2. beta * lambda == 0.5
      3. Multi-period posterior variance decreases monotonically.
    """
    eq = solve_kyle_single_period(p0=100.0, sigma0_var=9.0, noise_vol=12.0)
    assert np.isclose(eq.posterior_var, 4.5, atol=1e-12)
    assert np.isclose(eq.beta * eq.lambda_impact, 0.5, atol=1e-12)

    traj = simulate_kyle_sequential_auction(
        n_steps=20, p0=100.0, true_value=105.0, sigma0_var=4.0, noise_vol=10.0
    )
    diffs = np.diff(traj.sigma_var_path)
    assert np.all(diffs < 0.0), "Posterior variance must strictly decrease across auctions."


def test_vickrey_and_first_price_revenue_equivalence() -> None:
    """
    Verifies that empirical revenues in First-Price BNE and Second-Price Vickrey
    auctions match theoretical E[R] = (N - 1) / (N + 1) within 0.8% discrepancy,
    and that truthful bidding in Vickrey is weakly dominant (100% optimality).
    """
    res = simulate_independent_private_value_auctions(
        n_bidders=5, n_trials=100_000, seed=42
    )
    assert res.relative_discrepancy_pct < 0.8
    assert np.isclose(
        res.second_price_mean_revenue, res.theoretical_expected_revenue, rtol=0.01
    )
    assert np.isclose(res.vickrey_truthful_optimality_rate, 1.0, atol=1e-12)

    da = simulate_walrasian_double_auction(n_buyers=40, n_sellers=40, seed=42)
    assert 80.0 <= da["clearing_price"] <= 120.0
    assert da["volume"] > 0


def test_zero_sum_lp_strong_duality_and_iesds() -> None:
    """
    Verifies zero duality gap |v^* - w^*| < 1e-10 on symmetric and asymmetric
    zero-sum games, and tests iterated dominance elimination.
    """
    # Generalized asymmetric zero-sum payoff matrix
    payoff = np.array(
        [
            [3.0, -1.0, 2.0, 0.5],
            [-2.0, 4.0, 1.0, -0.5],
            [1.0, 2.0, -3.0, 1.5],
        ]
    )
    res = solve_zero_sum_game_lp(payoff)
    assert res.duality_gap < 1e-10
    assert np.isclose(np.sum(res.row_strategy), 1.0, atol=1e-10)
    assert np.isclose(np.sum(res.col_strategy), 1.0, atol=1e-10)

    # Verify saddle-point inequality: min_j (p^T A)_j >= v^* - 1e-9
    assert np.min(res.row_strategy @ payoff) >= res.primal_value - 1e-8
    assert np.max(payoff @ res.col_strategy) <= res.dual_value + 1e-8

    # IESDS test where row 0 strictly dominates row 2
    A_dom = np.array([[5.0, 4.0], [2.0, 6.0], [1.0, 1.0]])
    B_dom = np.array([[3.0, 1.0], [2.0, 4.0], [5.0, 0.0]])
    _, _, rows_left, _ = iterated_elimination_strictly_dominated(A_dom, B_dom)
    assert 2 not in rows_left

