"""
Multi-Agent Auction Mechanisms & Revenue Equivalence Verification.

Simulates First-Price Sealed-Bid, Second-Price (Vickrey), and Walrasian
Double Auctions under Independent Private Values (IPV) v_i ~ Uniform[0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass(frozen=True)
class AuctionSimulationResult:
    n_bidders: int
    n_trials: int
    theoretical_expected_revenue: float
    first_price_mean_revenue: float
    second_price_mean_revenue: float
    relative_discrepancy_pct: float
    vickrey_truthful_optimality_rate: float


def simulate_independent_private_value_auctions(
    n_bidders: int = 5,
    n_trials: int = 100_000,
    seed: int = 42,
) -> AuctionSimulationResult:
    """
    Verifies the Revenue Equivalence Theorem across First-Price and Second-Price
    (Vickrey) auctions for N symmetric risk-neutral bidders with v_i ~ U[0, 1]:
        - First-Price BNE bid: b^{(1)}(v_i) = ((N - 1) / N) * v_i
        - Second-Price dominant strategy: b^{(2)}(v_i) = v_i
        - Theoretical expected seller revenue: E[R] = (N - 1) / (N + 1)
    """
    if n_bidders < 2:
        raise ValueError("Auction requires at least 2 bidders.")

    rng = np.random.default_rng(seed)
    fp_valuations = rng.uniform(0.0, 1.0, size=(n_trials, n_bidders))
    sp_valuations = rng.uniform(0.0, 1.0, size=(n_trials, n_bidders))

    # First-Price Bayes-Nash Equilibrium bids
    fp_bids = ((n_bidders - 1.0) / n_bidders) * fp_valuations
    fp_revenues = np.max(fp_bids, axis=1)

    # Second-Price (Vickrey) truthful bids: winner pays second-highest bid
    sorted_vals = np.sort(sp_valuations, axis=1)
    sp_revenues = sorted_vals[:, -2]

    theo_rev = (n_bidders - 1.0) / (n_bidders + 1.0)
    fp_mean = float(np.mean(fp_revenues))
    sp_mean = float(np.mean(sp_revenues))

    rel_diff_pct = abs(fp_mean - sp_mean) / theo_rev * 100.0

    # Verify weak dominance of truthful bidding in Vickrey auction against random shade/overbid
    deviations = rng.uniform(-0.25, 0.25, size=n_trials)
    v_0 = sp_valuations[:, 0]
    max_rival = np.max(sp_valuations[:, 1:], axis=1)

    truthful_payoff = np.where(v_0 > max_rival, v_0 - max_rival, 0.0)
    dev_bid = np.clip(v_0 + deviations, 0.0, 1.25)
    dev_payoff = np.where(dev_bid > max_rival, v_0 - max_rival, 0.0)
    optimality_rate = float(np.mean(truthful_payoff >= dev_payoff - 1e-12))

    return AuctionSimulationResult(
        n_bidders=n_bidders,
        n_trials=n_trials,
        theoretical_expected_revenue=theo_rev,
        first_price_mean_revenue=fp_mean,
        second_price_mean_revenue=sp_mean,
        relative_discrepancy_pct=float(rel_diff_pct),
        vickrey_truthful_optimality_rate=optimality_rate,
    )


def simulate_walrasian_double_auction(
    n_buyers: int = 50,
    n_sellers: int = 50,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Simulates a two-sided call market (double auction) matching descending buyer
    limit prices against ascending seller marginal costs to compute Walrasian
    clearing price, volume, and allocative efficiency.
    """
    rng = np.random.default_rng(seed)
    buyer_vals = np.sort(rng.uniform(80.0, 120.0, size=n_buyers))[::-1]
    seller_costs = np.sort(rng.uniform(80.0, 120.0, size=n_sellers))

    trades = buyer_vals >= seller_costs
    k_star = int(np.sum(trades))
    if k_star == 0:
        return {"clearing_price": 100.0, "volume": 0.0, "efficiency_pct": 0.0}

    clearing_price = 0.5 * float(buyer_vals[k_star - 1] + seller_costs[k_star - 1])
    total_surplus = float(np.sum(buyer_vals[:k_star] - seller_costs[:k_star]))

    return {
        "clearing_price": clearing_price,
        "volume": float(k_star),
        "total_surplus": total_surplus,
        "efficiency_pct": 100.0,
    }

