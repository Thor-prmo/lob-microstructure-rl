"""
Unit tests verifying strict causality (zero lookahead leakage), 1-tick execution
latency buffering, and FIFO price-time priority lot conservation.
"""

from __future__ import annotations

import numpy as np
from src.lob_engine.order_book import LimitOrderBook, Order
from src.lob_engine.execution_sim import ExecutionSimulator
from src.lob_engine.walk_forward import WalkForwardBacktester


def test_fifo_price_time_priority_and_lot_conservation() -> None:
    """
    Verifies that two resting limit orders at the same price level are filled in
    strict FIFO timestamp arrival order, and that total matched volume equals
    the reduction in resting queue depth.
    """
    lob = LimitOrderBook(tick_size=0.01, latency_ticks=0)

    # Submit two resting sell limit orders at $100.05
    o1 = Order(order_id=101, side="SELL", price=100.05, quantity=30.0, timestamp=1)
    o2 = Order(order_id=102, side="SELL", price=100.05, quantity=50.0, timestamp=2)
    # And one deeper sell order at $100.06
    o3 = Order(order_id=103, side="SELL", price=100.06, quantity=40.0, timestamp=3)

    lob.submit_order(o1)
    lob.submit_order(o2)
    lob.submit_order(o3)

    # Incoming market buy for 45 lots should completely fill o1 (30 lots)
    # and partially fill o2 (15 lots), leaving 35 lots on o2 and 40 lots on o3.
    aggressor = Order(
        order_id=201,
        side="BUY",
        price=0.0,
        quantity=45.0,
        timestamp=4,
        order_type="MARKET",
    )
    fills = lob.submit_order(aggressor)

    assert len(fills) == 2
    assert fills[0].sell_order_id == 101
    assert np.isclose(fills[0].quantity, 30.0)
    assert fills[1].sell_order_id == 102
    assert np.isclose(fills[1].quantity, 15.0)
    assert np.isclose(o2.remaining_qty, 35.0)
    assert np.isclose(o3.remaining_qty, 40.0)


def test_one_tick_latency_prevents_instantaneous_fill() -> None:
    """
    Verifies that a strategy order submitted at tick t with latency_ticks=1
    does NOT match against tick-t resting liquidity, and only executes upon
    advancing the clock to tick t+1.
    """
    lob = LimitOrderBook(tick_size=0.01, latency_ticks=1)
    lob.advance_tick(10)

    lob.submit_order(
        Order(order_id=1, side="SELL", price=100.00, quantity=25.0, timestamp=10)
    )

    strat_buy = Order(
        order_id=99,
        side="BUY",
        price=100.00,
        quantity=10.0,
        timestamp=10,
        order_type="LIMIT",
        is_strategy=True,
    )
    immediate_fills = lob.submit_order(strat_buy)
    assert len(immediate_fills) == 0, "Strategy order violated 1-tick latency buffer!"

    delayed_fills = lob.advance_tick(11)
    assert len(delayed_fills) == 1
    assert delayed_fills[0].timestamp == 11
    assert np.isclose(delayed_fills[0].quantity, 10.0)


def test_zero_lookahead_in_walk_forward_features_and_positions() -> None:
    """
    Mutates future return observations r_{t+1 : T} with extreme outliers and
    asserts that causal features (z_mom, realized_vol) and generated positions
    at all times <= t remain bitwise/numerically invariant (atol=1e-12).
    """
    rng = np.random.default_rng(123)
    n = 400
    base_returns = rng.normal(0.0005, 0.018, size=n)

    # Create a corrupted series identical up to t_cutoff = 220, but wildly mutated afterward
    t_cutoff = 220
    mutated_returns = base_returns.copy()
    mutated_returns[t_cutoff + 1 :] = rng.normal(-0.08, 0.15, size=n - t_cutoff - 1)

    z_base, vol_base = WalkForwardBacktester.compute_causal_features(base_returns)
    z_mut, vol_mut = WalkForwardBacktester.compute_causal_features(mutated_returns)

    assert np.allclose(z_base[: t_cutoff + 1], z_mut[: t_cutoff + 1], atol=1e-12)
    assert np.allclose(vol_base[: t_cutoff + 1], vol_mut[: t_cutoff + 1], atol=1e-12)

    pos_base = WalkForwardBacktester.generate_positions_for_params(
        z_base, vol_base, z_entry=0.75, vol_cap_percentile_val=0.025
    )
    pos_mut = WalkForwardBacktester.generate_positions_for_params(
        z_mut, vol_mut, z_entry=0.75, vol_cap_percentile_val=0.025
    )
    assert np.allclose(pos_base[: t_cutoff + 1], pos_mut[: t_cutoff + 1], atol=1e-12)


def test_execution_simulator_square_root_slippage_monotonicity() -> None:
    """
    Verifies that square-root market impact slippage increases monotonically
    with order notional Q and realized volatility sigma_t.
    """
    sim = ExecutionSimulator(fee_bps=15.0, impact_coeff=0.35, half_spread_bps=2.5)
    vol_mkt = np.array([1e7, 1e7, 1e7])
    r_vol = np.array([0.015, 0.015, 0.015])
    notionals = np.array([1e4, 5e4, 2e5])

    slip = sim.compute_square_root_slippage(notionals, vol_mkt, r_vol)
    assert slip[0] < slip[1] < slip[2]

