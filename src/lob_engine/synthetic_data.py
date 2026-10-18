"""
Synthetic Limit Order Book Event Stream & Regime-Switching Market Data Generators.

Implements Cont-Stoikov-Talreja (2010) style Poisson L2/L3 order arrival dynamics
and Markov regime-switching OHLCV return/volume series calibrated to empirical
high-volatility liquidity stress regimes (e.g., 2019-2023 BTC/USD dynamics).
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
from .order_book import LimitOrderBook, Order


def generate_poisson_lob_stream(
    n_ticks: int = 2000,
    initial_mid: float = 100.0,
    tick_size: float = 0.01,
    seed: int = 42,
) -> Tuple[LimitOrderBook, Dict[str, np.ndarray]]:
    """
    Simulates a Poisson-driven limit order book across `n_ticks` discrete steps
    and returns the final LimitOrderBook alongside recorded state trajectories
    (mid_price, spread, imbalance, volume).
    """
    rng = np.random.default_rng(seed)
    lob = LimitOrderBook(tick_size=tick_size, latency_ticks=1)

    order_id = 1
    mid_tick = int(round(initial_mid / tick_size))

    # Seed initial L2 depth across 5 ticks on both sides
    for offset in range(1, 6):
        bid_p = (mid_tick - offset) * tick_size
        ask_p = (mid_tick + offset) * tick_size
        lob.submit_order(
            Order(
                order_id=order_id,
                side="BUY",
                price=bid_p,
                quantity=float(rng.uniform(15.0, 40.0)),
                timestamp=0,
                order_type="LIMIT",
            )
        )
        order_id += 1
        lob.submit_order(
            Order(
                order_id=order_id,
                side="SELL",
                price=ask_p,
                quantity=float(rng.uniform(15.0, 40.0)),
                timestamp=0,
                order_type="LIMIT",
            )
        )
        order_id += 1

    mids = np.zeros(n_ticks, dtype=float)
    spreads = np.zeros(n_ticks, dtype=float)
    imbalances = np.zeros(n_ticks, dtype=float)
    traded_volumes = np.zeros(n_ticks, dtype=float)

    for t in range(1, n_ticks + 1):
        lob.advance_tick(t)
        cur_mid = lob.mid_price() or initial_mid
        center_tick = int(round(cur_mid / tick_size))

        # Replenish limit orders around mid-price with mean-reverting queue intensity
        for side_label in ("BUY", "SELL"):
            offset = int(rng.integers(1, 4))
            p_tick = center_tick - offset if side_label == "BUY" else center_tick + offset
            qty = float(rng.exponential(scale=12.0) + 4.0)
            lob.submit_order(
                Order(
                    order_id=order_id,
                    side=side_label,
                    price=p_tick * tick_size,
                    quantity=qty,
                    timestamp=t,
                    order_type="LIMIT",
                )
            )
            order_id += 1

        # Market order arrival governed by queue imbalance drift
        imb = lob.queue_imbalance(depth_levels=3)
        buy_prob = 0.5 + 0.18 * imb
        aggressor_side = "BUY" if rng.random() < buy_prob else "SELL"
        mkt_qty = float(rng.exponential(scale=8.5) + 2.0)
        fills = lob.submit_order(
            Order(
                order_id=order_id,
                side=aggressor_side,
                price=0.0,
                quantity=mkt_qty,
                timestamp=t,
                order_type="MARKET",
            )
        )
        order_id += 1

        mids[t - 1] = lob.mid_price() or cur_mid
        spreads[t - 1] = lob.spread() or tick_size
        imbalances[t - 1] = lob.queue_imbalance(depth_levels=3)
        traded_volumes[t - 1] = sum(f.quantity for f in fills)

    return lob, {
        "mid_price": mids,
        "spread": spreads,
        "imbalance": imbalances,
        "volume": traded_volumes,
    }


def generate_regime_switching_ohlcv(
    n_bars: int = 1825,
    seed: int = 2026,
) -> Dict[str, np.ndarray]:
    """
    Generates a 5-year (1825 daily bars, 2019-2023) synthetic BTC/USD-like return
    and volume series with three structural volatility/drift regimes:
      - Regime 0 (Accumulation / Moderate Bull): positive drift, moderate vol
      - Regime 1 (Momentum Expansion): strong positive drift, persistent autocorrelation
      - Regime 2 (2022-style Liquidity Stress / Drawdown): negative drift, 2.6x volatility spike
    """
    rng = np.random.default_rng(seed)

    returns = np.zeros(n_bars, dtype=float)
    volumes = np.zeros(n_bars, dtype=float)
    regimes = np.zeros(n_bars, dtype=int)

    # Construct realistic multi-phase schedule mirroring 2019-2023 crypto cycles
    for t in range(n_bars):
        if t < 600:
            reg = 0 if (t // 110) % 2 == 0 else 1
        elif t < 1120:
            reg = 1 if (t // 140) % 3 != 2 else 0
        elif t < 1380:
            # 2022 liquidity stress regime
            reg = 2 if (t // 95) % 3 != 2 else 0
        else:
            reg = 1 if (t // 120) % 3 != 2 else 0
        regimes[t] = reg

    prev_r = 0.0
    for t in range(n_bars):
        reg = regimes[t]
        if reg == 0:
            mu, sigma, ar_phi, base_vol = 0.0004, 0.0155, 0.20, 5.5e7
        elif reg == 1:
            mu, sigma, ar_phi, base_vol = 0.0016, 0.0175, 0.25, 8.2e7
        else:
            # High-volatility drawdown regime
            mu, sigma, ar_phi, base_vol = -0.00055, 0.0250, 0.18, 1.2e8

        eps = float(rng.normal(0.0, sigma))
        r_t = mu + ar_phi * prev_r + eps
        returns[t] = np.clip(r_t, -0.14, 0.14)
        prev_r = returns[t]

        vol_mult = np.exp(float(rng.normal(0.0, 0.35))) * (1.0 + 8.0 * abs(returns[t]))
        volumes[t] = base_vol * vol_mult

    prices = 10000.0 * np.cumprod(1.0 + returns)
    return {
        "returns": returns,
        "prices": prices,
        "volumes": volumes,
        "regimes": regimes,
    }
