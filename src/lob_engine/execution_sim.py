"""
Square-Root Market Impact & Proportional Fee Execution Simulator.

Models non-linear execution slippage under the square-root impact law:
    \\Delta P = Y * \\sigma_t * \\sqrt{Q / V_t}
combined with half-spread crossing and proportional exchange fees (e.g., 15 bps).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ExecutionReport:
    net_returns: np.ndarray
    equity_curve: np.ndarray
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    tail_var_99: float
    win_rate: float
    avg_slippage_bps: float


class ExecutionSimulator:
    """
    Simulates realistic execution costs including proportional exchange fees
    and square-root market-impact slippage across time-series trading signals.
    """

    def __init__(
        self,
        fee_bps: float = 15.0,
        impact_coeff: float = 0.35,
        half_spread_bps: float = 2.5,
        annualization_factor: float = 365.0,
    ) -> None:
        self.fee_rate: float = fee_bps * 1e-4
        self.impact_coeff: float = impact_coeff
        self.half_spread_rate: float = half_spread_bps * 1e-4
        self.ann_factor: float = annualization_factor

    def compute_square_root_slippage(
        self,
        order_notional: np.ndarray,
        market_volume: np.ndarray,
        realized_vol: np.ndarray,
    ) -> np.ndarray:
        """
        Computes fractional price slippage:
            s_t = half_spread + Y * sigma_t * sqrt(|Q_t| / V_t)
        """
        participation = np.clip(
            np.abs(order_notional) / np.maximum(market_volume, 1e-8), 0.0, 0.25
        )
        impact = self.impact_coeff * realized_vol * np.sqrt(participation)
        active_mask = (np.abs(order_notional) > 1e-9).astype(float)
        return active_mask * (self.half_spread_rate + impact)

    def evaluate_strategy(
        self,
        asset_returns: np.ndarray,
        target_positions: np.ndarray,
        market_volume: np.ndarray,
        realized_vol: np.ndarray,
        trade_notional_scale: float = 25_000.0,
    ) -> ExecutionReport:
        """
        Evaluates net out-of-sample performance strictly causally:
        position w_{t-1} earns return r_t, while turnover |w_t - w_{t-1}|
        incurs exchange fees and square-root slippage at time t.
        """
        n = len(asset_returns)
        if len(target_positions) != n:
            raise ValueError(
                "asset_returns and target_positions must have identical length."
            )

        # Causal shift: position decided at end of bar t-1 earns bar t return
        lagged_pos = np.zeros_like(target_positions)
        lagged_pos[1:] = target_positions[:-1]

        turnover = np.abs(target_positions - lagged_pos)
        order_notional = turnover * trade_notional_scale

        slippage_rate = self.compute_square_root_slippage(
            order_notional, market_volume, realized_vol
        )
        total_cost = turnover * (self.fee_rate + slippage_rate)

        gross_returns = lagged_pos * asset_returns
        net_returns = gross_returns - total_cost

        equity_curve = np.cumprod(1.0 + net_returns)
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max
        max_dd = float(np.min(drawdowns))

        mean_ret = float(np.mean(net_returns))
        std_ret = float(np.std(net_returns, ddof=1)) + 1e-12
        sharpe = float((mean_ret / std_ret) * np.sqrt(self.ann_factor))

        downside = net_returns[net_returns < 0.0]
        downside_std = (
            float(np.std(downside, ddof=1)) + 1e-12
            if len(downside) > 1
            else std_ret
        )
        sortino = float((mean_ret / downside_std) * np.sqrt(self.ann_factor))

        ann_return = float(equity_curve[-1] ** (self.ann_factor / max(n, 1)) - 1.0)
        calmar = float(ann_return / max(abs(max_dd), 1e-6))

        tail_var_99 = float(-np.percentile(net_returns, 1.0))
        active_bars = net_returns[np.abs(lagged_pos) > 1e-6]
        win_rate = float(np.mean(active_bars > 0.0)) if len(active_bars) > 0 else 0.0

        active_trades = slippage_rate[turnover > 1e-6]
        avg_slippage_bps = (
            float(np.mean(active_trades) * 1e4) if len(active_trades) > 0 else 0.0
        )

        return ExecutionReport(
            net_returns=net_returns,
            equity_curve=equity_curve,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            tail_var_99=tail_var_99,
            win_rate=win_rate,
            avg_slippage_bps=avg_slippage_bps,
        )

