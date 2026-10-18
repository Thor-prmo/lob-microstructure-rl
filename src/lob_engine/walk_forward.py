"""
Anchored Walk-Forward Backtesting Engine with Volatility-Regime Filtering.

Enforces strict train-validation-test chronological partitioning to eliminate
lookahead bias when calibrating momentum z-score thresholds and volatility caps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from .execution_sim import ExecutionSimulator, ExecutionReport


@dataclass(frozen=True)
class WalkForwardResult:
    strategy_report: ExecutionReport
    benchmark_report: ExecutionReport
    positions: np.ndarray
    stress_regime_dd_reduction_pct: float
    selected_thresholds: List[Tuple[float, float]]


class WalkForwardBacktester:
    """
    Vectorized walk-forward signal generator and hyperparameter selector.
    """

    def __init__(
        self,
        simulator: ExecutionSimulator,
        lookback_fast: int = 14,
        lookback_slow: int = 42,
        vol_window: int = 25,
    ) -> None:
        self.simulator = simulator
        self.lookback_fast = lookback_fast
        self.lookback_slow = lookback_slow
        self.vol_window = vol_window

    @staticmethod
    def compute_causal_features(
        returns: np.ndarray,
        lookback_fast: int = 14,
        lookback_slow: int = 42,
        vol_window: int = 25,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes strictly causal rolling time-series momentum z-scores and realized
        volatility using only prefix observations [0 .. t].
        """
        n = len(returns)
        z_mom = np.zeros(n, dtype=float)
        realized_vol = np.zeros(n, dtype=float)

        for t in range(n):
            start_vol = max(0, t - vol_window + 1)
            window_r = returns[start_vol : t + 1]
            vol_t = float(np.std(window_r, ddof=0)) if len(window_r) > 2 else 0.015
            realized_vol[t] = max(vol_t, 1e-4)

            if t >= lookback_slow:
                fast_slice = returns[t - lookback_fast + 1 : t + 1]
                slow_slice = returns[t - lookback_slow + 1 : t + 1]
                mu_fast = float(np.mean(fast_slice))
                mu_slow = float(np.mean(slow_slice))
                sig_slow = float(np.std(slow_slice, ddof=1)) + 1e-8
                # Blended Moskowitz-Ooi-Pedersen time-series momentum t-statistic
                z_mom[t] = (0.65 * mu_fast + 0.35 * mu_slow) / (sig_slow / np.sqrt(21.0))

        return z_mom, realized_vol

    @staticmethod
    def generate_positions_for_params(
        z_mom: np.ndarray,
        realized_vol: np.ndarray,
        z_entry: float,
        vol_cap_percentile_val: float,
    ) -> np.ndarray:
        """
        Generates low-turnover discrete regime positions in {-0.35, 0.0, 0.50, 1.20}
        with hysteresis deadbands so turnover only occurs on regime switches.
        """
        n = len(z_mom)
        pos = np.zeros(n, dtype=float)
        current_p = 0.50

        for t in range(n):
            if realized_vol[t] > vol_cap_percentile_val:
                # High-volatility liquidity stress regime: de-risk or hold protective short
                if z_mom[t] < -0.50 * z_entry:
                    current_p = -0.35
                elif z_mom[t] < 0.50 * z_entry:
                    current_p = 0.0
            else:
                if z_mom[t] > z_entry:
                    current_p = 1.20
                elif z_mom[t] < 0.0:
                    current_p = 0.0
            pos[t] = current_p

        return pos

    def run_anchored_walk_forward(
        self,
        returns: np.ndarray,
        volumes: np.ndarray,
        initial_train_size: int = 600,
        val_size: int = 200,
        test_fold_size: int = 250,
    ) -> WalkForwardResult:
        """
        Executes anchored walk-forward optimization:
        For each fold k:
          1. Fit volatility percentiles on [0 : train_end]
          2. Select (z_entry, vol_quantile) maximizing net Sharpe on [train_end : val_end]
          3. Freeze parameters and evaluate out-of-sample on [val_end : test_end]
        """
        n = len(returns)
        z_mom, realized_vol = self.compute_causal_features(
            returns, self.lookback_fast, self.lookback_slow, self.vol_window
        )

        oos_positions = np.zeros(n, dtype=float)
        selected_params: List[Tuple[float, float]] = []

        z_grid = [0.35, 0.50, 0.65, 0.80]
        vol_q_grid = [68.0, 74.0, 80.0]

        cursor = initial_train_size
        oos_start_idx = initial_train_size + val_size

        while cursor + val_size < n:
            train_end = cursor
            val_end = min(cursor + val_size, n)
            test_end = min(val_end + test_fold_size, n)

            best_sharpe = -1e9
            best_pair = (0.50, float(np.percentile(realized_vol[:train_end], 74.0)))

            for z_cand in z_grid:
                for q_cand in vol_q_grid:
                    vol_thresh = float(np.percentile(realized_vol[:train_end], q_cand))
                    val_pos = self.generate_positions_for_params(
                        z_mom[:val_end], realized_vol[:val_end], z_cand, vol_thresh
                    )[train_end:val_end]

                    rep = self.simulator.evaluate_strategy(
                        returns[train_end:val_end],
                        val_pos,
                        volumes[train_end:val_end],
                        realized_vol[train_end:val_end],
                    )
                    if rep.sharpe_ratio > best_sharpe:
                        best_sharpe = rep.sharpe_ratio
                        best_pair = (z_cand, vol_thresh)

            selected_params.append(best_pair)
            full_prefix_pos = self.generate_positions_for_params(
                z_mom[:test_end], realized_vol[:test_end], best_pair[0], best_pair[1]
            )
            oos_positions[val_end:test_end] = full_prefix_pos[val_end:test_end]
            cursor += test_fold_size

        oos_ret = returns[oos_start_idx:]
        oos_vol_mkt = volumes[oos_start_idx:]
        oos_rvol = realized_vol[oos_start_idx:]
        oos_pos = oos_positions[oos_start_idx:]

        strat_report = self.simulator.evaluate_strategy(
            oos_ret, oos_pos, oos_vol_mkt, oos_rvol
        )
        bh_pos = np.ones_like(oos_pos)
        bh_report = self.simulator.evaluate_strategy(
            oos_ret, bh_pos, oos_vol_mkt, oos_rvol
        )

        dd_reduction_pct = (
            (abs(bh_report.max_drawdown) - abs(strat_report.max_drawdown))
            / max(abs(bh_report.max_drawdown), 1e-8)
            * 100.0
        )

        return WalkForwardResult(
            strategy_report=strat_report,
            benchmark_report=bh_report,
            positions=oos_pos,
            stress_regime_dd_reduction_pct=float(dd_reduction_pct),
            selected_thresholds=selected_params,
        )

