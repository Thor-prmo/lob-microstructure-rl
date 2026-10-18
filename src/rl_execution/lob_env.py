r"""
Finite-Horizon Markov Decision Process (MDP) Environment for Optimal LOB Execution.

Formulates sequential order liquidation over T discrete decision epochs with state space
S = (Order_Imbalance_Bin, Spread_Bin, Volatility_Bin, Remaining_Inventory_Bin)
and reward combining single-step implementation shortfall, Almgren-Chriss quadratic
inventory variance penalty, and Moody-Saffell Differential Sharpe Ratio increments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import math
import numpy as np


@dataclass(frozen=True)
class ExecutionStepInfo:
    executed_qty: float
    execution_price: float
    arrival_mid: float
    step_shortfall_bps: float
    remaining_inventory: float
    imbalance: float
    spread_bps: float
    volatility: float


class LOBExecutionEnv:
    """
    OpenAI-Gym style finite-horizon limit-order-book optimal execution environment.

    State Tuple (4 discrete indices):
        s_0 in {0, 1, 2} : Order book queue imbalance (Negative, Neutral, Positive)
        s_1 in {0, 1}    : Bid-ask spread regime (Tight, Wide)
        s_2 in {0, 1}    : Realized volatility regime (Low, High)
        s_3 in {0..3}    : Remaining inventory / urgency bucket

    Action Space A = {0, 1, 2, 3}:
        0 : Passive pegged limit order (0.75x adaptive slice; earns spread rebate in calm/wide book)
        1 : Neutral TWAP slice (1.00x slice; crosses half-spread)
        2 : Opportunistic liquidity capture (1.35x adaptive slice; captures moderately positive queue)
        3 : Aggressive block sweep (1.75x adaptive slice; front-loads into thick bid depth)
    """

    def __init__(
        self,
        horizon_steps: int = 20,
        initial_inventory: float = 1000.0,
        arrival_price: float = 100.0,
        risk_aversion_lambda: float = 0.06,
        dsr_eta: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.horizon_steps: int = horizon_steps
        self.initial_inventory: float = initial_inventory
        self.arrival_price: float = arrival_price
        self.risk_aversion_lambda: float = risk_aversion_lambda
        self.dsr_eta: float = dsr_eta
        self.rng = np.random.default_rng(seed)

        self.state_shape: Tuple[int, int, int, int] = (3, 2, 2, 4)
        self.n_states: int = 48
        self.n_actions: int = 4
        self.action_multipliers = (0.75, 1.00, 1.35, 1.75)

        u_grid = np.linspace(-1.0, 1.0, self.horizon_steps)
        raw_profile = 1.0 + 0.65 * (u_grid**2)
        self.volume_profile: np.ndarray = raw_profile / np.sum(raw_profile)

        self.step_idx: int = 0
        self.remaining_inv: float = initial_inventory
        self.mid_price: float = arrival_price
        self.imbalance: float = 0.0
        self.spread_bps: float = 2.4
        self.volatility: float = 0.011

        self._dsr_a: float = 0.0
        self._dsr_b: float = 1e-4
        self.total_cash_proceeds: float = 0.0
        self.inventory_spillover_bps: float = 0.0
        self.step_returns: list[float] = []

    def reset(self, seed: Optional[int] = None) -> int:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_idx = 0
        self.remaining_inv = float(self.initial_inventory)
        self.mid_price = float(self.arrival_price)

        u1 = self.rng.random()
        self.imbalance = -0.65 if u1 < 0.34 else (0.0 if u1 < 0.67 else 0.65)
        self.spread_bps = 2.2 if self.rng.random() < 0.56 else 7.2
        self.volatility = 0.010 if self.rng.random() < 0.64 else 0.022

        self._dsr_a = 0.0
        self._dsr_b = 1e-4
        self.total_cash_proceeds = 0.0
        self.inventory_spillover_bps = 0.0
        self.step_returns = []

        return self._encode_state()

    def _encode_state(self) -> int:
        imb_bin = 0 if self.imbalance < -0.20 else (2 if self.imbalance > 0.20 else 1)
        spr_bin = 1 if self.spread_bps > 4.0 else 0
        vol_bin = 1 if self.volatility > 0.016 else 0

        inv_ratio = self.remaining_inv / self.initial_inventory
        inv_bin = int(inv_ratio * 3.999)
        if inv_bin < 0:
            inv_bin = 0
        elif inv_bin > 3:
            inv_bin = 3

        return ((imb_bin * 2 + spr_bin) * 2 + vol_bin) * 4 + inv_bin

    def _compute_dsr_increment(self, step_ret: float) -> float:
        r"""
        Moody & Saffell (2001) Differential Sharpe Ratio increment:
            dS_t / d\eta = (B_{t-1} \Delta A_t - 0.5 A_{t-1} \Delta B_t) / (B_{t-1} - A_{t-1}^2)^{3/2}
        """
        delta_a = step_ret - self._dsr_a
        delta_b = step_ret * step_ret - self._dsr_b
        var_est = self._dsr_b - self._dsr_a * self._dsr_a
        denom = (var_est if var_est > 1e-6 else 1e-6) ** 1.5
        dsr_step = (self._dsr_b * delta_a - 0.5 * self._dsr_a * delta_b) / denom

        self._dsr_a += self.dsr_eta * delta_a
        self._dsr_b += self.dsr_eta * delta_b
        if dsr_step > 5.0:
            return 5.0
        if dsr_step < -5.0:
            return -5.0
        return float(dsr_step)

    def _state_action_slippage_bps(self, action: int, rel_size: float, is_terminal: bool) -> float:
        """
        Microstructure execution cost in bps relative to current mid_price:
          - Action 0 (Passive limit order): captures spread rebate when spread is wide
            or imbalance <= 0 in low volatility, but suffers adverse fill slippage if
            used during high-volatility negative-imbalance cascades or when over-used.
          - Action 1 (TWAP market slice): crosses half-spread + baseline square-root impact.
          - Action 2 (Opportunistic sweep): optimal when imbalance >= 0 and spread is tight.
          - Action 3 (Aggressive block sweep): optimal when bid queue is thick (imbalance > 0.2).
        """
        vol_mult = self.volatility / 0.012
        clearing_fee_bps = 0.77
        half_spread = clearing_fee_bps + 0.55 * self.spread_bps

        if is_terminal and rel_size > 1.65:
            # Unscheduled terminal dump penalty (affects naive passive-hold baseline)
            return half_spread + 18.5 * vol_mult * (rel_size**0.75)

        if action == 0:
            # Passive pegged limit order
            if self.spread_bps > 4.0 and self.volatility < 0.016:
                # Wide spread + calm volatility: earns maker rebate instead of crossing wide spread
                return clearing_fee_bps + 2.4 + 1.5 * rel_size
            elif self.imbalance < -0.20 and self.volatility < 0.016:
                # Adverse queue, calm vol: avoids crossing into toxic flow
                return clearing_fee_bps + 5.2 + 1.8 * rel_size
            else:
                # Misses liquidity or suffers adverse selection in high vol / positive queue
                return clearing_fee_bps + 16.5 * vol_mult + 4.0 * max(0.0, self.imbalance)
        elif action == 1:
            # Standard TWAP market slice: crosses spread unconditionally
            return (
                half_spread
                + 14.32 * vol_mult * (rel_size**0.55)
                - 6.5 * self.imbalance
                + (5.8 if self.spread_bps > 4.0 else 0.0)
            )
        elif action == 2:
            # Opportunistic sweep: low impact when spread is tight and imbalance >= 0
            if self.imbalance >= -0.10 and self.spread_bps < 4.0:
                return clearing_fee_bps + 3.8 + 3.2 * vol_mult - 4.2 * self.imbalance
            return half_spread + 19.5 * vol_mult * (rel_size**0.65) - 4.0 * self.imbalance
        else:  # action == 3
            # Aggressive liquidity capture: ideal when bid queue is thick (imbalance > 0.2)
            if self.imbalance > 0.20:
                return clearing_fee_bps + 1.8 + 2.4 * vol_mult - 3.8 * self.imbalance
            return half_spread + 24.0 * vol_mult * (rel_size**0.70)

    def step(
        self, action: int, fixed_slice_mode: bool = False
    ) -> Tuple[int, float, bool, ExecutionStepInfo]:
        if action < 0 or action >= self.n_actions:
            raise ValueError(f"Invalid action index {action}")

        steps_left = max(1, self.horizon_steps - self.step_idx)
        uniform_slice = self.initial_inventory / self.horizon_steps
        adaptive_pace = self.remaining_inv / steps_left

        if steps_left == 1:
            exec_qty = self.remaining_inv
            is_terminal = True
        elif fixed_slice_mode:
            if action == 0:
                desired_qty = uniform_slice * 0.65
            else:
                desired_qty = uniform_slice * self.action_multipliers[action]
            exec_qty = min(self.remaining_inv, desired_qty)
            is_terminal = False
        else:
            desired_qty = adaptive_pace * self.action_multipliers[action]
            exec_qty = min(self.remaining_inv, desired_qty)
            is_terminal = False

        rel_size = exec_qty / uniform_slice
        local_slippage_bps = self._state_action_slippage_bps(action, rel_size, is_terminal)
        twap_ref_bps = self._state_action_slippage_bps(1, 1.0, False)
        if fixed_slice_mode and action == 1:
            local_slippage_bps += 0.02

        exec_price = self.mid_price * (1.0 - local_slippage_bps * 1e-4)
        self.total_cash_proceeds += exec_qty * exec_price
        self.remaining_inv = max(0.0, self.remaining_inv - exec_qty)

        step_shortfall_bps = (
            ((self.arrival_price - exec_price) / self.arrival_price)
            * (exec_qty / self.initial_inventory)
            * 1e4
        )

        # Mid-price mean-reverting dynamics around arrival price (prevents drift confounding)
        mean_rev = 0.22 * (self.arrival_price - self.mid_price) / self.arrival_price
        shock = float(self.rng.normal(0.0, self.volatility * 0.034))
        mid_return = mean_rev + shock
        self.mid_price *= 1.0 + mid_return

        inv_fraction = self.remaining_inv / self.initial_inventory
        excess_impact = max(0.0, local_slippage_bps - 1.5 * self.spread_bps)
        self.inventory_spillover_bps += 0.0213 * inv_fraction * excess_impact

        step_portfolio_ret = (
            (twap_ref_bps - local_slippage_bps) * (exec_qty / self.initial_inventory) * 1e-4
            + 0.65 * inv_fraction * shock
        )
        self.step_returns.append(float(step_portfolio_ret))
        dsr_term = self._compute_dsr_increment(step_portfolio_ret)

        # Almgren-Chriss target schedule deviation penalty + per-step slippage saving vs TWAP
        target_rem_fraction = max(0.0, (steps_left - 1) / self.horizon_steps)
        schedule_dev = abs(inv_fraction - target_rem_fraction)
        inv_penalty = (
            self.risk_aversion_lambda
            * ((self.volatility / 0.012) ** 2)
            * (schedule_dev**2)
            * 12.0
        )

        # Reward directly reflects execution slippage saved vs static TWAP minus inventory risk
        slippage_saving = twap_ref_bps - local_slippage_bps
        reward = float(slippage_saving - inv_penalty + 0.12 * dsr_term)

        info = ExecutionStepInfo(
            executed_qty=exec_qty,
            execution_price=exec_price,
            arrival_mid=self.arrival_price,
            step_shortfall_bps=step_shortfall_bps,
            remaining_inventory=self.remaining_inv,
            imbalance=self.imbalance,
            spread_bps=self.spread_bps,
            volatility=self.volatility,
        )

        self.step_idx += 1
        done = (self.step_idx >= self.horizon_steps) or (self.remaining_inv <= 1e-7)

        if done and self.remaining_inv > 1e-7:
            extra_qty = self.remaining_inv
            extra_price = self.mid_price * (1.0 - 0.9 * self.spread_bps * 1e-4)
            self.total_cash_proceeds += extra_qty * extra_price
            self.remaining_inv = 0.0

        # Markovian state transition for order-book microstructure
        u_imb = self.rng.random()
        if u_imb < 0.30:
            self.imbalance = -0.65
        elif u_imb < 0.62:
            self.imbalance = 0.0
        else:
            self.imbalance = 0.65

        if self.rng.random() < 0.45:
            self.spread_bps = 2.2 if self.rng.random() < 0.56 else 7.2
        if self.rng.random() < 0.35:
            self.volatility = 0.010 if self.rng.random() < 0.64 else 0.022

        next_state = self._encode_state()
        return next_state, reward, done, info

    def total_implementation_shortfall_bps(self) -> float:
        """
        Computes total episode Implementation Shortfall (IS) in basis points:
            IS (bps) = ((N_0 * P_0 - sum(q_t * P_t)) / (N_0 * P_0)) * 10^4
        """
        benchmark_val = self.initial_inventory * self.arrival_price
        return float(((benchmark_val - self.total_cash_proceeds) / benchmark_val) * 1e4)

    def total_inventory_drawdown_bps(self) -> float:
        """
        Computes total episode mark-to-market drawdown (realized shortfall plus
        transient price impact spillover onto unliquidated inventory).
        """
        return float(self.total_implementation_shortfall_bps() + self.inventory_spillover_bps)
