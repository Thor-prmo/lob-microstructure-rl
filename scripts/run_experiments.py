#!/usr/bin/env python3
"""
Master Reproducibility Script for `lob-microstructure-rl`.

Executes all four core experimental suites from raw random seeds:
  1. Tabular Q-Learning Optimal LOB Execution (10,000 episodes) vs. TWAP/VWAP/Passive
  2. Multi-Seed Inventory Liquidation Trajectory & Implementation Shortfall Analysis
  3. Kyle (1985) Insider Price-Impact Equilibrium & Auction Revenue Equivalence MC
  4. Anchored Walk-Forward Regime-Filtered Backtest under Square-Root Market Impact

Generates four 300-DPI diagnostic figures in `figures/` and prints summary benchmark tables.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure writable Matplotlib cache dir in sandboxed environments
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-lob-cache")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.lob_engine.synthetic_data import (
    generate_poisson_lob_stream,
    generate_regime_switching_ohlcv,
)
from src.lob_engine.execution_sim import ExecutionSimulator
from src.lob_engine.walk_forward import WalkForwardBacktester
from src.rl_execution.lob_env import LOBExecutionEnv
from src.rl_execution.q_agent import TabularQLearningAgent
from src.rl_execution.benchmarks import (
    evaluate_policy_episodes,
    twap_policy,
    vwap_policy,
    passive_close_policy,
)
from src.game_theory.kyle_model import (
    solve_kyle_single_period,
    simulate_kyle_sequential_auction,
)
from src.game_theory.auctions import (
    simulate_independent_private_value_auctions,
    simulate_walrasian_double_auction,
)
from src.game_theory.lp_equilibrium import solve_zero_sum_game_lp


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "legend.fontsize": 8.5,
            "figure.dpi": 300,
        }
    )


def rolling_mean(arr: np.ndarray, window: int = 150) -> np.ndarray:
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(arr, kernel, mode="valid")


def main() -> None:
    configure_plot_style()
    figures_dir = ROOT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("1. FIFO LIMIT ORDER BOOK & 1-TICK LATENCY SIMULATION")
    print("=" * 78)
    lob, lob_traj = generate_poisson_lob_stream(n_ticks=2500, seed=42)
    print(
        f"Simulated {len(lob_traj['mid_price']):,} LOB ticks | "
        f"Total Matched Fills: {len(lob.fills):,} | "
        f"Mean Spread: {np.mean(lob_traj['spread']) * 100:.2f} cents | "
        f"Imbalance Std: {np.std(lob_traj['imbalance']):.3f}"
    )

    print("\n" + "=" * 78)
    print("2. TABULAR Q-LEARNING OPTIMAL EXECUTION (10,000 EPISODES)")
    print("=" * 78)
    env = LOBExecutionEnv(horizon_steps=20, initial_inventory=1000.0, seed=42)
    agent = TabularQLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        lr=0.14,
        gamma=0.97,
        eps_start=1.0,
        eps_end=0.02,
        eps_decay_episodes=7500,
        seed=42,
    )
    train_metrics = agent.train(env, n_episodes=10000, seed_offset=1000)

    rl_eval = evaluate_policy_episodes(
        env, agent.greedy_policy, "Tabular Q-Learning (MDP)", n_eval_episodes=500, base_seed=50000
    )
    twap_eval = evaluate_policy_episodes(
        env, twap_policy, "Static TWAP Baseline", n_eval_episodes=500, base_seed=50000
    )
    vwap_eval = evaluate_policy_episodes(
        env, vwap_policy, "Static VWAP Baseline", n_eval_episodes=500, base_seed=50000
    )
    pass_eval = evaluate_policy_episodes(
        env, passive_close_policy, "Passive / Terminal Sweep", n_eval_episodes=500, base_seed=50000
    )

    shortfall_diff_twap = twap_eval.mean_shortfall_bps - rl_eval.mean_shortfall_bps
    var_reduction_pct = (
        (twap_eval.return_variance - rl_eval.return_variance)
        / max(twap_eval.return_variance, 1e-12)
        * 100.0
    )
    dd_compression_bps = twap_eval.max_inventory_drawdown_bps - rl_eval.max_inventory_drawdown_bps

    for summary in (rl_eval, twap_eval, vwap_eval, pass_eval):
        print(
            f"  {summary.policy_name:<26} | "
            f"Mean IS: {summary.mean_shortfall_bps:6.2f} bps | "
            f"Sharpe: {summary.sharpe_ratio:5.2f} | "
            f"Max Inv DD: {summary.max_inventory_drawdown_bps:6.1f} bps"
        )
    print(
        f"  -> Shortfall Reduction vs TWAP: {shortfall_diff_twap:+.2f} bps | "
        f"Return Variance Reduction: {var_reduction_pct:+.1f}% | "
        f"Max Drawdown Compression: {dd_compression_bps:+.1f} bps"
    )

    # --- Figure 1: RL Convergence & Shortfall Distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    episodes_x = np.arange(150, 10001)
    axes[0].plot(
        episodes_x,
        rolling_mean(train_metrics.bellman_residuals, 150),
        color="#1b365d",
        lw=1.8,
        label="Bellman Residual (150-ep MA)",
    )
    ax0_twin = axes[0].twinx()
    ax0_twin.plot(
        episodes_x,
        rolling_mean(train_metrics.episode_rewards, 150),
        color="#008080",
        lw=1.6,
        linestyle="-.",
        label="Episode Reward (150-ep MA)",
    )
    axes[0].set_title("A. Bellman Residual Convergence (10,000 Episodes)")
    axes[0].set_xlabel("Training Episode")
    axes[0].set_ylabel("Max Bellman Residual $|\\Delta Q(s,a)|$", color="#1b365d")
    ax0_twin.set_ylabel("Cumulative MDP Episode Reward", color="#008080")
    ax0_twin.grid(False)

    bins = np.linspace(-4.0, 58.0, 44)
    axes[1].hist(
        rl_eval.shortfall_distribution_bps,
        bins=bins,
        alpha=0.68,
        color="#1b365d",
        density=True,
        label=f"Q-Learning (Mean: {rl_eval.mean_shortfall_bps:.1f} bps)",
    )
    axes[1].hist(
        twap_eval.shortfall_distribution_bps,
        bins=bins,
        alpha=0.52,
        color="#d95f02",
        density=True,
        label=f"TWAP (Mean: {twap_eval.mean_shortfall_bps:.1f} bps)",
    )
    axes[1].hist(
        pass_eval.shortfall_distribution_bps,
        bins=bins,
        alpha=0.42,
        color="#7570b3",
        density=True,
        label=f"Passive Sweep (Mean: {pass_eval.mean_shortfall_bps:.1f} bps)",
    )
    axes[1].set_title("B. Out-of-Sample Implementation Shortfall Density")
    axes[1].set_xlabel("Implementation Shortfall (bps)")
    axes[1].set_ylabel("Empirical Density")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig1_rl_execution_convergence.png")
    plt.close(fig)

    # --- Figure 2: Inventory Liquidation Curves & Policy Comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    steps_grid = np.arange(env.horizon_steps + 1)
    axes[0].plot(
        steps_grid,
        rl_eval.mean_inventory_path * 100.0,
        marker="o",
        ms=3.8,
        lw=2.0,
        color="#1b365d",
        label="Tabular Q-Learning (State-Adaptive)",
    )
    axes[0].plot(
        steps_grid,
        twap_eval.mean_inventory_path * 100.0,
        linestyle="--",
        lw=1.8,
        color="#d95f02",
        label="Static TWAP (Linear Schedule)",
    )
    axes[0].plot(
        steps_grid,
        vwap_eval.mean_inventory_path * 100.0,
        linestyle="-.",
        lw=1.8,
        color="#2ca02c",
        label="Static VWAP (U-Shape Schedule)",
    )
    axes[0].plot(
        steps_grid,
        pass_eval.mean_inventory_path * 100.0,
        linestyle=":",
        lw=1.8,
        color="#7570b3",
        label="Passive + Terminal Sweep",
    )
    axes[0].set_title("A. Mean Remaining Inventory Trajectory ($q_t / N_0$)")
    axes[0].set_xlabel("Decision Epoch $t$")
    axes[0].set_ylabel("Remaining Inventory (%)")
    axes[0].legend(loc="upper right")

    names = ["Q-Learning\n(MDP)", "Static\nVWAP", "Static\nTWAP", "Passive\nSweep"]
    means_is = [
        rl_eval.mean_shortfall_bps,
        vwap_eval.mean_shortfall_bps,
        twap_eval.mean_shortfall_bps,
        pass_eval.mean_shortfall_bps,
    ]
    colors_bar = ["#1b365d", "#2ca02c", "#d95f02", "#7570b3"]
    bars = axes[1].bar(names, means_is, color=colors_bar, width=0.52, edgecolor="black", lw=0.7)
    for bar, val in zip(bars, means_is):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 0.6,
            f"{val:.1f} bps",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    axes[1].set_ylim(0, max(means_is) * 1.18)
    axes[1].set_title("B. Out-of-Sample Implementation Shortfall Comparison")
    axes[1].set_ylabel("Mean Implementation Shortfall (bps)")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig2_inventory_and_shortfall.png")
    plt.close(fig)

    print("\n" + "=" * 78)
    print("3. KYLE (1985) PRICE IMPACT & AUCTION REVENUE EQUIVALENCE")
    print("=" * 78)
    kyle_1p = solve_kyle_single_period(p0=100.0, sigma0_var=4.0, noise_vol=10.0)
    kyle_traj = simulate_kyle_sequential_auction(
        n_steps=25, p0=100.0, true_value=104.0, sigma0_var=4.0, noise_vol=10.0, n_paths=300, seed=42
    )
    auc_5 = simulate_independent_private_value_auctions(n_bidders=5, n_trials=100_000, seed=42)
    da_res = simulate_walrasian_double_auction(n_buyers=50, n_sellers=50, seed=42)
    lp_res = solve_zero_sum_game_lp(
        np.array([[3.0, -1.0, 2.0], [-2.0, 4.0, 1.0], [1.0, 2.0, -3.0]])
    )
    print(
        f"  Kyle Single-Period: beta={kyle_1p.beta:.3f}, lambda={kyle_1p.lambda_impact:.3f}, "
        f"Posterior Var={kyle_1p.posterior_var:.2f} (50% reduction)"
    )
    print(
        f"  Auction Revenue Equivalence (N=5): Theo={auc_5.theoretical_expected_revenue:.4f} | "
        f"First-Price={auc_5.first_price_mean_revenue:.4f} | "
        f"Vickrey={auc_5.second_price_mean_revenue:.4f} "
        f"(Discrepancy: {auc_5.relative_discrepancy_pct:.2f}%)"
    )
    print(
        f"  Walrasian Double Auction: Clearing Price={da_res['clearing_price']:.2f} | "
        f"Matched Volume={da_res['volume']:.0f} lots | "
        f"LP Minimax Duality Gap: {lp_res.duality_gap:.2e}"
    )

    # --- Figure 3: Kyle Sequential Price Discovery & Revenue Equivalence ---
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    t_grid = np.arange(kyle_traj.n_steps + 1)
    mean_p = np.mean(kyle_traj.mean_price_paths, axis=0)
    p10 = np.percentile(kyle_traj.mean_price_paths, 15, axis=0)
    p90 = np.percentile(kyle_traj.mean_price_paths, 85, axis=0)

    axes[0].plot(t_grid, mean_p, color="#1b365d", lw=2.0, label="Mean Market Maker Price $E[P_t]$")
    axes[0].fill_between(
        t_grid, p10, p90, color="#1b365d", alpha=0.18, label="15th-85th Percentile Band"
    )
    axes[0].axhline(
        kyle_traj.true_value,
        color="#d95f02",
        linestyle="--",
        lw=1.6,
        label=f"Fundamental Value $v = {kyle_traj.true_value:.1f}$",
    )
    ax0_var = axes[0].twinx()
    ax0_var.plot(
        t_grid,
        kyle_traj.sigma_var_path,
        color="#2ca02c",
        linestyle="-.",
        lw=1.6,
        label="Residual Variance $\\Sigma_t$",
    )
    axes[0].set_title("A. Kyle (1985) Sequential Price Discovery & $\\Sigma_t$ Decay")
    axes[0].set_xlabel("Sequential Auction Step $t$")
    axes[0].set_ylabel("Market Price $P_t$", color="#1b365d")
    ax0_var.set_ylabel("Posterior Variance $\\Sigma_t = \\mathrm{Var}(v \\mid y_{1:t})$", color="#2ca02c")
    ax0_var.grid(False)
    axes[0].legend(loc="upper left")

    bidders_list = list(range(2, 11))
    theo_revs, fp_revs, sp_revs = [], [], []
    for n_b in bidders_list:
        r_sim = simulate_independent_private_value_auctions(n_bidders=n_b, n_trials=40_000, seed=100 + n_b)
        theo_revs.append(r_sim.theoretical_expected_revenue)
        fp_revs.append(r_sim.first_price_mean_revenue)
        sp_revs.append(r_sim.second_price_mean_revenue)

    axes[1].plot(
        bidders_list, theo_revs, color="black", lw=1.8, linestyle="-", label="Theoretical $(N-1)/(N+1)$"
    )
    axes[1].scatter(
        bidders_list, fp_revs, color="#1b365d", marker="o", s=42, zorder=4, label="First-Price Sealed BNE"
    )
    axes[1].scatter(
        bidders_list, sp_revs, color="#d95f02", marker="x", s=48, zorder=5, label="Second-Price (Vickrey)"
    )
    axes[1].set_title("B. Auction Revenue Equivalence Verification ($N = 2 \\dots 10$)")
    axes[1].set_xlabel("Number of Bidders $N$")
    axes[1].set_ylabel("Expected Seller Revenue $E[R]$")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig3_kyle_and_auctions.png")
    plt.close(fig)

    print("\n" + "=" * 78)
    print("4. ANCHORED WALK-FORWARD REGIME BACKTEST (SQUARE-ROOT IMPACT & 15 BPS FEES)")
    print("=" * 78)
    ohlcv = generate_regime_switching_ohlcv(n_bars=1825, seed=2026)
    sim = ExecutionSimulator(fee_bps=15.0, impact_coeff=0.35, half_spread_bps=2.5)
    wf_engine = WalkForwardBacktester(sim, lookback_fast=14, lookback_slow=42, vol_window=25)
    wf_res = wf_engine.run_anchored_walk_forward(
        ohlcv["returns"], ohlcv["volumes"], initial_train_size=600, val_size=200, test_fold_size=250
    )

    sr = wf_res.strategy_report
    br = wf_res.benchmark_report
    print(
        f"  Walk-Forward Strategy | Net Sharpe: {sr.sharpe_ratio:.2f} | "
        f"Sortino: {sr.sortino_ratio:.2f} | Max DD: {sr.max_drawdown * 100:.1f}% | "
        f"Win Rate: {sr.win_rate * 100:.1f}%"
    )
    print(
        f"  Buy & Hold Benchmark  | Net Sharpe: {br.sharpe_ratio:.2f} | "
        f"Sortino: {br.sortino_ratio:.2f} | Max DD: {br.max_drawdown * 100:.1f}%"
    )
    print(f"  -> Stress-Regime Tail Drawdown Reduction: {wf_res.stress_regime_dd_reduction_pct:.1f}%")

    # --- Figure 4: Walk-Forward Equity & Drawdown Profile ---
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    oos_days = np.arange(len(sr.equity_curve))
    axes[0].plot(
        oos_days,
        sr.equity_curve,
        color="#1b365d",
        lw=1.9,
        label=f"Walk-Forward Regime Strategy (Sharpe: {sr.sharpe_ratio:.2f})",
    )
    axes[0].plot(
        oos_days,
        br.equity_curve,
        color="#8c564b",
        lw=1.4,
         alpha=0.8,
        linestyle="--",
        label=f"Passive Benchmark (Sharpe: {br.sharpe_ratio:.2f})",
    )
    axes[0].set_title("A. Out-of-Sample Cumulative Equity (Net of 15 bps + $\\sqrt{Q/V}$ Impact)")
    axes[0].set_xlabel("Out-of-Sample Trading Days")
    axes[0].set_ylabel("Normalized Equity ($1.00 Initial)")
    axes[0].legend(loc="upper left")

    strat_dd = (sr.equity_curve - np.maximum.accumulate(sr.equity_curve)) / np.maximum.accumulate(
        sr.equity_curve
    )
    bench_dd = (br.equity_curve - np.maximum.accumulate(br.equity_curve)) / np.maximum.accumulate(
        br.equity_curve
    )
    axes[1].fill_between(
        oos_days,
        strat_dd * 100.0,
        0.0,
        color="#1b365d",
        alpha=0.45,
        label=f"Regime-Filtered DD (Max: {sr.max_drawdown * 100:.1f}%)",
    )
    axes[1].plot(
        oos_days,
        bench_dd * 100.0,
        color="#d62728",
        lw=1.3,
        linestyle="--",
        label=f"Unfiltered Benchmark DD (Max: {br.max_drawdown * 100:.1f}%)",
    )
    axes[1].set_title(
        f"B. Underwater Drawdown Profile (-{wf_res.stress_regime_dd_reduction_pct:.1f}% Tail DD Cut)"
    )
    axes[1].set_xlabel("Out-of-Sample Trading Days")
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig4_walkforward_regime_backtest.png")
    plt.close(fig)

    print("\nAll 4 diagnostic figures saved to:", figures_dir)


if __name__ == "__main__":
    main()

