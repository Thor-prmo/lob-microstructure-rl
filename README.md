# Limit Order Book Microstructure, Optimal Execution & Strategic Equilibrium

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: Pytest](https://img.shields.io/badge/tests-pytest%20passing-brightgreen.svg)]()

A unified research codebase implementing **price-time priority limit order book (LOB) simulation**, **finite-horizon Markov Decision Process (MDP) optimal trade execution**, **square-root market impact backtesting**, and **strategic market microstructure equilibria**.

📄 **[Read the 2-Page Technical Note (`Technical_Note.pdf`)](Technical_Note.pdf)** | **[LaTeX Source (`paper/microstructure_execution_note.tex`)](paper/microstructure_execution_note.tex)**

---

## Visual Overview

| Bellman Convergence & Shortfall Distribution | Kyle (1985) Price Impact & Auction Revenue Equivalence |
| :---: | :---: |
| ![RL Execution](figures/fig1_rl_execution_convergence.png) | ![Game Theory](figures/fig3_kyle_and_auctions.png) |
| **Execution Trajectory vs. TWAP / VWAP** | **Walk-Forward Regime Filtering & Drawdown Profile** |
| ![Inventory Trajectory](figures/fig2_inventory_and_shortfall.png) | ![Walk-Forward Crypto](figures/fig4_walkforward_regime_backtest.png) |

---

## Mathematical Formulation

### 1. Price-Time Priority LOB & Square-Root Impact (`src/lob_engine/`)

Orders at price level $p$ are maintained in strict First-In-First-Out (FIFO) queues $\mathcal{Q}_t(p) = (q_1, q_2, \dots, q_k)$. To prevent instantaneous fill illusions in high-frequency backtests, an order submitted at timestamp $t_i$ enters an execution buffer and can only match against the book state at $t_{i+1}$ (1-tick latency). For aggregate meta-orders of size $Q$ executed over volume horizon $V_t$ with rolling volatility $\sigma_t$, macroscopic price impact follows the square-root law alongside proportional exchange fee $\phi$:

$$\Delta P_{\text{impact}}(Q) = Y \cdot \sigma_t \sqrt{\frac{\lvert Q \rvert}{V_t}}, \qquad \text{Implementation Shortfall (bps)} = \frac{\sum_{t=1}^T n_t p_t^{\text{exec}} - N_0 p_0}{N_0 p_0} \times 10^4 + \phi$$

---

### 2. Sequential Execution as a Finite-Horizon MDP (`src/rl_execution/`)

Liquidating an initial inventory $N_0$ over $T$ discrete decision epochs is formulated as an MDP $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$. The state vector $s_t \in \mathcal{S}$ discretizes queue imbalance $I_t \in [-1, 1]$, normalized bid-ask spread $S_t$, realized volatility $\sigma_t$, and remaining inventory ratio $q_t / N_0$:

$$I_t = \frac{V_t^{\text{bid}} - V_t^{\text{ask}}}{V_t^{\text{bid}} + V_t^{\text{ask}}}, \qquad R_t(s_t, a_t) = -\left( \Delta c_t(a_t) + \lambda_{\text{inv}} \sigma_t^2 q_t^2 \right) + \eta_{\text{dsr}} \cdot \Delta \mathrm{DSR}_t$$

where $\Delta c_t(a_t)$ is the single-step implementation shortfall relative to arrival mid-price, $\lambda_{\text{inv}} \sigma_t^2 q_t^2$ is the Almgren–Chriss quadratic inventory variance penalty, and $\Delta \mathrm{DSR}_t$ is the online Differential Sharpe Ratio increment (Moody & Saffell, 2001).

---

### 3. Informed Order Flow & Mechanism Design (`src/game_theory/`)

In the sequential Kyle (1985) auction, an insider observing fundamental value $v \sim \mathcal{N}(p_0, \Sigma_0)$ submits order $x_t = \beta_t (v - p_{t-1})\,\Delta t$ alongside noise trader flow $\Delta u_t \sim \mathcal{N}(0, \sigma_u^2 \Delta t)$. Competitive risk-neutral market makers set linear pricing rules $p_t = p_{t-1} + \lambda_t (\Delta x_t + \Delta u_t)$, where the single-period equilibrium satisfies:

$$\beta = \frac{\sigma_u}{\sqrt{\Sigma_0}}, \qquad \lambda = \frac{\sqrt{\Sigma_0}}{2\sigma_u}, \qquad \Sigma_1 = \mathrm{Var}(v \mid y) = \frac{1}{2}\Sigma_0$$

For finite zero-sum games with payoff matrix $A \in \mathbb{R}^{m \times n}$, mixed-strategy Nash equilibria are computed via primal-dual Linear Programming following iterated elimination of strictly dominated strategies:

$$\max_{p \in \Delta^m} \; v \quad \text{s.t.} \quad A^\top p \ge v \mathbf{1}_n \qquad \Longleftrightarrow \qquad \min_{q \in \Delta^n} \; w \quad \text{s.t.} \quad A q \le w \mathbf{1}_m$$

---

## Repository Structure

```text
lob-microstructure-rl/
├── README.md
├── Technical_Note.pdf                      # Compiled 2-page two-column LaTeX note
├── Makefile
├── pyproject.toml
├── requirements.txt
├── paper/
│   └── microstructure_execution_note.tex   # LaTeX source for Technical_Note.pdf
├── src/
│   ├── lob_engine/
│   │   ├── order_book.py                   # FIFO price-time priority L2/L3 book with 1-tick latency
│   │   ├── execution_sim.py                # Square-root market impact & proportional fee simulator
│   │   ├── walk_forward.py                 # Anchored walk-forward regime backtester (zero lookahead)
│   │   └── synthetic_data.py               # Multi-regime OHLCV & order flow generator
│   ├── rl_execution/
│   │   ├── lob_env.py                      # Finite-horizon MDP execution environment
│   │   ├── q_agent.py                      # Tabular Q-learning with Bellman residual tracking
│   │   └── benchmarks.py                   # Static TWAP, VWAP, and passive baseline policies
│   └── game_theory/
│       ├── kyle_model.py                   # Single-period & multi-period sequential Kyle (1985) solver
│       ├── auctions.py                     # First-Price, Vickrey, and Walrasian Double Auction MC
│       └── lp_equilibrium.py               # Iterated dominance (IESDS) & Primal-Dual LP minimax solver
├── scripts/
│   └── run_experiments.py                  # Reproduces all benchmarks and saves 4 figures
├── figures/                                # Auto-generated 300-DPI diagnostic plots
└── tests/
    ├── test_causality_and_fifo.py          # Proves zero lookahead leakage & strict FIFO lot matching
    └── test_game_theory_duality.py         # Verifies LP strong duality & Revenue Equivalence bounds
```

---

## Empirical & Verification Summary

| Module / Experiment | Key Metric | Benchmark / Theoretical Baseline | Measured Result |
| :--- | :--- | :--- | :--- |
| **FIFO Limit Order Book** | Lookahead Perturbation Sensitivity | Exact $0.0$ state mutation at $t$ | `max_diff = 0.00e+00` (Verified) |
| **Tabular Q-Learning Execution** | Out-of-Sample Implementation Shortfall | Static TWAP / VWAP Baselines | **-14.4 bps** reduction vs. TWAP |
| **RL Inventory Control** | Out-of-Sample Cost Variance & Drawdown | Passive / Static Baselines | **31.8%** variance cut, **-820 bps** DD cut |
| **Walk-Forward Regime Strategy** | 2022 Stress-Regime Tail Drawdown | Passive Buy & Hold Benchmark | **-38.5%** tail drawdown cut (Sharpe 1.48) |
| **Vickrey vs. First-Price Auction** | Empirical Revenue Discrepancy ($N=5$) | Theoretical $\mathbb{E}[R] = \frac{N-1}{N+1} = 0.6667$ | **0.24%** relative discrepancy ($<0.8\%$) |
| **LP Minimax Equilibrium** | Primal-Dual Duality Gap $\lvert v^* - w^* \rvert$ | Strong Duality ($0.0$) | `< 1.0e-11` |

---

## Quickstart & Reproducibility

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run mathematical invariant & zero-lookahead unit tests
pytest tests/

# 3. Reproduce all numerical experiments and generate figures/
python3 scripts/run_experiments.py

# 4. Compile the 2-page technical note PDF
make paper && cp paper/microstructure_execution_note.pdf Technical_Note.pdf
```

---

## Author

**Yash Kothari**  
Department of Electrical Engineering (Minor in Mathematics), IIT Bombay (`25b1309@iitb.ac.in`)
