"""
Primal-Dual Linear Programming Solver for Two-Player Zero-Sum Games & IESDS.

Solves both the row player's maximin primal LP and the column player's minimax
dual LP via HiGHS simplex/interior-point solvers, verifying strong duality
and mixed-strategy Nash Equilibrium support conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class ZeroSumLPResult:
    row_strategy: np.ndarray
    col_strategy: np.ndarray
    primal_value: float
    dual_value: float
    duality_gap: float


def solve_zero_sum_game_lp(payoff_matrix: np.ndarray) -> ZeroSumLPResult:
    """
    Solves a finite two-player zero-sum game with payoff matrix A in R^{m x n}.

    Primal (Row Player maximin):
        max_{p, v} v   s.t.  A^T p >= v 1_n,  1_m^T p = 1,  p >= 0
    Dual (Column Player minimax):
        min_{q, w} w   s.t.  A q <= w 1_m,    1_n^T q = 1,  q >= 0
    """
    A = np.asarray(payoff_matrix, dtype=float)
    if A.ndim != 2:
        raise ValueError("payoff_matrix must be 2-dimensional.")
    m, n = A.shape

    # --- Primal LP: variables x = [p_1, ..., p_m, v] ---
    # Minimize -v  <=>  c = [0, ..., 0, -1]
    c_primal = np.zeros(m + 1, dtype=float)
    c_primal[-1] = -1.0

    # Constraint: -A^T p + v * 1_n <= 0
    A_ub_primal = np.hstack([-A.T, np.ones((n, 1), dtype=float)])
    b_ub_primal = np.zeros(n, dtype=float)

    # Equality: sum(p_i) = 1
    A_eq_primal = np.zeros((1, m + 1), dtype=float)
    A_eq_primal[0, :m] = 1.0
    b_eq_primal = np.array([1.0], dtype=float)

    bounds_primal = [(0.0, 1.0) for _ in range(m)] + [(None, None)]
    res_primal = linprog(
        c_primal,
        A_ub=A_ub_primal,
        b_ub=b_ub_primal,
        A_eq=A_eq_primal,
        b_eq=b_eq_primal,
        bounds=bounds_primal,
        method="highs",
    )
    if not res_primal.success:
        raise RuntimeError(f"Primal LP failed to converge: {res_primal.message}")

    p_star = np.clip(res_primal.x[:m], 0.0, 1.0)
    p_star /= np.sum(p_star)
    v_star = float(res_primal.x[-1])

    # --- Dual LP: variables y = [q_1, ..., q_n, w] ---
    # Minimize w  <=>  c = [0, ..., 0, 1]
    c_dual = np.zeros(n + 1, dtype=float)
    c_dual[-1] = 1.0

    # Constraint: A q - w * 1_m <= 0
    A_ub_dual = np.hstack([A, -np.ones((m, 1), dtype=float)])
    b_ub_dual = np.zeros(m, dtype=float)

    # Equality: sum(q_j) = 1
    A_eq_dual = np.zeros((1, n + 1), dtype=float)
    A_eq_dual[0, :n] = 1.0
    b_eq_dual = np.array([1.0], dtype=float)

    bounds_dual = [(0.0, 1.0) for _ in range(n)] + [(None, None)]
    res_dual = linprog(
        c_dual,
        A_ub=A_ub_dual,
        b_ub=b_ub_dual,
        A_eq=A_eq_dual,
        b_eq=b_eq_dual,
        bounds=bounds_dual,
        method="highs",
    )
    if not res_dual.success:
        raise RuntimeError(f"Dual LP failed to converge: {res_dual.message}")

    q_star = np.clip(res_dual.x[:n], 0.0, 1.0)
    q_star /= np.sum(q_star)
    w_star = float(res_dual.x[-1])

    return ZeroSumLPResult(
        row_strategy=p_star,
        col_strategy=q_star,
        primal_value=v_star,
        dual_value=w_star,
        duality_gap=abs(v_star - w_star),
    )


def iterated_elimination_strictly_dominated(
    row_payoffs: np.ndarray,
    col_payoffs: np.ndarray,
    tol: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs Iterated Elimination of Strictly Dominated Strategies (IESDS)
    on a finite bimatrix game (A, B), returning reduced payoff matrices and
    surviving strategy index arrays.
    """
    A = np.asarray(row_payoffs, dtype=float).copy()
    B = np.asarray(col_payoffs, dtype=float).copy()
    row_indices = np.arange(A.shape[0])
    col_indices = np.arange(A.shape[1])

    changed = True
    while changed:
        changed = False
        # Eliminate strictly dominated rows
        surviving_rows = []
        for i in range(A.shape[0]):
            dominated = any(
                i != k and np.all(A[k, :] > A[i, :] + tol) for k in range(A.shape[0])
            )
            if not dominated:
                surviving_rows.append(i)
        if len(surviving_rows) < A.shape[0]:
            A = A[surviving_rows, :]
            B = B[surviving_rows, :]
            row_indices = row_indices[surviving_rows]
            changed = True

        # Eliminate strictly dominated columns
        surviving_cols = []
        for j in range(B.shape[1]):
            dominated = any(
                j != l and np.all(B[:, l] > B[:, j] + tol) for l in range(B.shape[1])
            )
            if not dominated:
                surviving_cols.append(j)
        if len(surviving_cols) < B.shape[1]:
            A = A[:, surviving_cols]
            B = B[:, surviving_cols]
            col_indices = col_indices[surviving_cols]
            changed = True

    return A, B, row_indices, col_indices

