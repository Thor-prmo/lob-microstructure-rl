"""
Tabular Q-Learning Agent with Decaying Epsilon-Greedy Exploration & Bellman Residual Tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from .lob_env import LOBExecutionEnv


@dataclass(frozen=True)
class TrainingMetrics:
    q_table: np.ndarray
    episode_rewards: np.ndarray
    bellman_residuals: np.ndarray
    shortfall_history_bps: np.ndarray


class TabularQLearningAgent:
    """
    Tabular Q-Learning solver for finite-horizon LOB optimal execution.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        lr: float = 0.12,
        gamma: float = 0.97,
        eps_start: float = 1.0,
        eps_end: float = 0.02,
        eps_decay_episodes: int = 7500,
        seed: int = 42,
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_episodes = eps_decay_episodes
        self.rng = np.random.default_rng(seed)
        self.q_table = np.zeros((n_states, n_actions), dtype=float)

    def epsilon_for_episode(self, episode_idx: int) -> float:
        progress = min(1.0, episode_idx / max(1, self.eps_decay_episodes))
        return float(
            self.eps_end + (self.eps_start - self.eps_end) * np.exp(-4.2 * progress)
        )

    def select_action(self, state: int, epsilon: float = 0.0) -> int:
        if epsilon > 0.0 and self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(self.q_table[state]))

    def train(
        self,
        env: LOBExecutionEnv,
        n_episodes: int = 10000,
        seed_offset: int = 1000,
    ) -> TrainingMetrics:
        rewards = np.zeros(n_episodes, dtype=float)
        residuals = np.zeros(n_episodes, dtype=float)
        shortfalls = np.zeros(n_episodes, dtype=float)

        for ep in range(n_episodes):
            state = env.reset(seed=seed_offset + ep)
            eps = self.epsilon_for_episode(ep)
            # Decay learning rate smoothly for Robbins-Monro convergence
            alpha = self.lr / (1.0 + 0.00045 * ep)

            ep_reward = 0.0
            max_q_delta = 0.0
            done = False

            while not done:
                action = self.select_action(state, epsilon=eps)
                next_state, reward, done, _ = env.step(action)

                best_next = 0.0 if done else float(np.max(self.q_table[next_state]))
                td_target = reward + self.gamma * best_next
                td_error = td_target - self.q_table[state, action]

                q_update = alpha * td_error
                self.q_table[state, action] += q_update
                max_q_delta = max(max_q_delta, abs(q_update))
                ep_reward += reward
                state = next_state

            rewards[ep] = ep_reward
            residuals[ep] = max_q_delta
            shortfalls[ep] = env.total_implementation_shortfall_bps()

        return TrainingMetrics(
            q_table=self.q_table.copy(),
            episode_rewards=rewards,
            bellman_residuals=residuals,
            shortfall_history_bps=shortfalls,
        )

    def greedy_policy(self, state: int, env: Optional[LOBExecutionEnv] = None) -> int:
        return int(np.argmax(self.q_table[state]))

