from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

NUM_DISCRETE_ACTIONS = 15
JOINT_LOWER_LIMITS = np.array([-2.90, -1.76, -2.90, -3.07, -2.90, -0.02, -2.90])
JOINT_UPPER_LIMITS = np.array([2.90, 1.76, 2.90, -0.07, 2.90, 3.75, 2.90])

DiscreteState = tuple[int, ...]


class StateDiscretizer:
    """Quantize robot configuration and pose error for sparse tabular Q-Learning."""

    def __init__(
        self,
        *,
        joint_bins: int = 5,
        position_bins: int = 9,
        orientation_bins: int = 5,
    ) -> None:
        self.joint_bins = joint_bins
        self.position_bins = position_bins
        self.orientation_bins = orientation_bins

    @staticmethod
    def _uniform_bins(
        values: np.ndarray,
        low: np.ndarray,
        high: np.ndarray,
        count: int,
    ) -> list[int]:
        normalized = np.clip((values - low) / (high - low), 0.0, 1.0 - 1e-7)
        return np.floor(normalized * count).astype(np.int16).tolist()

    def encode(self, observation: np.ndarray) -> DiscreteState:
        values = np.asarray(observation, dtype=np.float32)
        if values.shape != (20,):
            raise ValueError(f"Expected a 20-D reach observation, got {values.shape}")
        joint_state = self._uniform_bins(
            values[:7],
            JOINT_LOWER_LIMITS,
            JOINT_UPPER_LIMITS,
            self.joint_bins,
        )
        position_state = self._uniform_bins(
            values[14:17],
            np.full(3, -0.40),
            np.full(3, 0.40),
            self.position_bins,
        )
        orientation_state = self._uniform_bins(
            values[17:20],
            np.full(3, -np.pi),
            np.full(3, np.pi),
            self.orientation_bins,
        )
        return tuple(joint_state + position_state + orientation_state)


class TabularQAgent:
    """Sparse Q-table with epsilon-greedy exploration."""

    def __init__(self, n_actions: int = NUM_DISCRETE_ACTIONS) -> None:
        self.n_actions = n_actions
        self.q_values: defaultdict[DiscreteState, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float32)
        )

    def action(
        self,
        state: DiscreteState,
        *,
        epsilon: float,
        rng: np.random.Generator,
    ) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(0, self.n_actions))
        values = self.q_values[state]
        best = np.flatnonzero(values == values.max())
        return int(rng.choice(best))

    def greedy_action(self, state: DiscreteState) -> int:
        return int(np.argmax(self.q_values[state]))

    def update(
        self,
        state: DiscreteState,
        action: int,
        reward: float,
        next_state: DiscreteState,
        *,
        learning_rate: float,
        discount: float,
        terminal: bool,
    ) -> None:
        future = 0.0 if terminal else float(np.max(self.q_values[next_state]))
        target = reward + discount * future
        current = float(self.q_values[state][action])
        self.q_values[state][action] = current + learning_rate * (target - current)

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if self.q_values:
            states = np.asarray(list(self.q_values), dtype=np.int16)
            values = np.stack([self.q_values[state] for state in self.q_values]).astype(np.float32)
        else:
            states = np.empty((0, 13), dtype=np.int16)
            values = np.empty((0, self.n_actions), dtype=np.float32)
        np.savez_compressed(output, states=states, q_values=values, n_actions=self.n_actions)
        return output

    @classmethod
    def load(cls, path: str | Path) -> TabularQAgent:
        checkpoint = Path(path)
        with np.load(checkpoint, allow_pickle=False) as data:
            agent = cls(int(np.asarray(data["n_actions"]).item()))
            for state, values in zip(data["states"], data["q_values"], strict=True):
                agent.q_values[tuple(int(item) for item in state)] = values.astype(np.float32)
        return agent
