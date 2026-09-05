from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from gpt_lmps.control.base import MotionResult, SimulationStepper, as_target
from gpt_lmps.control.reach import (
    apply_discrete_joint_action,
    reach_errors,
    reach_observation,
)
from gpt_lmps.envs.robot import PandaRobot
from gpt_lmps.rl.q_learning import StateDiscretizer, TabularQAgent


class _DiscreteValueController:
    name = "value_based"

    def __init__(
        self,
        policy: Callable[[np.ndarray], int],
        *,
        max_steps: int = 180,
        joint_delta: float = 0.035,
        simulation_steps_per_action: int = 12,
    ) -> None:
        self.policy = policy
        self.max_steps = max_steps
        self.joint_delta = joint_delta
        self.simulation_steps_per_action = simulation_steps_per_action

    def move_to(
        self,
        robot: PandaRobot,
        target_position: Sequence[float],
        *,
        step_simulation: SimulationStepper,
        target_orientation: Sequence[float] | None = None,
        tolerance: float = 0.025,
    ) -> MotionResult:
        target = as_target(target_position)
        observation = reach_observation(robot, target, target_orientation)
        position_error, orientation_error = reach_errors(observation)
        if position_error < tolerance:
            return MotionResult(True, 0, position_error, orientation_error, "target_reached")

        for step in range(1, self.max_steps + 1):
            action = int(self.policy(observation))
            apply_discrete_joint_action(robot, action, joint_delta=self.joint_delta)
            step_simulation(self.simulation_steps_per_action)
            observation = reach_observation(robot, target, target_orientation)
            position_error, orientation_error = reach_errors(observation)
            if position_error < tolerance:
                return MotionResult(
                    True,
                    step,
                    position_error,
                    orientation_error,
                    "target_reached",
                )
        return MotionResult(
            False,
            self.max_steps,
            position_error,
            orientation_error,
            "step_limit",
        )


class QLearningController(_DiscreteValueController):
    """Greedy controller backed by a sparse tabular Q-function."""

    name = "q_learning"

    def __init__(self, checkpoint: str | Path, **kwargs: object) -> None:
        agent = TabularQAgent.load(checkpoint)
        discretizer = StateDiscretizer()

        def policy(observation: np.ndarray) -> int:
            return agent.greedy_action(discretizer.encode(observation))

        super().__init__(policy, **kwargs)


class DQNController(_DiscreteValueController):
    """Greedy controller backed by a Stable-Baselines3 DQN checkpoint."""

    name = "dqn"

    def __init__(self, checkpoint: str | Path, **kwargs: object) -> None:
        try:
            from stable_baselines3 import DQN
        except ImportError as exc:
            raise ImportError("Install the DQN dependencies with: pip install -e '.[rl]'") from exc
        model = DQN.load(str(checkpoint))

        def policy(observation: np.ndarray) -> int:
            action, _ = model.predict(observation, deterministic=True)
            return int(np.asarray(action).item())

        super().__init__(policy, **kwargs)
