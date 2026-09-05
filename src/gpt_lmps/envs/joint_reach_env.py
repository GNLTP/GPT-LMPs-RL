from __future__ import annotations

from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gpt_lmps.control.reach import (
    NUM_DISCRETE_ACTIONS,
    REACH_OBSERVATION_SIZE,
    apply_discrete_joint_action,
    reach_errors,
    reach_observation,
    shaped_reach_reward,
)
from gpt_lmps.envs.robot import PandaRobot
from gpt_lmps.envs.scene import TabletopScene


class JointReachEnv(gym.Env[np.ndarray, int]):
    """Goal-conditioned joint-space reaching task for Q-Learning and DQN.

    The policy chooses one of 15 discrete actions: hold or a positive/negative increment on one
    of the seven Franka joints. PyBullet supplies forward kinematics and contact dynamics; no
    inverse-kinematics solver is used inside ``step``.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 20}

    GRID_TARGETS = np.asarray(
        [
            [0.42, -0.16, 0.12],
            [0.42, 0.00, 0.18],
            [0.42, 0.16, 0.24],
            [0.52, -0.16, 0.24],
            [0.52, 0.00, 0.12],
            [0.52, 0.16, 0.18],
            [0.62, -0.16, 0.18],
            [0.62, 0.00, 0.24],
            [0.62, 0.16, 0.12],
        ],
        dtype=np.float32,
    )

    def __init__(
        self,
        *,
        target_mode: Literal["grid", "continuous"] = "continuous",
        render_mode: Literal["rgb_array", "human"] | None = None,
        max_episode_steps: int = 160,
        distance_threshold: float = 0.025,
        joint_delta: float = 0.035,
        initial_joint_noise: float = 0.04,
        simulation_steps_per_action: int = 12,
    ) -> None:
        super().__init__()
        if target_mode not in {"grid", "continuous"}:
            raise ValueError(f"Unknown target mode: {target_mode}")
        self.target_mode = target_mode
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.distance_threshold = distance_threshold
        self.joint_delta = joint_delta
        self.initial_joint_noise = initial_joint_noise
        self.simulation_steps_per_action = simulation_steps_per_action
        self.action_space = spaces.Discrete(NUM_DISCRETE_ACTIONS)
        observation_low = np.concatenate(
            [
                PandaRobot.LOWER_LIMITS,
                np.full(7, -10.0),
                np.full(3, -1.0),
                np.full(3, -np.pi),
            ]
        ).astype(np.float32)
        observation_high = np.concatenate(
            [
                PandaRobot.UPPER_LIMITS,
                np.full(7, 10.0),
                np.full(3, 1.0),
                np.full(3, np.pi),
            ]
        ).astype(np.float32)
        self.observation_space = spaces.Box(
            low=observation_low,
            high=observation_high,
            shape=(REACH_OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self.scene = TabletopScene(
            gui=render_mode == "human",
            realtime=render_mode == "human",
        )
        self.goal = np.zeros(3, dtype=np.float32)
        self.steps = 0
        self.previous_position_error = 0.0
        self.previous_orientation_error = 0.0

    @property
    def robot(self) -> PandaRobot:
        if self.scene.robot is None:
            raise RuntimeError("Call reset() before using the reach environment")
        return self.scene.robot

    def _sample_goal(self) -> np.ndarray:
        if self.target_mode == "grid":
            index = int(self.np_random.integers(0, len(self.GRID_TARGETS)))
            return self.GRID_TARGETS[index].copy()
        return np.asarray(
            [
                self.np_random.uniform(0.40, 0.64),
                self.np_random.uniform(-0.20, 0.20),
                self.np_random.uniform(0.10, 0.28),
            ],
            dtype=np.float32,
        )

    def _observation(self) -> np.ndarray:
        return reach_observation(self.robot, self.goal, self.robot.DOWN_ORIENTATION)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.scene.reset([])
        options = options or {}
        initial = np.asarray(PandaRobot.HOME_JOINTS, dtype=np.float32)
        noise_scale = float(options.get("initial_joint_noise", self.initial_joint_noise))
        initial += self.np_random.uniform(-noise_scale, noise_scale, size=7).astype(np.float32)
        initial = np.clip(initial, PandaRobot.LOWER_LIMITS, PandaRobot.UPPER_LIMITS)
        self.robot.reset(initial)
        self.scene.step_simulation(30)
        self.goal = np.asarray(options.get("goal", self._sample_goal()), dtype=np.float32)
        self.steps = 0
        observation = self._observation()
        self.previous_position_error, self.previous_orientation_error = reach_errors(observation)
        return observation, self._info(False, False)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        apply_discrete_joint_action(self.robot, int(action), joint_delta=self.joint_delta)
        self.scene.step_simulation(self.simulation_steps_per_action)
        self.steps += 1

        observation = self._observation()
        position_error, orientation_error = reach_errors(observation)
        success = position_error < self.distance_threshold
        collision = self.robot.has_self_collision()
        reward = shaped_reach_reward(
            previous_position_error=self.previous_position_error,
            position_error=position_error,
            previous_orientation_error=self.previous_orientation_error,
            orientation_error=orientation_error,
            joint_limit_cost=self.robot.joint_limit_cost(),
            collision=collision,
            success=success,
        )
        self.previous_position_error = position_error
        self.previous_orientation_error = orientation_error
        terminated = bool(success)
        truncated = self.steps >= self.max_episode_steps
        return observation, reward, terminated, truncated, self._info(success, collision)

    def _info(self, success: bool, collision: bool) -> dict[str, Any]:
        observation = self._observation()
        position_error, orientation_error = reach_errors(observation)
        return {
            "is_success": bool(success),
            "position_error": position_error,
            "orientation_error": orientation_error,
            "joint_limit_cost": self.robot.joint_limit_cost(),
            "self_collision": bool(collision),
            "target_mode": self.target_mode,
        }

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            return self.scene.observe()["oblique"].rgb
        return None

    def close(self) -> None:
        self.scene.close()
