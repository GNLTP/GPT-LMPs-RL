from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from gpt_lmps.envs.robot import PandaRobot

NUM_DISCRETE_ACTIONS = 15
REACH_OBSERVATION_SIZE = 20


def quaternion_error_vector(
    current_xyzw: Sequence[float],
    target_xyzw: Sequence[float],
) -> np.ndarray:
    """Return the shortest axis-angle rotation from current to target."""

    current = np.asarray(current_xyzw, dtype=np.float64)
    target = np.asarray(target_xyzw, dtype=np.float64)
    current /= np.linalg.norm(current)
    target /= np.linalg.norm(target)

    x1, y1, z1, w1 = target
    x2, y2, z2, w2 = (-current[0], -current[1], -current[2], current[3])
    error = np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )
    if error[3] < 0.0:
        error = -error
    vector_norm = np.linalg.norm(error[:3])
    if vector_norm < 1e-9:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(error[3], -1.0, 1.0))
    return (error[:3] / vector_norm * angle).astype(np.float32)


def reach_observation(
    robot: PandaRobot,
    target_position: Sequence[float],
    target_orientation: Sequence[float] | None = None,
) -> np.ndarray:
    """Build the shared 20-D observation used by Q-Learning and DQN."""

    target = np.asarray(target_position, dtype=np.float32)
    desired_orientation = np.asarray(
        target_orientation if target_orientation is not None else robot.DOWN_ORIENTATION,
        dtype=np.float32,
    )
    joint_positions = robot.joint_positions()
    joint_velocities = robot.joint_velocities()
    ee_position, ee_orientation = robot.ee_pose()
    position_error = target - ee_position
    orientation_error = quaternion_error_vector(ee_orientation, desired_orientation)
    return np.concatenate(
        [joint_positions, joint_velocities, position_error, orientation_error]
    ).astype(np.float32)


def reach_errors(observation: np.ndarray) -> tuple[float, float]:
    position_error = float(np.linalg.norm(observation[14:17]))
    orientation_error = float(np.linalg.norm(observation[17:20]))
    return position_error, orientation_error


def apply_discrete_joint_action(
    robot: PandaRobot,
    action: int,
    *,
    joint_delta: float,
) -> None:
    """Map 15 discrete actions to hold or +/- increments on one of seven joints."""

    if not 0 <= int(action) < NUM_DISCRETE_ACTIONS:
        raise ValueError(f"Action must be in [0, {NUM_DISCRETE_ACTIONS - 1}], got {action}")
    delta = np.zeros(7, dtype=np.float32)
    if action:
        joint_index = (int(action) - 1) // 2
        direction = 1.0 if int(action) % 2 == 1 else -1.0
        delta[joint_index] = direction * joint_delta
    robot.command_joint_delta(delta)


def shaped_reach_reward(
    *,
    previous_position_error: float,
    position_error: float,
    previous_orientation_error: float,
    orientation_error: float,
    joint_limit_cost: float,
    collision: bool,
    success: bool,
) -> float:
    """Progress-shaped value target shared by tabular Q-Learning and DQN."""

    position_progress = previous_position_error - position_error
    orientation_progress = previous_orientation_error - orientation_error
    reward = (
        8.0 * position_progress
        + 0.35 * orientation_progress
        - 0.35 * position_error
        - 0.015 * orientation_error
        - 0.10 * joint_limit_cost
        - 0.01
    )
    if collision:
        reward -= 0.25
    if success:
        reward += 10.0
    return float(reward)
