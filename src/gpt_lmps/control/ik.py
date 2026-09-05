from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from gpt_lmps.control.base import MotionResult, SimulationStepper, as_target
from gpt_lmps.control.reach import quaternion_error_vector
from gpt_lmps.envs.robot import PandaRobot


class IKController:
    """Analytic PyBullet inverse-kinematics baseline."""

    name = "ik"

    def __init__(self, max_steps: int = 360) -> None:
        self.max_steps = max_steps

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
        desired_orientation = np.asarray(
            target_orientation if target_orientation is not None else robot.DOWN_ORIENTATION,
            dtype=np.float32,
        )
        reached, steps = robot.move_ee(
            target,
            orientation=desired_orientation,
            tolerance=tolerance,
            max_steps=self.max_steps,
            step_simulation=step_simulation,
        )
        ee_position, ee_orientation = robot.ee_pose()
        position_error = float(np.linalg.norm(ee_position - target))
        orientation_error = float(
            np.linalg.norm(quaternion_error_vector(ee_orientation, desired_orientation))
        )
        return MotionResult(
            success=reached,
            steps=steps,
            position_error=position_error,
            orientation_error=orientation_error,
            reason="target_reached" if reached else "step_limit",
        )
