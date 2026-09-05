from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pybullet as p
import pybullet_data


class PandaRobot:
    """Small Cartesian-control wrapper around PyBullet's Franka Panda URDF."""

    ARM_JOINTS = tuple(range(7))
    FINGER_JOINTS = (9, 10)
    EE_LINK = 11
    HOME_JOINTS = (0.0, -0.45, 0.0, -2.25, 0.0, 1.85, 0.78)
    DOWN_ORIENTATION = p.getQuaternionFromEuler((np.pi, 0.0, 0.0))
    LOWER_LIMITS = np.array([-2.90, -1.76, -2.90, -3.07, -2.90, -0.02, -2.90])
    UPPER_LIMITS = np.array([2.90, 1.76, 2.90, -0.07, 2.90, 3.75, 2.90])
    JOINT_RANGES = UPPER_LIMITS - LOWER_LIMITS

    def __init__(self, client_id: int) -> None:
        self.client_id = client_id
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
        self.body_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=(0.0, 0.0, 0.0),
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION,
            physicsClientId=client_id,
        )
        self.reset()

    def reset(self, joint_positions: Sequence[float] | None = None) -> None:
        positions = tuple(self.HOME_JOINTS if joint_positions is None else joint_positions)
        if len(positions) != len(self.ARM_JOINTS):
            raise ValueError("Franka arm reset requires seven joint positions")
        for joint, value in zip(self.ARM_JOINTS, positions, strict=True):
            p.resetJointState(self.body_id, joint, value, physicsClientId=self.client_id)
        for joint in self.FINGER_JOINTS:
            p.resetJointState(self.body_id, joint, 0.04, physicsClientId=self.client_id)
        self.open_gripper()

    def ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        state = p.getLinkState(
            self.body_id,
            self.EE_LINK,
            computeLinkVelocity=1,
            physicsClientId=self.client_id,
        )
        return np.asarray(state[4], dtype=np.float32), np.asarray(state[5], dtype=np.float32)

    def ee_velocity(self) -> np.ndarray:
        state = p.getLinkState(
            self.body_id,
            self.EE_LINK,
            computeLinkVelocity=1,
            physicsClientId=self.client_id,
        )
        return np.asarray(state[6], dtype=np.float32)

    def joint_positions(self) -> np.ndarray:
        states = p.getJointStates(
            self.body_id,
            list(self.ARM_JOINTS),
            physicsClientId=self.client_id,
        )
        return np.asarray([state[0] for state in states], dtype=np.float32)

    def joint_velocities(self) -> np.ndarray:
        states = p.getJointStates(
            self.body_id,
            list(self.ARM_JOINTS),
            physicsClientId=self.client_id,
        )
        return np.asarray([state[1] for state in states], dtype=np.float32)

    def command_joints(self, joint_positions: Sequence[float]) -> None:
        target = np.clip(
            np.asarray(joint_positions, dtype=np.float32),
            self.LOWER_LIMITS,
            self.UPPER_LIMITS,
        )
        p.setJointMotorControlArray(
            self.body_id,
            jointIndices=list(self.ARM_JOINTS),
            controlMode=p.POSITION_CONTROL,
            targetPositions=target.tolist(),
            forces=[87.0] * 7,
            positionGains=[0.08] * 7,
            physicsClientId=self.client_id,
        )

    def command_joint_delta(self, delta: Sequence[float]) -> None:
        update = np.asarray(delta, dtype=np.float32)
        if update.shape != (7,):
            raise ValueError(f"Expected seven joint increments, got shape {update.shape}")
        self.command_joints(self.joint_positions() + update)

    def joint_limit_cost(self, margin_ratio: float = 0.08) -> float:
        positions = self.joint_positions()
        lower_margin = (positions - self.LOWER_LIMITS) / self.JOINT_RANGES
        upper_margin = (self.UPPER_LIMITS - positions) / self.JOINT_RANGES
        violation = np.maximum(0.0, margin_ratio - np.minimum(lower_margin, upper_margin))
        return float(np.sum(np.square(violation / margin_ratio)))

    def has_self_collision(self) -> bool:
        contacts = p.getContactPoints(
            bodyA=self.body_id,
            bodyB=self.body_id,
            physicsClientId=self.client_id,
        )
        return bool(contacts)

    def command_ee(
        self,
        position: Sequence[float],
        orientation: Sequence[float] | None = None,
    ) -> None:
        target = np.asarray(position, dtype=np.float32)
        target_orientation = orientation if orientation is not None else self.DOWN_ORIENTATION
        joints = p.calculateInverseKinematics(
            self.body_id,
            self.EE_LINK,
            targetPosition=target.tolist(),
            targetOrientation=list(target_orientation),
            lowerLimits=self.LOWER_LIMITS.tolist(),
            upperLimits=self.UPPER_LIMITS.tolist(),
            jointRanges=self.JOINT_RANGES.tolist(),
            restPoses=list(self.HOME_JOINTS),
            maxNumIterations=100,
            residualThreshold=1e-4,
            physicsClientId=self.client_id,
        )
        self.command_joints(joints[:7])

    def move_ee(
        self,
        position: Sequence[float],
        *,
        orientation: Sequence[float] | None = None,
        tolerance: float = 0.012,
        max_steps: int = 360,
        step_simulation: Callable[[int], None] | None = None,
    ) -> tuple[bool, int]:
        target = np.asarray(position, dtype=np.float32)
        for step in range(1, max_steps + 1):
            self.command_ee(target, orientation)
            if step_simulation is None:
                p.stepSimulation(physicsClientId=self.client_id)
            else:
                step_simulation(1)
            current, _ = self.ee_pose()
            if np.linalg.norm(current - target) < tolerance:
                return True, step
        return False, max_steps

    def open_gripper(self) -> None:
        self._command_gripper(0.04)

    def close_gripper(self) -> None:
        self._command_gripper(0.0)

    def _command_gripper(self, target: float) -> None:
        p.setJointMotorControlArray(
            self.body_id,
            jointIndices=list(self.FINGER_JOINTS),
            controlMode=p.POSITION_CONTROL,
            targetPositions=[target, target],
            forces=[20.0, 20.0],
            physicsClientId=self.client_id,
        )
