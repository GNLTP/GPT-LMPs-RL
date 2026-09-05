from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image

from gpt_lmps.control import create_motion_controller
from gpt_lmps.control.base import MotionControlError, MotionController
from gpt_lmps.envs.camera import (
    CameraConfig,
    RGBDObservation,
    default_cameras,
    depth_to_world_points,
    render_rgbd,
)
from gpt_lmps.envs.robot import PandaRobot

COLORS: dict[str, tuple[float, float, float, float]] = {
    "red": (0.90, 0.12, 0.12, 1.0),
    "blue": (0.10, 0.32, 0.90, 1.0),
    "green": (0.12, 0.65, 0.24, 1.0),
    "yellow": (0.95, 0.75, 0.08, 1.0),
    "orange": (0.95, 0.42, 0.08, 1.0),
    "pink": (0.95, 0.35, 0.60, 1.0),
    "brown": (0.42, 0.20, 0.08, 1.0),
    "gray": (0.48, 0.50, 0.54, 1.0),
}


class TabletopScene:
    """Franka tabletop simulator and the API consumed by LMP-generated programs."""

    BOUNDS = np.array([[0.35, 0.70], [-0.28, 0.28]], dtype=np.float32)
    BLOCK_HALF_SIZE = 0.025

    def __init__(
        self,
        *,
        gui: bool = False,
        seed: int = 0,
        cameras: Sequence[CameraConfig] | None = None,
        realtime: bool = False,
        controller: str = "ik",
        controller_checkpoint: str | Path | None = None,
        motion_controller: MotionController | None = None,
    ) -> None:
        self.gui = gui
        self.realtime = realtime
        self.rng = np.random.default_rng(seed)
        self.client_id = p.connect(p.GUI if gui else p.DIRECT)
        self.cameras = tuple(cameras or default_cameras())
        self.motion_controller = motion_controller or create_motion_controller(
            controller,
            controller_checkpoint,
        )
        self.object_ids: dict[str, int] = {}
        self._object_half_heights: dict[str, float] = {}
        self._grasp_constraint: int | None = None
        self._grasped_name: str | None = None
        self.robot: PandaRobot | None = None

    def close(self) -> None:
        if p.isConnected(self.client_id):
            p.disconnect(physicsClientId=self.client_id)

    def __enter__(self) -> TabletopScene:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def reset(self, object_names: Sequence[str]) -> dict[str, RGBDObservation]:
        p.resetSimulation(physicsClientId=self.client_id)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client_id)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.client_id)
        p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        self._create_table()
        self.robot = PandaRobot(self.client_id)
        self.object_ids.clear()
        self._object_half_heights.clear()
        self._grasp_constraint = None
        self._grasped_name = None

        positions = self._sample_positions(len(object_names))
        for name, xy in zip(object_names, positions, strict=True):
            self._create_object(name, xy)
        self.step_simulation(240)
        return self.observe()

    def _create_table(self) -> None:
        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.38, 0.34, 0.025],
            physicsClientId=self.client_id,
        )
        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.38, 0.34, 0.025],
            rgbaColor=[0.10, 0.11, 0.13, 1.0],
            physicsClientId=self.client_id,
        )
        p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[0.52, 0.0, -0.025],
            physicsClientId=self.client_id,
        )

    def _sample_positions(self, count: int, minimum_distance: float = 0.12) -> list[np.ndarray]:
        positions: list[np.ndarray] = []
        for _ in range(count):
            for _attempt in range(500):
                candidate = np.array(
                    [
                        self.rng.uniform(self.BOUNDS[0, 0] + 0.04, self.BOUNDS[0, 1] - 0.04),
                        self.rng.uniform(self.BOUNDS[1, 0] + 0.04, self.BOUNDS[1, 1] - 0.04),
                    ],
                    dtype=np.float32,
                )
                if all(
                    np.linalg.norm(candidate - previous) >= minimum_distance
                    for previous in positions
                ):
                    positions.append(candidate)
                    break
            else:
                raise RuntimeError("Could not sample non-overlapping tabletop objects")
        return positions

    def _create_object(self, name: str, xy: Sequence[float]) -> None:
        words = name.split()
        if len(words) != 2 or words[0] not in COLORS or words[1] not in {"block", "bowl"}:
            raise ValueError(f"Object names must look like 'red block' or 'blue bowl': {name}")
        color = COLORS[words[0]]
        if words[1] == "block":
            half = self.BLOCK_HALF_SIZE
            collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[half, half, half],
                physicsClientId=self.client_id,
            )
            visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[half, half, half],
                rgbaColor=color,
                physicsClientId=self.client_id,
            )
            body = p.createMultiBody(
                baseMass=0.04,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=[float(xy[0]), float(xy[1]), half],
                physicsClientId=self.client_id,
            )
            half_height = half
        else:
            radius, half_height = 0.052, 0.008
            collision = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=radius,
                height=2 * half_height,
                physicsClientId=self.client_id,
            )
            visual = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=radius,
                length=2 * half_height,
                rgbaColor=color,
                physicsClientId=self.client_id,
            )
            body = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=[float(xy[0]), float(xy[1]), half_height],
                physicsClientId=self.client_id,
            )
        self.object_ids[name] = body
        self._object_half_heights[name] = half_height

    def get_obj_names(self) -> list[str]:
        return list(self.object_ids)

    def get_obj_pos(self, name: str) -> np.ndarray:
        body = self.object_ids[name]
        position, _ = p.getBasePositionAndOrientation(body, physicsClientId=self.client_id)
        return np.asarray(position, dtype=np.float32)

    def perceive_object_positions(self, camera_name: str = "top") -> dict[str, np.ndarray]:
        """Estimate object centers from top-view depth and instance segmentation."""

        camera = next((item for item in self.cameras if item.name == camera_name), None)
        if camera is None:
            raise KeyError(f"Unknown camera: {camera_name}")
        observation = render_rgbd(self.client_id, camera)
        world_points = depth_to_world_points(observation)
        body_indices = observation.segmentation & ((1 << 24) - 1)
        positions: dict[str, np.ndarray] = {}
        for name, body_id in self.object_ids.items():
            mask = body_indices == body_id
            points = world_points[mask]
            points = points[np.all(np.isfinite(points), axis=1)]
            if len(points) < 4:
                positions[name] = self.get_obj_pos(name)
                continue
            center = np.median(points, axis=0).astype(np.float32)
            center[2] -= self._object_half_heights[name]
            positions[name] = center
        return positions

    def get_scene_positions(self) -> dict[str, np.ndarray]:
        """Scene positions exposed to LMPs through the top RGB-D observation."""

        return self.perceive_object_positions("top")

    def get_bounding_box(self, name: str) -> np.ndarray:
        return np.asarray(p.getAABB(self.object_ids[name], physicsClientId=self.client_id))

    def denormalize_xy(self, xy: Sequence[float]) -> np.ndarray:
        normalized = np.asarray(xy, dtype=np.float32)
        return self.BOUNDS[:, 0] + normalized * (self.BOUNDS[:, 1] - self.BOUNDS[:, 0])

    def put_first_on_second(self, source: str, target: str | Sequence[float]) -> None:
        if source not in self.object_ids:
            raise KeyError(f"Unknown source object: {source}")
        if source.endswith("bowl"):
            raise ValueError("Only blocks can be moved; bowls are fixed target markers")
        if isinstance(target, str):
            perceived = self.perceive_object_positions("top")
            target_pos = perceived.get(target, self.get_obj_pos(target))
            place_z = (
                target_pos[2]
                + self._object_half_heights[target]
                + self._object_half_heights[source]
                + 0.006
            )
        else:
            target_xy = np.asarray(target, dtype=np.float32)[:2]
            target_pos = np.r_[target_xy, 0.0]
            place_z = self._object_half_heights[source] + 0.006
        self.pick_and_place(source, [target_pos[0], target_pos[1], place_z])

    def pick_and_place(self, source: str, place_position: Sequence[float]) -> None:
        if self.robot is None:
            raise RuntimeError("Call reset() before executing a task")
        pick = self.perceive_object_positions("top").get(source, self.get_obj_pos(source))
        place = np.asarray(place_position, dtype=np.float32)
        self.robot.open_gripper()
        self._move_to([pick[0], pick[1], 0.24], "pre_grasp")
        self._move_to([pick[0], pick[1], pick[2] + 0.075], "grasp")
        self.robot.close_gripper()
        self.step_simulation(30)
        self.attach_object(source)
        self._move_to([pick[0], pick[1], 0.24], "lift")
        self._move_to([place[0], place[1], 0.24], "transport")
        self._move_to([place[0], place[1], place[2] + 0.075], "place")
        self.release_object()
        self.robot.open_gripper()
        self.step_simulation(90)
        self._move_to([place[0], place[1], 0.24], "retreat")

    def _move_to(self, position: Sequence[float], stage: str) -> None:
        if self.robot is None:
            raise RuntimeError("Robot is not initialized")
        result = self.motion_controller.move_to(
            self.robot,
            position,
            step_simulation=self.step_simulation,
            target_orientation=self.robot.DOWN_ORIENTATION,
        )
        if not result.success:
            raise MotionControlError(
                f"{self.motion_controller.name} failed during {stage}: "
                f"position_error={result.position_error:.4f}, reason={result.reason}"
            )

    def attach_object(self, name: str) -> None:
        if self.robot is None:
            raise RuntimeError("Robot is not initialized")
        self.release_object()
        parent_pos, parent_orn = self.robot.ee_pose()
        child_pos, child_orn = p.getBasePositionAndOrientation(
            self.object_ids[name],
            physicsClientId=self.client_id,
        )
        inverse_parent = p.invertTransform(parent_pos.tolist(), parent_orn.tolist())
        relative_pos, relative_orn = p.multiplyTransforms(
            inverse_parent[0],
            inverse_parent[1],
            child_pos,
            child_orn,
        )
        self._grasp_constraint = p.createConstraint(
            parentBodyUniqueId=self.robot.body_id,
            parentLinkIndex=self.robot.EE_LINK,
            childBodyUniqueId=self.object_ids[name],
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0.0, 0.0, 0.0],
            parentFramePosition=relative_pos,
            childFramePosition=[0.0, 0.0, 0.0],
            parentFrameOrientation=relative_orn,
            childFrameOrientation=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client_id,
        )
        self._grasped_name = name

    def release_object(self) -> None:
        if self._grasp_constraint is not None:
            p.removeConstraint(self._grasp_constraint, physicsClientId=self.client_id)
        self._grasp_constraint = None
        self._grasped_name = None

    @property
    def grasped_name(self) -> str | None:
        return self._grasped_name

    def try_grasp_nearest(self, maximum_distance: float = 0.075) -> str | None:
        if self.robot is None:
            return None
        ee_pos, _ = self.robot.ee_pose()
        blocks = [name for name in self.object_ids if name.endswith("block")]
        if not blocks:
            return None
        nearest = min(blocks, key=lambda name: np.linalg.norm(self.get_obj_pos(name) - ee_pos))
        if np.linalg.norm(self.get_obj_pos(nearest) - ee_pos) <= maximum_distance:
            self.robot.close_gripper()
            self.attach_object(nearest)
            return nearest
        return None

    def observe(self) -> dict[str, RGBDObservation]:
        return {camera.name: render_rgbd(self.client_id, camera) for camera in self.cameras}

    def save_camera(self, path: str | Path, camera_name: str = "oblique") -> Path:
        observations = self.observe()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(observations[camera_name].rgb).save(output)
        return output

    def step_simulation(self, steps: int = 1) -> None:
        for _ in range(steps):
            p.stepSimulation(physicsClientId=self.client_id)
            if self.gui and self.realtime:
                time.sleep(1.0 / 240.0)
