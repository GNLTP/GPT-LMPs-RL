from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pybullet as p


@dataclass(frozen=True)
class CameraConfig:
    name: str
    eye: tuple[float, float, float]
    target: tuple[float, float, float]
    up: tuple[float, float, float]
    width: int = 320
    height: int = 240
    fov: float = 55.0
    near: float = 0.02
    far: float = 3.0


@dataclass(frozen=True)
class RGBDObservation:
    rgb: np.ndarray
    depth: np.ndarray
    segmentation: np.ndarray
    view_matrix: np.ndarray
    projection_matrix: np.ndarray


def default_cameras() -> tuple[CameraConfig, CameraConfig]:
    """Top and oblique views used for perception and visualization."""

    return (
        CameraConfig(
            name="top",
            eye=(0.52, 0.0, 0.95),
            target=(0.52, 0.0, 0.0),
            up=(0.0, 1.0, 0.0),
        ),
        CameraConfig(
            name="oblique",
            eye=(0.95, -0.75, 0.65),
            target=(0.50, 0.0, 0.05),
            up=(0.0, 0.0, 1.0),
        ),
    )


def render_rgbd(client_id: int, config: CameraConfig) -> RGBDObservation:
    view = p.computeViewMatrix(config.eye, config.target, config.up)
    projection = p.computeProjectionMatrixFOV(
        fov=config.fov,
        aspect=config.width / config.height,
        nearVal=config.near,
        farVal=config.far,
    )
    _, _, rgba, depth_buffer, segmentation = p.getCameraImage(
        width=config.width,
        height=config.height,
        viewMatrix=view,
        projectionMatrix=projection,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=client_id,
    )
    rgba_array = np.asarray(rgba, dtype=np.uint8).reshape(config.height, config.width, 4)
    depth_buffer_array = np.asarray(depth_buffer, dtype=np.float32).reshape(
        config.height, config.width
    )
    metric_depth = (
        config.far * config.near / (config.far - (config.far - config.near) * depth_buffer_array)
    )
    return RGBDObservation(
        rgb=rgba_array[..., :3],
        depth=metric_depth,
        segmentation=np.asarray(segmentation, dtype=np.int32).reshape(config.height, config.width),
        view_matrix=np.asarray(view, dtype=np.float32).reshape(4, 4, order="F"),
        projection_matrix=np.asarray(projection, dtype=np.float32).reshape(4, 4, order="F"),
    )


def depth_to_world_points(observation: RGBDObservation) -> np.ndarray:
    """Unproject metric-compatible OpenGL depth into an ``H x W x 3`` world point cloud."""

    height, width = observation.depth.shape
    xs = (np.arange(width, dtype=np.float32) + 0.5) / width * 2.0 - 1.0
    ys = 1.0 - (np.arange(height, dtype=np.float32) + 0.5) / height * 2.0
    grid_x, grid_y = np.meshgrid(xs, ys)

    projection = observation.projection_matrix
    # PyBullet returns OpenGL column-major matrices. After reshaping with
    # ``order="F"``, P[2, 3] stores the depth translation term while P[3, 2]
    # is the constant -1 perspective term.
    near = projection[2, 3] / (projection[2, 2] - 1.0)
    far = projection[2, 3] / (projection[2, 2] + 1.0)
    depth_buffer = (far - far * near / observation.depth) / (far - near)
    clip = np.stack(
        [grid_x, grid_y, depth_buffer * 2.0 - 1.0, np.ones_like(grid_x)],
        axis=-1,
    )
    inverse = np.linalg.inv(observation.projection_matrix @ observation.view_matrix)
    homogeneous = clip.reshape(-1, 4) @ inverse.T
    points = homogeneous[:, :3] / homogeneous[:, 3:4]
    return points.reshape(height, width, 3).astype(np.float32)
