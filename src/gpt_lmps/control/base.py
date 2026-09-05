from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from gpt_lmps.envs.robot import PandaRobot

SimulationStepper = Callable[[int], None]


@dataclass(frozen=True)
class MotionResult:
    """Result returned by every motion-control backend."""

    success: bool
    steps: int
    position_error: float
    orientation_error: float
    reason: str


class MotionController(Protocol):
    """Reusable interface shared by analytic and learned controllers."""

    name: str

    def move_to(
        self,
        robot: PandaRobot,
        target_position: Sequence[float],
        *,
        step_simulation: SimulationStepper,
        target_orientation: Sequence[float] | None = None,
        tolerance: float = 0.025,
    ) -> MotionResult: ...


class MotionControlError(RuntimeError):
    """Raised when a controller cannot reach a required task waypoint."""


def as_target(position: Sequence[float]) -> np.ndarray:
    target = np.asarray(position, dtype=np.float32)
    if target.shape != (3,):
        raise ValueError(f"Expected a 3-D target position, got shape {target.shape}")
    return target
