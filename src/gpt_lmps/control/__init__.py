from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gpt_lmps.control.base import MotionController

ControllerBuilder = Callable[[str | Path | None], Any]
_CONTROLLER_BUILDERS: dict[str, ControllerBuilder] = {}


def _normalize_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    if not normalized:
        raise ValueError("Controller name cannot be empty")
    return normalized


def register_motion_controller(
    name: str,
    builder: ControllerBuilder,
    *,
    replace: bool = False,
) -> None:
    """Register a controller builder without changing the LMP or scene layers."""

    normalized = _normalize_name(name)
    if normalized in _CONTROLLER_BUILDERS and not replace:
        raise ValueError(f"Motion controller is already registered: {normalized}")
    _CONTROLLER_BUILDERS[normalized] = builder


def available_motion_controllers() -> tuple[str, ...]:
    return tuple(sorted(_CONTROLLER_BUILDERS))


def _build_ik(_checkpoint: str | Path | None) -> MotionController:
    from gpt_lmps.control.ik import IKController

    return IKController()


def _required_checkpoint(
    checkpoint: str | Path | None,
    default: str,
    display_name: str,
) -> Path:
    model_path = Path(checkpoint or default)
    if not model_path.exists():
        raise FileNotFoundError(
            f"{display_name} checkpoint not found: {model_path}. Train it with gpt-lmps-train."
        )
    return model_path


def _build_q_learning(checkpoint: str | Path | None) -> MotionController:
    from gpt_lmps.control.value_based import QLearningController

    return QLearningController(
        _required_checkpoint(
            checkpoint,
            "outputs/q_learning/q_table.npz",
            "Q-Learning",
        )
    )


def _build_dqn(checkpoint: str | Path | None) -> MotionController:
    from gpt_lmps.control.value_based import DQNController

    return DQNController(
        _required_checkpoint(
            checkpoint,
            "outputs/dqn/final_model.zip",
            "DQN",
        )
    )


def create_motion_controller(
    name: str,
    checkpoint: str | Path | None = None,
) -> MotionController:
    normalized = _normalize_name(name)
    try:
        builder = _CONTROLLER_BUILDERS[normalized]
    except KeyError as exc:
        available = ", ".join(available_motion_controllers())
        raise ValueError(f"Unknown controller: {name}. Available controllers: {available}") from exc
    return builder(checkpoint)


register_motion_controller("ik", _build_ik)
register_motion_controller("q_learning", _build_q_learning)
register_motion_controller("dqn", _build_dqn)


__all__ = [
    "available_motion_controllers",
    "create_motion_controller",
    "register_motion_controller",
]
