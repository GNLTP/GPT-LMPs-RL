from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from gpt_lmps.llm import LLMBackend
from gpt_lmps.lmp.core import LanguageModelProgram
from gpt_lmps.lmp.prompts import (
    COMPOSER_PROMPT,
    PARSE_OBJECT_PROMPT,
    PARSE_POSITION_PROMPT,
    PARSE_QUESTION_PROMPT,
    PLANNER_PROMPT,
    TRANSFORM_POINTS_PROMPT,
)
from gpt_lmps.lmp.safety import SafeExecutor
from gpt_lmps.types import ExecutionTrace


class TabletopAPI(Protocol):
    """Perception and control contract exposed to generated programs."""

    def get_obj_names(self) -> list[str]: ...

    def get_obj_pos(self, name: str) -> np.ndarray: ...

    def get_scene_positions(self) -> dict[str, np.ndarray]: ...

    def denormalize_xy(self, xy: Sequence[float]) -> np.ndarray: ...

    def put_first_on_second(self, source: str, target: str | Sequence[float]) -> None: ...


def _add_xy(first: Sequence[float], second: Sequence[float]) -> np.ndarray:
    return np.asarray(first, dtype=np.float32)[:2] + np.asarray(second, dtype=np.float32)[:2]


def _scale_points(points: Sequence[Sequence[float]], factor: float) -> np.ndarray:
    array = np.asarray(points, dtype=np.float32)
    center = array.mean(axis=0, keepdims=True)
    return center + factor * (array - center)


def _rotate_points(points: Sequence[Sequence[float]], degrees: float) -> np.ndarray:
    array = np.asarray(points, dtype=np.float32)
    center = array.mean(axis=0, keepdims=True)
    theta = np.deg2rad(degrees)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return (array - center) @ rotation.T + center


class LMPRobotPipeline:
    """Six-stage hierarchical language-model program pipeline.

    Planner code calls ``composer``. Composer code may call the four low-level LMPs and the
    whitelisted pick-and-place primitive. Every generated program is AST-validated before use.
    """

    def __init__(self, api: TabletopAPI, backend: LLMBackend) -> None:
        self.api = api
        self.backend = backend
        self.executor = SafeExecutor()
        self._active_trace: ExecutionTrace | None = None

        common_symbols: dict[str, Any] = {
            # Generated LMPs consume the same perception result used to build
            # their scene context. In simulation this routes object queries
            # through the top RGB-D camera rather than PyBullet body state.
            "get_obj_pos": self._get_perceived_obj_pos,
            "get_obj_names": self.api.get_obj_names,
            "denormalize_xy": self.api.denormalize_xy,
            "add_xy": _add_xy,
            "scale_points": _scale_points,
            "rotate_points": _rotate_points,
        }
        self.parse_obj_name_lmp = LanguageModelProgram(
            "parse_obj_name",
            PARSE_OBJECT_PROMPT,
            backend,
            self.executor,
            common_symbols,
            return_name="ret_val",
        )
        self.parse_position_lmp = LanguageModelProgram(
            "parse_position",
            PARSE_POSITION_PROMPT,
            backend,
            self.executor,
            common_symbols,
            return_name="ret_val",
        )
        self.parse_question_lmp = LanguageModelProgram(
            "parse_question",
            PARSE_QUESTION_PROMPT,
            backend,
            self.executor,
            common_symbols,
            return_name="ret_val",
        )
        self.transform_shape_lmp = LanguageModelProgram(
            "transform_shape_pts",
            TRANSFORM_POINTS_PROMPT,
            backend,
            self.executor,
            common_symbols,
            return_name="new_shape_pts",
        )
        composer_symbols = {
            **common_symbols,
            "put_first_on_second": self._put_first_on_second,
            "parse_obj_name": self.parse_obj_name,
            "parse_position": self.parse_position,
            "parse_question": self.parse_question,
            "transform_shape_pts": self.transform_shape_pts,
            "say": self.say,
        }
        self.composer_lmp = LanguageModelProgram(
            "composer",
            COMPOSER_PROMPT,
            backend,
            self.executor,
            composer_symbols,
            maintain_session=True,
        )
        self.planner_lmp = LanguageModelProgram(
            "planner",
            PLANNER_PROMPT,
            backend,
            self.executor,
            {"composer": self.composer, "say": self.say},
            maintain_session=True,
        )

    def _get_perceived_obj_pos(self, name: str) -> np.ndarray:
        positions = self.api.get_scene_positions()
        if name not in positions:
            raise KeyError(f"Unknown perceived object: {name}")
        return np.asarray(positions[name], dtype=np.float32).copy()

    def context(self) -> str:
        names = self.api.get_obj_names()
        scene_positions = self.api.get_scene_positions()
        positions = {name: scene_positions[name][:2].tolist() for name in names}
        return json.dumps({"objects": names, "positions": positions}, ensure_ascii=False)

    def run(self, instruction: str) -> ExecutionTrace:
        trace = ExecutionTrace(instruction=instruction)
        self._active_trace = trace
        try:
            self.planner_lmp(instruction, context=self.context(), trace=trace)
        finally:
            self._active_trace = None
        return trace

    def clear_history(self) -> None:
        self.planner_lmp.clear_history()
        self.composer_lmp.clear_history()

    def _trace(self) -> ExecutionTrace:
        if self._active_trace is None:
            raise RuntimeError("Nested LMP call was made outside pipeline.run()")
        return self._active_trace

    def composer(self, query: str) -> None:
        self.composer_lmp(query, context=self.context(), trace=self._trace())

    def parse_obj_name(self, query: str, context: str = "") -> Any:
        return self.parse_obj_name_lmp(
            query,
            context=context or self.context(),
            trace=self._trace(),
        )

    def parse_position(self, query: str, context: str = "") -> np.ndarray:
        return np.asarray(
            self.parse_position_lmp(
                query,
                context=context or self.context(),
                trace=self._trace(),
            ),
            dtype=np.float32,
        )

    def parse_question(self, query: str, context: str = "") -> Any:
        return self.parse_question_lmp(
            query,
            context=context or self.context(),
            trace=self._trace(),
        )

    def transform_shape_pts(
        self,
        query: str,
        *,
        shape_pts: Sequence[Sequence[float]],
    ) -> np.ndarray:
        return np.asarray(
            self.transform_shape_lmp(
                query,
                context=self.context(),
                trace=self._trace(),
                extra_symbols={"shape_pts": np.asarray(shape_pts, dtype=np.float32)},
            ),
            dtype=np.float32,
        )

    def _put_first_on_second(self, source: str, target: str | Sequence[float]) -> None:
        target_value: Any = target if isinstance(target, str) else np.asarray(target).tolist()
        self._trace().add_primitive("put_first_on_second", source, target_value)
        self.api.put_first_on_second(source, target)

    @staticmethod
    def say(message: str) -> None:
        print(f"robot: {message}")
