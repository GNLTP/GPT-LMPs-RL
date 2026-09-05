from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np

DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class LLMBackend(Protocol):
    """Minimal interface used by every language-model program."""

    def generate(self, *, name: str, system: str, query: str, context: str) -> str:
        """Return Python source for one named LMP stage."""


@dataclass
class DeepSeekBackend:
    """DeepSeek Responses API adapter built on the OpenAI-compatible SDK.

    The import and client construction are lazy so offline demos do not require credentials.
    """

    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None

    def __post_init__(self) -> None:
        from openai import OpenAI

        self.model = self.model or os.environ.get("DEEPSEEK_MODEL") or DEFAULT_DEEPSEEK_MODEL
        self.base_url = (
            self.base_url
            or os.environ.get("DEEPSEEK_BASE_URL")
            or DEFAULT_DEEPSEEK_BASE_URL
        )
        self.api_key = self.api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set. Create your own DeepSeek API key and export it "
                "before selecting --backend deepseek."
            )
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate(self, *, name: str, system: str, query: str, context: str) -> str:
        response = self._client.responses.create(
            model=self.model,
            instructions=system,
            input=f"LMP stage: {name}\nContext:\n{context}\n\nQuery:\n{query}",
            reasoning={"effort": "none"},
        )
        return response.output_text.strip()


class RuleBasedBackend:
    """Deterministic, API-free backend for tests and the public quick start.

    It intentionally covers a compact subset of the supported tabletop instructions. The same
    pipeline can be switched to an LLM without changing robot-control code.
    """

    _COLOR_ALIASES = {
        "sun": "yellow",
        "banana": "yellow",
        "太阳": "yellow",
        "草": "green",
        "grass": "green",
        "ocean": "blue",
        "海洋": "blue",
    }

    def generate(self, *, name: str, system: str, query: str, context: str) -> str:
        del system
        state = self._state(context)
        if name == "planner":
            return self._plan(query, state)
        if name == "composer":
            return self._compose(query, state)
        if name == "parse_obj_name":
            return f"ret_val = {self._select_objects(query, state)!r}"
        if name == "parse_position":
            return self._position(query, state)
        if name == "parse_question":
            return self._question(query, state)
        if name == "transform_shape_pts":
            return self._transform(query)
        raise KeyError(f"Unknown LMP stage: {name}")

    @staticmethod
    def _state(context: str) -> dict:
        context = context.split("\nExecution history:", maxsplit=1)[0]
        try:
            payload = json.loads(context)
        except (TypeError, json.JSONDecodeError):
            return {"objects": [], "positions": {}}
        payload.setdefault("objects", [])
        payload.setdefault("positions", {})
        return payload

    @staticmethod
    def _normalized(text: str) -> str:
        return re.sub(r"[_\s]+", " ", text.lower()).strip(" .")

    def _mentioned(self, text: str, objects: list[str]) -> list[str]:
        normalized = self._normalized(text)
        indexed_mentions = [
            (normalized.find(self._normalized(obj)), obj)
            for obj in objects
            if self._normalized(obj) in normalized
        ]
        mentions = [obj for _, obj in sorted(indexed_mentions)]
        if mentions:
            return mentions
        for alias, color in self._COLOR_ALIASES.items():
            if alias in normalized:
                matches = [obj for obj in objects if obj.startswith(color)]
                if matches:
                    return matches
        return []

    def _select_objects(self, query: str, state: dict) -> str | list[str] | None:
        objects = list(state["objects"])
        text = self._normalized(query)
        mentions = self._mentioned(text, objects)
        if mentions:
            return (
                mentions
                if any(word in text for word in ("objects", "blocks", "bowls", "所有"))
                else mentions[0]
            )
        if "block" in text or "物块" in text:
            candidates = [obj for obj in objects if obj.endswith("block")]
        elif "bowl" in text or "碗" in text:
            candidates = [obj for obj in objects if obj.endswith("bowl")]
        else:
            candidates = objects
        positions = state.get("positions", {})
        if candidates and ("left most" in text or "leftmost" in text or "最左" in text):
            return min(candidates, key=lambda name: positions.get(name, [0.0, 0.0])[0])
        if candidates and ("right most" in text or "rightmost" in text or "最右" in text):
            return max(candidates, key=lambda name: positions.get(name, [0.0, 0.0])[0])
        if "all" in text or "the blocks" in text or "所有" in text:
            return candidates
        return candidates[0] if candidates else None

    def _plan(self, query: str, state: dict) -> str:
        text = self._normalized(query)
        objects = list(state["objects"])
        blocks = [obj for obj in objects if obj.endswith("block")]

        if any(word in text for word in ("stack", "堆", "叠")) and len(blocks) >= 2:
            mentioned = self._mentioned(text, blocks)
            top = (
                mentioned[-1]
                if mentioned and any(word in text for word in ("top", "顶部", "最上"))
                else None
            )
            order = [obj for obj in blocks if obj != top]
            if top:
                order.append(top)
            lines = [f"composer('put {order[1]} on {order[0]}')"]
            lines.extend(
                f"composer('put {order[i]} on {order[i - 1]}')" for i in range(2, len(order))
            )
            return "\n".join(lines)

        if any(word in text for word in ("put", "place", "move", "放", "移动", "搬")):
            return f"composer({query!r})"
        return "say('I can only plan tabletop pick, place, and stack tasks')"

    def _compose(self, query: str, state: dict) -> str:
        objects = list(state["objects"])
        mentions = self._mentioned(query, objects)
        if len(mentions) >= 2:
            return f"put_first_on_second({mentions[0]!r}, {mentions[1]!r})"
        if len(mentions) == 1:
            text = self._normalized(query)
            corner_map = {
                "top right": [1.0, 1.0],
                "top left": [0.0, 1.0],
                "bottom right": [1.0, 0.0],
                "bottom left": [0.0, 0.0],
                "右上": [1.0, 1.0],
                "左上": [0.0, 1.0],
                "右下": [1.0, 0.0],
                "左下": [0.0, 0.0],
            }
            for phrase, normalized_xy in corner_map.items():
                if phrase in text:
                    return (
                        f"target_pos = denormalize_xy({normalized_xy!r})\n"
                        f"put_first_on_second({mentions[0]!r}, target_pos)"
                    )
        return "say('The instruction does not identify a valid source and target')"

    def _position(self, query: str, state: dict) -> str:
        text = self._normalized(query)
        corners = {
            "top right": [1.0, 1.0],
            "top left": [0.0, 1.0],
            "bottom right": [1.0, 0.0],
            "bottom left": [0.0, 0.0],
            "middle": [0.5, 0.5],
        }
        for phrase, value in corners.items():
            if phrase in text:
                return f"ret_val = denormalize_xy({value!r})"
        mentions = self._mentioned(text, list(state["objects"]))
        if mentions:
            offset = [0.0, 0.0]
            match = re.search(r"(\d+)\s*cm\s*(left|right|above|below)", text)
            if match:
                distance = float(match.group(1)) / 100.0
                direction = match.group(2)
                offset = {
                    "left": [-distance, 0.0],
                    "right": [distance, 0.0],
                    "above": [0.0, distance],
                    "below": [0.0, -distance],
                }[direction]
            return f"ret_val = add_xy(get_obj_pos({mentions[0]!r}), {offset!r})"
        return "ret_val = denormalize_xy([0.5, 0.5])"

    def _question(self, query: str, state: dict) -> str:
        text = self._normalized(query)
        objects = list(state["objects"])
        if "how many" in text or "多少" in text:
            selected = self._select_objects(query, state)
            values = (
                selected if isinstance(selected, list) else ([] if selected is None else [selected])
            )
            return f"ret_val = {len(values)}"
        mentions = self._mentioned(text, objects)
        positions = state.get("positions", {})
        if len(mentions) >= 2:
            first = np.asarray(positions.get(mentions[0], [0.0, 0.0]))
            second = np.asarray(positions.get(mentions[1], [0.0, 0.0]))
            if "right" in text or "右" in text:
                answer = bool(first[0] > second[0])
            elif "left" in text or "左" in text:
                answer = bool(first[0] < second[0])
            elif "above" in text or "上" in text:
                answer = bool(first[1] > second[1])
            else:
                answer = bool(first[1] < second[1])
            return f"ret_val = {answer!r}"
        return "ret_val = False"

    @staticmethod
    def _transform(query: str) -> str:
        text = query.lower()
        scale_match = re.search(r"(?:scale|缩放).*?([0-9]*\.?[0-9]+)", text)
        if scale_match:
            return f"new_shape_pts = scale_points(shape_pts, {float(scale_match.group(1))!r})"
        rotate_match = re.search(r"(?:rotate|旋转).*?(-?[0-9]*\.?[0-9]+)", text)
        if rotate_match:
            return f"new_shape_pts = rotate_points(shape_pts, {float(rotate_match.group(1))!r})"
        return "new_shape_pts = shape_pts"
