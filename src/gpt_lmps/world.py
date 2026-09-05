from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class SymbolicTabletop:
    """Fast no-physics world used for dry runs and unit tests."""

    def __init__(
        self,
        objects: Sequence[str] | None = None,
        positions: dict[str, Sequence[float]] | None = None,
    ) -> None:
        self.objects = list(objects or ["red block", "blue block", "green block", "orange bowl"])
        default_positions = {
            name: [0.42 + 0.08 * (index % 3), -0.15 + 0.15 * (index // 3), 0.02]
            for index, name in enumerate(self.objects)
        }
        if positions:
            default_positions.update({name: list(pos) for name, pos in positions.items()})
        self.positions = {
            name: np.asarray(pos, dtype=np.float32) for name, pos in default_positions.items()
        }
        self.actions: list[tuple[str, str | list[float]]] = []
        self.bounds = np.array([[0.35, 0.70], [-0.28, 0.28]], dtype=np.float32)

    def get_obj_names(self) -> list[str]:
        return list(self.objects)

    def get_obj_pos(self, name: str) -> np.ndarray:
        if name not in self.positions:
            raise KeyError(f"Unknown object: {name}")
        return self.positions[name].copy()

    def get_scene_positions(self) -> dict[str, np.ndarray]:
        return {name: position.copy() for name, position in self.positions.items()}

    def denormalize_xy(self, xy: Sequence[float]) -> np.ndarray:
        normalized = np.asarray(xy, dtype=np.float32)
        return self.bounds[:, 0] + normalized * (self.bounds[:, 1] - self.bounds[:, 0])

    def put_first_on_second(self, source: str, target: str | Sequence[float]) -> None:
        if source not in self.positions:
            raise KeyError(f"Unknown source object: {source}")
        if isinstance(target, str):
            if target not in self.positions:
                raise KeyError(f"Unknown target object: {target}")
            destination = self.positions[target].copy()
            destination[2] += 0.05
            target_record: str | list[float] = target
        else:
            target_xy = np.asarray(target, dtype=np.float32)[:2]
            destination = np.r_[target_xy, 0.02]
            target_record = target_xy.tolist()
        self.positions[source] = destination.astype(np.float32)
        self.actions.append((source, target_record))
