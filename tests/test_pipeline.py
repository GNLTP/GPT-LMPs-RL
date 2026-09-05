import numpy as np

from gpt_lmps.llm import RuleBasedBackend
from gpt_lmps.lmp import LMPRobotPipeline
from gpt_lmps.world import SymbolicTabletop


class _PerceptionOnlyBackend:
    def generate(self, *, name: str, system: str, query: str, context: str) -> str:
        del system, query, context
        outputs = {
            "planner": "composer('move the red block to its perceived position')",
            "composer": (
                "target = parse_position('red block')\n"
                "put_first_on_second('red block', target)"
            ),
            "parse_position": "ret_val = get_obj_pos('red block')",
        }
        return outputs[name]


class _PerceptionFirstWorld(SymbolicTabletop):
    def get_obj_pos(self, name: str) -> np.ndarray:
        del name
        return np.asarray([9.0, 9.0, 9.0], dtype=np.float32)

    def get_scene_positions(self) -> dict[str, np.ndarray]:
        return {"red block": np.asarray([0.51, -0.12, 0.025], dtype=np.float32)}


def test_stack_is_decomposed_and_red_is_last() -> None:
    world = SymbolicTabletop(
        objects=["red block", "blue block", "green block", "orange bowl"],
        positions={
            "red block": [0.40, -0.15, 0.025],
            "blue block": [0.52, 0.00, 0.025],
            "green block": [0.64, 0.15, 0.025],
            "orange bowl": [0.62, -0.18, 0.008],
        },
    )
    trace = LMPRobotPipeline(world, RuleBasedBackend()).run(
        "stack all blocks with the red block on top"
    )

    assert [event.stage for event in trace.events] == ["planner", "composer", "composer"]
    assert len(trace.primitive_calls) == 2
    assert trace.primitive_calls[-1][1][0] == "red block"
    assert world.positions["red block"][2] > world.positions["blue block"][2]


def test_move_to_corner_uses_denormalized_position() -> None:
    world = SymbolicTabletop(objects=["blue block", "orange bowl"])
    trace = LMPRobotPipeline(world, RuleBasedBackend()).run(
        "move the blue block to the top right corner"
    )

    assert trace.primitive_calls[0][1][0] == "blue block"
    np.testing.assert_allclose(world.positions["blue block"][:2], world.denormalize_xy([1.0, 1.0]))


def test_nested_lmp_object_query_uses_perception_interface() -> None:
    world = _PerceptionFirstWorld(objects=["red block"])
    LMPRobotPipeline(world, _PerceptionOnlyBackend()).run("move the red block")

    np.testing.assert_allclose(world.positions["red block"][:2], [0.51, -0.12])
