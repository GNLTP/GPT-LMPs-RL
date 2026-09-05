from __future__ import annotations

import argparse
from pathlib import Path

from gpt_lmps.llm import DeepSeekBackend, RuleBasedBackend
from gpt_lmps.lmp import LMPRobotPipeline
from gpt_lmps.world import SymbolicTabletop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the GPT-LMPs Franka manipulation pipeline")
    parser.add_argument(
        "--instruction",
        default="stack all blocks with the red block on top",
        help="Natural-language tabletop instruction",
    )
    parser.add_argument("--backend", choices=["rule", "deepseek"], default="rule")
    parser.add_argument(
        "--model",
        default=None,
        help="DeepSeek model override; defaults to DEEPSEEK_MODEL or deepseek-v4-flash",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="DeepSeek-compatible API base URL; defaults to DEEPSEEK_BASE_URL",
    )
    parser.add_argument(
        "--controller",
        default="ik",
        metavar="NAME",
        help="Motion backend for each waypoint: ik, q_learning, dqn, or a registered name",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Q-Learning or DQN checkpoint; otherwise use the default outputs path",
    )
    parser.add_argument("--simulate", action="store_true", help="Execute in PyBullet")
    parser.add_argument(
        "--gui", action="store_true", help="Show PyBullet GUI (requires --simulate)"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("outputs/final_scene.png"))
    return parser


def _print_trace(trace) -> None:
    for index, event in enumerate(trace.events, start=1):
        print(f"\n[{index}] {event.stage}: {event.query}\n{event.code}")
    print("\nPrimitive calls:")
    for name, args in trace.primitive_calls:
        print(f"- {name}{args}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    backend = (
        RuleBasedBackend()
        if args.backend == "rule"
        else DeepSeekBackend(model=args.model, base_url=args.base_url)
    )

    if not args.simulate:
        world = SymbolicTabletop()
        trace = LMPRobotPipeline(world, backend).run(args.instruction)
        _print_trace(trace)
        return

    from gpt_lmps.envs.scene import TabletopScene

    with TabletopScene(
        gui=args.gui,
        seed=args.seed,
        realtime=args.gui,
        controller=args.controller,
        controller_checkpoint=args.checkpoint,
    ) as world:
        world.reset(["red block", "blue block", "green block", "orange bowl"])
        trace = LMPRobotPipeline(world, backend).run(args.instruction)
        world.save_camera(args.output)
        _print_trace(trace)
        print(f"\nSaved final camera frame to {args.output}")


if __name__ == "__main__":
    main()
