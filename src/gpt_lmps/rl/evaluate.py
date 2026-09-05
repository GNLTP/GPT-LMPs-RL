from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np

from gpt_lmps.rl.q_learning import StateDiscretizer, TabularQAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a learned Franka reach controller")
    parser.add_argument("model", type=Path)
    parser.add_argument("--algorithm", choices=["q_learning", "dqn"], required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--gui", action="store_true")
    return parser


def _load_policy(algorithm: str, model_path: Path) -> Callable[[np.ndarray], int]:
    if algorithm == "q_learning":
        agent = TabularQAgent.load(model_path)
        discretizer = StateDiscretizer()
        return lambda observation: agent.greedy_action(discretizer.encode(observation))
    try:
        from stable_baselines3 import DQN
    except ImportError as exc:
        raise ImportError("Install the DQN dependencies with: pip install -e '.[rl]'") from exc
    model = DQN.load(str(model_path))

    def dqn_policy(observation: np.ndarray) -> int:
        action, _ = model.predict(observation, deterministic=True)
        return int(np.asarray(action).item())

    return dqn_policy


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    from gpt_lmps.envs.joint_reach_env import JointReachEnv

    target_mode = "grid" if args.algorithm == "q_learning" else "continuous"
    env = JointReachEnv(
        target_mode=target_mode,
        render_mode="human" if args.gui else None,
    )
    policy = _load_policy(args.algorithm, args.model)
    successes = 0
    final_errors: list[float] = []
    for episode in range(args.episodes):
        observation, _ = env.reset(seed=episode)
        terminated = truncated = False
        info = {"is_success": False, "position_error": float("inf")}
        while not (terminated or truncated):
            observation, _, terminated, truncated, info = env.step(policy(observation))
        successes += int(info["is_success"])
        final_errors.append(float(info["position_error"]))
    env.close()
    print(f"success_rate={successes / args.episodes:.3f}")
    print(f"mean_final_position_error={np.mean(final_errors):.4f}")


if __name__ == "__main__":
    main()
