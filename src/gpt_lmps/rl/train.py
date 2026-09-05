from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gpt_lmps.rl.q_learning import StateDiscretizer, TabularQAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a value-based Franka reach controller")
    parser.add_argument("--algorithm", choices=["q_learning", "dqn"], required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=8_000, help="Q-Learning episodes")
    parser.add_argument("--timesteps", type=int, default=300_000, help="DQN environment steps")
    parser.add_argument("--max-episode-steps", type=int, default=160)
    return parser


def train_q_learning(
    *,
    episodes: int,
    seed: int,
    output_dir: Path,
    max_episode_steps: int,
) -> Path:
    from gpt_lmps.envs.joint_reach_env import JointReachEnv

    env = JointReachEnv(target_mode="grid", max_episode_steps=max_episode_steps)
    discretizer = StateDiscretizer()
    agent = TabularQAgent()
    rng = np.random.default_rng(seed)
    recent_success: list[float] = []

    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        state = discretizer.encode(observation)
        epsilon = max(0.05, 1.0 - episode / max(1, int(episodes * 0.8)))
        terminated = truncated = False
        info = {"is_success": False}
        while not (terminated or truncated):
            action = agent.action(state, epsilon=epsilon, rng=rng)
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_state = discretizer.encode(next_observation)
            agent.update(
                state,
                action,
                reward,
                next_state,
                learning_rate=0.18,
                discount=0.97,
                terminal=terminated or truncated,
            )
            state = next_state
        recent_success.append(float(info["is_success"]))
        if (episode + 1) % 250 == 0:
            rate = np.mean(recent_success[-250:])
            print(
                f"episode={episode + 1} epsilon={epsilon:.3f} "
                f"success_rate_250={rate:.3f} states={len(agent.q_values)}"
            )

    env.close()
    return agent.save(output_dir / "q_table.npz")


def train_dqn(
    *,
    timesteps: int,
    seed: int,
    output_dir: Path,
    max_episode_steps: int,
) -> Path:
    try:
        from stable_baselines3 import DQN
        from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise ImportError("Install the DQN dependencies with: pip install -e '.[rl]'") from exc
    from gpt_lmps.envs.joint_reach_env import JointReachEnv

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    model_dir.mkdir(exist_ok=True)
    env = Monitor(JointReachEnv(target_mode="continuous", max_episode_steps=max_episode_steps))
    eval_env = Monitor(JointReachEnv(target_mode="continuous", max_episode_steps=max_episode_steps))
    checkpoint = CheckpointCallback(
        save_freq=25_000,
        save_path=str(model_dir),
        name_prefix="dqn_franka",
    )
    evaluation = EvalCallback(
        eval_env,
        eval_freq=25_000,
        n_eval_episodes=20,
        best_model_save_path=str(model_dir / "best"),
        log_path=str(output_dir / "eval"),
        deterministic=True,
    )
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=2_000,
        batch_size=128,
        gamma=0.97,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=2_000,
        exploration_fraction=0.35,
        exploration_final_eps=0.05,
        policy_kwargs={"net_arch": [256, 256]},
        tensorboard_log=str(output_dir / "tensorboard"),
        seed=seed,
        verbose=1,
    )
    model.learn(total_timesteps=timesteps, callback=[checkpoint, evaluation])
    output = output_dir / "final_model"
    model.save(output)
    env.close()
    eval_env.close()
    return output.with_suffix(".zip")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir or Path("outputs") / args.algorithm
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.algorithm == "q_learning":
        checkpoint = train_q_learning(
            episodes=args.episodes,
            seed=args.seed,
            output_dir=output_dir,
            max_episode_steps=args.max_episode_steps,
        )
    else:
        checkpoint = train_dqn(
            timesteps=args.timesteps,
            seed=args.seed,
            output_dir=output_dir,
            max_episode_steps=args.max_episode_steps,
        )
    print(f"Saved checkpoint to {checkpoint}")


if __name__ == "__main__":
    main()
