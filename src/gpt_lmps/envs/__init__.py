"""PyBullet Franka environments.

Imports are intentionally lazy: planning-only workflows do not need to initialize PyBullet.
"""

__all__ = ["TabletopScene", "JointReachEnv"]


def __getattr__(name: str):
    if name == "TabletopScene":
        from gpt_lmps.envs.scene import TabletopScene

        return TabletopScene
    if name == "JointReachEnv":
        from gpt_lmps.envs.joint_reach_env import JointReachEnv

        return JointReachEnv
    raise AttributeError(name)
