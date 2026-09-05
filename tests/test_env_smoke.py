import numpy as np
from gymnasium.utils.env_checker import check_env

from gpt_lmps.envs.joint_reach_env import JointReachEnv
from gpt_lmps.envs.scene import TabletopScene


def test_scene_rgbd_smoke() -> None:
    with TabletopScene(seed=1) as scene:
        observations = scene.reset(["red block", "blue block", "green bowl"])
        assert set(observations) == {"top", "oblique"}
        assert observations["top"].rgb.shape == (240, 320, 3)
        assert observations["top"].depth.shape == (240, 320)
        perceived = scene.perceive_object_positions()
        assert set(perceived) == {"red block", "blue block", "green bowl"}
        for name in perceived:
            np.testing.assert_allclose(perceived[name][:2], scene.get_obj_pos(name)[:2], atol=0.02)


def test_goal_env_contract_and_step() -> None:
    env = JointReachEnv(target_mode="grid", max_episode_steps=3)
    check_env(env, skip_render_check=True)
    observation, info = env.reset(seed=3)
    assert observation.shape == (20,)
    assert info["target_mode"] == "grid"
    next_observation, reward, terminated, truncated, step_info = env.step(0)
    assert next_observation.shape == (20,)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "is_success" in step_info
    env.close()
