from gpt_lmps.control import create_motion_controller, register_motion_controller


def test_custom_motion_controller_can_be_registered() -> None:
    sentinel = object()
    register_motion_controller("test_controller", lambda checkpoint: sentinel, replace=True)

    assert create_motion_controller("test-controller") is sentinel
