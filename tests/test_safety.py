import pytest

from gpt_lmps.lmp.safety import SafeExecutor, UnsafeCodeError, strip_markdown_fences


def test_markdown_fence_is_removed() -> None:
    assert strip_markdown_fences("```python\nsay('ok')\n```") == "say('ok')"


@pytest.mark.parametrize(
    "source",
    [
        "import os",
        "open('/tmp/payload', 'w')",
        "value = robot.__class__",
        "def hidden():\n    return 1",
    ],
)
def test_unsafe_code_is_rejected(source: str) -> None:
    with pytest.raises(UnsafeCodeError):
        SafeExecutor().execute(source, {"say": print})


def test_allowlisted_program_executes() -> None:
    messages: list[str] = []
    locals_dict = SafeExecutor().execute("say('ok')\nret_val = 3", {"say": messages.append})
    assert messages == ["ok"]
    assert locals_dict["ret_val"] == 3
