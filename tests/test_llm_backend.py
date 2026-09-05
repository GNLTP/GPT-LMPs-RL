import sys
from types import SimpleNamespace

from gpt_lmps.llm import DeepSeekBackend


def test_deepseek_backend_uses_configured_api(monkeypatch) -> None:
    calls = {}

    class _Responses:
        def create(self, **kwargs):
            calls["request"] = kwargs
            return SimpleNamespace(output_text="  put_first_on_second('red', 'blue')  ")

    class _Client:
        def __init__(self, **kwargs):
            calls["client"] = kwargs
            self.responses = _Responses()

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_Client))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-test-model")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://example.invalid")

    backend = DeepSeekBackend()
    result = backend.generate(
        name="composer",
        system="Return Python only.",
        query="put red on blue",
        context='{"objects": ["red", "blue"]}',
    )

    assert calls["client"] == {
        "api_key": "test-key",
        "base_url": "https://example.invalid",
    }
    assert calls["request"]["model"] == "deepseek-test-model"
    assert calls["request"]["reasoning"] == {"effort": "none"}
    assert "LMP stage: composer" in calls["request"]["input"]
    assert result == "put_first_on_second('red', 'blue')"


def test_deepseek_backend_requires_private_key(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=object))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    try:
        DeepSeekBackend()
    except RuntimeError as error:
        assert "DEEPSEEK_API_KEY" in str(error)
    else:
        raise AssertionError("DeepSeekBackend accepted a missing API key")
