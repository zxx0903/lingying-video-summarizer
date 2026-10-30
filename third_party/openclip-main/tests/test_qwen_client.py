from core.llm.qwen_api_client import QwenAPIClient


def test_qwen_client_uses_environment_overrides(monkeypatch):
    with monkeypatch.context() as env:
        env.setenv("QWEN_API_KEY", "test-key")
        env.setenv("QWEN_BASE_URL", "https://qwen.example/compatible-mode/v1/")
        env.setenv("QWEN_MODEL", "qwen-local-test")

        client = QwenAPIClient()

        assert client.base_url == "https://qwen.example/compatible-mode/v1/chat/completions"
        assert client.default_model == "qwen-local-test"


def test_qwen_client_preserves_explicit_full_endpoint(monkeypatch):
    monkeypatch.setenv("QWEN_API_KEY", "test-key")

    client = QwenAPIClient(
        base_url="https://qwen.example/custom/chat/completions",
    )

    assert client.base_url == "https://qwen.example/custom/chat/completions"


def test_qwen_client_preserves_explicit_zero_temperature():
    client = QwenAPIClient(api_key="test-key")
    captured = {}

    def fake_make_request(payload, model):
        captured["payload"] = payload
        captured["model"] = model
        return {"choices": [{"message": {"content": "ok"}}]}

    client._make_request = fake_make_request

    reply = client.simple_chat("translate this", temperature=0)

    assert reply == "ok"
    assert captured["payload"]["temperature"] == 0
