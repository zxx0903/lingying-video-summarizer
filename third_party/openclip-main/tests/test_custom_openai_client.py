from core.llm.custom_openai_api_client import CustomOpenAIAPIClient
from core.subtitle_burner import SubtitleBurner
from core.config import LLM_CONFIG
import requests


def test_custom_openai_client_uses_environment_overrides_without_api_key(monkeypatch):
    with monkeypatch.context() as env:
        env.delenv("CUSTOM_OPENAI_API_KEY", raising=False)
        env.setenv("CUSTOM_OPENAI_BASE_URL", "https://gateway.example/v1/")
        env.setenv("CUSTOM_OPENAI_MODEL", "gateway-model")

        client = CustomOpenAIAPIClient()

        assert client.api_key is None
        assert client.base_url == "https://gateway.example/v1/chat/completions"
        assert client.default_model == "gateway-model"


def test_custom_openai_client_preserves_explicit_full_endpoint():
    client = CustomOpenAIAPIClient(
        base_url="https://gateway.example/custom/chat/completions",
    )

    assert client.base_url == "https://gateway.example/custom/chat/completions"


def test_subtitle_burner_allows_custom_openai_without_api_key():
    burner = SubtitleBurner(
        provider="custom_openai",
        model="local-model",
        base_url="https://gateway.example/v1",
        enable_llm=True,
    )

    assert burner.client is not None
    assert burner.model == "local-model"


def test_custom_openai_uses_default_sampling_fields_by_default(monkeypatch):
    client = CustomOpenAIAPIClient(
        api_key="test-key",
        base_url="https://gateway.example/v1",
    )
    captured = {}

    class FakeResponse:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "ok"}}]}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("core.llm.custom_openai_api_client.requests.post", fake_post)

    reply = client.simple_chat("hello", model="deepseek-chat")

    assert reply == "ok"
    assert captured["json"] == {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": LLM_CONFIG["custom_openai"]["default_params"]["max_tokens"],
        "temperature": LLM_CONFIG["custom_openai"]["default_params"]["temperature"],
        "top_p": LLM_CONFIG["custom_openai"]["default_params"]["top_p"],
        "stream": LLM_CONFIG["custom_openai"]["default_params"]["stream"],
    }


def test_custom_openai_includes_error_body_on_http_error(monkeypatch):
    client = CustomOpenAIAPIClient(
        api_key="test-key",
        base_url="https://gateway.example/v1",
    )

    class FakeResponse:
        status_code = 400

        @staticmethod
        def json():
            return {"error": {"message": "invalid model", "type": "invalid_request_error"}}

        text = '{"error":{"message":"invalid model","type":"invalid_request_error"}}'

        def raise_for_status(self):
            raise requests.exceptions.HTTPError(
                "400 Client Error: Bad Request for url: https://gateway.example/v1/chat/completions"
            )

    def fake_post(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr("core.llm.custom_openai_api_client.requests.post", fake_post)

    try:
        client.simple_chat("hello", model="bad-model")
        assert False, "Expected HTTP error to be raised"
    except Exception as exc:
        message = str(exc)
        assert "Response body:" in message
        assert "invalid model" in message
