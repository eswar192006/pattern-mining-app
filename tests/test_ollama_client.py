import json
from urllib.error import URLError

import pytest

import src.llm.client as client_module
from src.core.errors import LLMUnavailable
from src.llm.client import OllamaClient, create_llm_client


class Response:
    def __init__(self, body):
        self.body = json.dumps(body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return self.body


def test_ollama_checks_model_and_uses_json_schema(monkeypatch):
    seen = []

    def fake_urlopen(request, timeout):
        seen.append((request, timeout))
        if request.get_method() == "GET":
            return Response({"models": [{"name": "qwen3.5:4b"}]})
        return Response({"message": {"content": '{"ok":true}'}})

    monkeypatch.setattr(client_module, "urlopen", fake_urlopen)
    client = OllamaClient(model="qwen3.5:4b", base_url="http://ollama.local/", timeout_s=9)
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}

    assert client.generate_json("system rules", "user question", schema) == '{"ok":true}'
    assert seen[0][0].full_url == "http://ollama.local/api/tags" and seen[0][1] == 4
    request, timeout = seen[1]
    body = json.loads(request.data)
    assert request.full_url == "http://ollama.local/api/chat" and timeout == 9
    assert body["model"] == "qwen3.5:4b" and body["stream"] is False and body["think"] is False
    assert body["format"] == schema and body["options"]["temperature"] == 0
    assert body["messages"] == [{"role": "system", "content": "system rules"},
                                {"role": "user", "content": "user question"}]


def test_ollama_text_request_omits_structured_format(monkeypatch):
    bodies = []

    def fake_urlopen(request, timeout):
        if request.get_method() == "GET":
            return Response({"models": [{"name": "qwen3.5:4b"}]})
        bodies.append(json.loads(request.data))
        return Response({"message": {"content": "Ready."}})

    monkeypatch.setattr(client_module, "urlopen", fake_urlopen)
    client = OllamaClient()

    assert client.generate_text("be brief", "status?") == "Ready."
    assert "format" not in bodies[0]


def test_ollama_reports_missing_model_and_unavailable_server(monkeypatch):
    monkeypatch.setattr(client_module, "urlopen", lambda *_a, **_kw: Response({"models": []}))
    with pytest.raises(LLMUnavailable, match="ollama pull qwen3.5:4b"):
        OllamaClient()

    monkeypatch.setattr(client_module, "urlopen", lambda *_a, **_kw: (_ for _ in ()).throw(URLError("offline")))
    with pytest.raises(LLMUnavailable, match="Could not reach Ollama"):
        OllamaClient()

    monkeypatch.setattr(client_module, "urlopen", lambda *_a, **_kw: Response({"models": None}))
    with pytest.raises(LLMUnavailable, match="Could not reach Ollama"):
        OllamaClient()


def test_provider_factory_selects_ollama_and_rejects_unknown(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(client_module, "OllamaClient", lambda: sentinel)
    assert create_llm_client("ollama") is sentinel
    with pytest.raises(LLMUnavailable, match="Choose 'ollama' or 'gemini'"):
        create_llm_client("other")
