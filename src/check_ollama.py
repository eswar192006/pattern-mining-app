"""Check that the configured local Ollama service and model are ready."""
from src.core.errors import LLMUnavailable
from src.llm.client import OllamaClient


def main():
    try:
        client = OllamaClient()
        answer = client.generate_json(
            "Return only a JSON object with an 'ok' boolean.",
            "Set ok to true.",
            {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
        )
    except LLMUnavailable as exc:
        raise SystemExit(f"[error] {exc}") from None
    print(f"[ok] Ollama model {client.model} is ready.")
    print(f"     Structured response: {answer.strip()[:300]}")


if __name__ == "__main__":
    main()
