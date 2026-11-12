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
        explanation = client.generate_text("Answer in one short sentence.", "Say Ollama is ready.")
    except LLMUnavailable as exc:
        raise SystemExit(f"[error] {exc}") from None
    if not answer.strip() or not explanation.strip():
        raise SystemExit("[error] Ollama returned an empty response. Check the model and retry.")
    print(f"[ok] Ollama model {client.model} is ready.")
    print(f"     Structured response: {answer.strip()[:300]}")
    print(f"     Text response: {explanation.strip()[:200]}")


if __name__ == "__main__":
    main()
