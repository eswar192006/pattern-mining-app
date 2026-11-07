"""Check that your Gemini key, model and the request format all work. Run once after setting up .env:

    python -m src.check_gemini

It sends ONE small question and prints what came back. The key itself is never printed.
"""
import sys

from src.core.errors import LLMUnavailable
from src.llm import prompts
from src.llm.client import GeminiClient
from src.llm.parser import read_reply
from src.llm.reply import llm_json_schema

QUESTION = "Which products are often bought together on Sundays?"

if __name__ == "__main__":
    try:
        client = GeminiClient()
    except LLMUnavailable as e:
        sys.exit(f"[FAIL] {e}")
    print(f"[ok]   API key found in the environment or .env (hidden). Model: {client.model}")
    print(f"       Question: {QUESTION}\n       Calling Gemini ...")
    try:
        raw = client.generate_json(prompts.build_system_prompt(), prompts.user_prompt(QUESTION), llm_json_schema())
    except LLMUnavailable as e:
        sys.exit(f"[FAIL] {e}")
    print(f"[ok]   Gemini answered ({len(raw)} characters):\n       {raw.strip()[:600]}\n")
    reply, problem = read_reply(raw)
    if problem:
        sys.exit(f"[FAIL] The answer could not be read: {problem}")
    print("[ok]   The answer matches the required format:")
    print("       " + reply.model_dump_json(exclude_defaults=True))
    print("\n       Now checking plain-text answers (used to write the final explanation) ...")
    try:
        text = client.generate_text("Reply with exactly one word.", "Reply with the single word: OK")
    except LLMUnavailable as e:
        sys.exit(f"[FAIL] {e}")
    print(f"[ok]   Plain-text answer received: {text.strip()[:80]!r}\n")
    ok = reply.spec is not None and reply.spec.kind == "frequent" and "Sunday" in reply.spec.filters.weekdays
    print("[ok]   And it is the expected translation." if ok else
          "[note] The format is right, but the translation differs from the expected one "
          "(kind=frequent, Sunday). Models vary; try python -m src.demo_parse for more.")
