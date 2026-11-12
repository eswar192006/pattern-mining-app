"""Local, keyless language responses for the six-question presentation demo.

The saved interpretations only select a validated mining request. Results are still
computed by the normal mining engine from the local retail data.
"""
import json
import re
from pathlib import Path

from src.core import config as C
from src.core.errors import LLMUnavailable


def _normalise(text: str) -> str:
    clean = re.sub(r"[?!.,]+$", "", text.strip())
    return " ".join(clean.casefold().split())


class ReplayLLM:
    model = "local-replay"

    def __init__(self, fixture_path: Path | None = None):
        path = fixture_path or (C.ROOT / "demo" / "replay_responses.json")
        try:
            entries = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LLMUnavailable(f"Could not load local demo responses from {path}: {exc}") from None
        self._entries = {_normalise(row["question"]): row for row in entries}
        if len(self._entries) != len(entries):
            raise ValueError("Replay question fixtures must be unique, ignoring case and whitespace.")

    def _entry(self, prompt: str) -> dict:
        if "CURRENT REQUEST:" in prompt:
            questions = re.findall(r"(?:^|\n)Message:\s*([^\n]+)", prompt)
        else:
            questions = re.findall(r"(?:^|\n)Question:\s*([^\n]+)", prompt)
        question = questions[-1].strip() if questions else ""
        entry = self._entries.get(_normalise(question))
        if entry is None and "CURRENT REQUEST:" in prompt:
            entry = next((row for row in self._entries.values()
                          if row.get("clarification_reply")
                          and _normalise(row["clarification_reply"]) == _normalise(question)), None)
        if entry is None:
            raise LLMUnavailable(
                "Replay mode supports only the six saved questions in demo/walkthrough.md. "
                "Use one of those prompts or restart in live mode."
            )
        return entry

    def generate_json(self, system: str, user: str, schema: dict | None = None) -> str:
        entry = self._entry(user)
        if "CURRENT REQUEST:" in user:
            message = re.findall(r"(?:^|\n)Message:\s*([^\n]+)", user)
            is_clarification = bool(message and entry.get("clarification_reply")
                                    and _normalise(message[-1]) == _normalise(entry["clarification_reply"]))
            is_followup = entry.get("is_followup", False)
            if is_clarification:
                spec = entry.get("clarification_spec", {})
                fresh = False
            elif is_followup:
                spec = entry.get("followup_spec", entry["spec"])
                fresh = False
            else:
                spec = entry["spec"]
                fresh = True
            response = {"intent": "mining_request", "spec": spec, "fresh": fresh}
        else:
            response = {"intent": "mining_request", "spec": entry["spec"], "message": None}
        return json.dumps(response)

    def generate_text(self, system: str, user: str) -> str:
        try:
            return self._entry(user)["answer"]
        except LLMUnavailable:
            # Follow-up mining may not carry a standalone question into the explainer.
            if user.startswith("FACTS:\n"):
                return ("These combinations are supported by the local transaction data. Consider testing a few "
                        "leading pairs as bundles and monitoring their performance over time.")
            raise
