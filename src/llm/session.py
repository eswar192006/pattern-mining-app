"""Conversation state: what the user is asking for right now, and how to go back.

A session holds the CURRENT SPEC (not the chat transcript): the state of the request is what matters
for a follow-up like "only Sundays", and a spec is small, exact and easy to check. It also keeps
  * the clarifying questions still waiting for an answer ("which cakestand?"),
  * a short history of earlier specs, so "undo" works (reset is undoable too),
  * a log of recent turns for the interface.
Everything is plain data (to_dict / from_dict), so the API in Step 11 can store sessions anywhere.
"""
from dataclasses import dataclass, field

from src.core.spec import Spec

MAX_HISTORY = 10
MAX_TURNS = 20


@dataclass
class Session:
    spec: Spec | None = None
    questions: list = field(default_factory=list)      # pending clarifying questions, as dicts
    history: list = field(default_factory=list)        # earlier specs as dicts, most recent last
    turns: list = field(default_factory=list)          # [{"user", "turn_type", "status"}]

    @property
    def has_context(self) -> bool:
        """True once there is something to follow up on (a spec, or a spec that can be undone)."""
        return self.spec is not None or bool(self.history)

    def current(self) -> dict | None:
        return self.spec.model_dump(exclude_defaults=True) if self.spec is not None else None

    def push_history(self) -> None:
        if self.spec is not None:
            self.history.append(self.current())
            del self.history[:-MAX_HISTORY]

    def set_spec(self, spec: Spec, questions=None) -> None:
        """Make `spec` current, remembering the one it replaces (only if it really differs)."""
        if self.spec is None or self.spec.model_dump() != spec.model_dump():
            self.push_history()
        self.spec = spec
        self.questions = list(questions or [])

    def undo(self) -> Spec | None:
        """Go back to the previous spec. None if there is nothing to go back to."""
        if not self.history:
            return None
        self.spec = Spec.model_validate(self.history.pop())
        self.questions = []
        return self.spec

    def reset(self) -> None:
        """Start over. The old request goes to history, so this can be undone."""
        self.push_history()
        self.spec = None
        self.questions = []

    def record(self, user: str, turn_type, status: str) -> None:
        self.turns.append({"user": user, "turn_type": turn_type, "status": status})
        del self.turns[:-MAX_TURNS]

    def to_dict(self) -> dict:
        return {"spec": self.current(), "questions": self.questions, "history": self.history, "turns": self.turns}

    @classmethod
    def from_dict(cls, d: dict) -> "Session":
        spec = Spec.model_validate(d["spec"]) if d.get("spec") is not None else None
        return cls(spec, list(d.get("questions", [])), list(d.get("history", [])), list(d.get("turns", [])))
