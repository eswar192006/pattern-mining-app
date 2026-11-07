"""Multi-turn conversation: converse(session, message, llm) -> TurnResult.

    session = Session()
    converse(session, "which products sell together?", llm)      # first question (Step 8 parser)
    converse(session, "only on Sundays", llm)                    # follow-up: an EDIT of the request
    converse(session, "now by money", llm)                       # kind switches, Sunday filter stays
    converse(session, "undo", llm)                               # back to the frequent-patterns request

The model only says WHAT CHANGES; src/llm/edit.py applies it, so every setting the user did not mention
is guaranteed to stay as it was. A bad edit gets one self-correction attempt, like in Step 8. The session
is only changed by an edit that validates; a failed turn never damages the request the user already has.
"""
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from pydantic import ValidationError

from src.core import config as C
from src.core.catalog import Catalog, get_catalog
from src.core.errors import LLMUnavailable
from src.core.spec import Spec
from src.core.validation import ValidationResult, validate_spec
from src.llm import prompts
from src.llm.edit import apply_edit, diff_specs
from src.llm.parser import (FAILED_MESSAGE, LOG_DEFAULT, OUT_OF_SCOPE_MESSAGE, _extract_json, _log, _prune_nulls,
                            parse_question)
from src.llm.reply import FollowUpReply, followup_json_schema
from src.llm.session import Session

RESET_MESSAGE = "Starting over. What would you like to look at?"
NOTHING_TO_UNDO = "There is nothing to undo."
NO_CHANGE = "That did not change the request."


@dataclass
class TurnResult:
    status: str                    # ok | needs_clarification | invalid | out_of_scope | failed |
                                   # llm_unavailable | reset | nothing_to_undo
    turn_type: str | None = None   # new | update | undo | reset
    message: str | None = None
    spec: Spec | None = None
    validation: ValidationResult | None = None
    changes: list = field(default_factory=list)       # what changed, in words
    notes: list = field(default_factory=list)         # e.g. thresholds dropped after a kind switch
    attempts: int = 0
    problems: list = field(default_factory=list)
    raw_replies: list = field(default_factory=list)
    model: str = ""
    latency_s: float = 0.0

    def to_dict(self) -> dict:
        return {"status": self.status, "turn_type": self.turn_type, "message": self.message,
                "spec": self.spec.model_dump(exclude_defaults=True) if self.spec else None,
                "validation": self.validation.to_dict() if self.validation else None,
                "changes": self.changes, "notes": self.notes, "attempts": self.attempts, "problems": self.problems,
                "model": self.model, "latency_s": round(self.latency_s, 3)}


def read_followup(raw: str):
    """-> (FollowUpReply or None, problem text or None)"""
    import json
    if not raw or not raw.strip():
        return None, "The reply was empty."
    try:
        data = _extract_json(raw)
    except json.JSONDecodeError as e:
        return None, f"The reply was not valid JSON ({e.msg})."
    if not isinstance(data, dict):
        return None, "The reply must be a single JSON object."
    try:
        return FollowUpReply.model_validate(_prune_nulls(data)), None
    except ValidationError as e:
        parts = [f"{'.'.join(str(p) for p in err['loc']) or 'reply'}: {err['msg']}" for err in e.errors()[:5]]
        return None, "The JSON did not match the required format: " + "; ".join(parts) + "."


def _pending(validation) -> list:
    return [q.to_dict() for q in validation.questions] if validation.status == "needs_clarification" else []


def _catalog_for(catalog, df):
    if catalog is None and df is not None:
        catalog = Catalog(df)
    return catalog if catalog is not None else (get_catalog() if df is None else None)


def _apply_reset(session: Session, result: TurnResult) -> None:
    session.reset()
    result.status, result.turn_type, result.message = "reset", "reset", RESET_MESSAGE


def _apply_undo(session: Session, result: TurnResult, df, cat) -> None:
    before = session.spec
    restored = session.undo()
    if restored is None:
        result.status, result.message = "nothing_to_undo", NOTHING_TO_UNDO
        return
    v = validate_spec(restored, df=df, catalog=cat)
    session.questions = _pending(v)
    result.status, result.turn_type, result.spec, result.validation = v.status, "undo", restored, v
    result.changes = diff_specs(before, restored)


def reset_session(session: Session) -> TurnResult:
    """Start over without asking the model (used by the API and the chat demo)."""
    result = TurnResult("failed")
    _apply_reset(session, result)
    session.record("/reset", result.turn_type, result.status)
    return result


def undo_session(session: Session, catalog: Catalog | None = None, df=None) -> TurnResult:
    """Go back one request without asking the model (used by the API and the chat demo)."""
    result = TurnResult("failed")
    _apply_undo(session, result, df, _catalog_for(catalog, df))
    session.record("/undo", result.turn_type, result.status)
    return result


def converse(session: Session, message: str, llm, catalog: Catalog | None = None, df=None,
             log_path=LOG_DEFAULT, max_attempts: int = 2) -> TurnResult:
    t0 = time.perf_counter()
    if log_path is LOG_DEFAULT:
        log_path = C.LLM_LOG_FILE
    text = (message or "").strip()
    result = TurnResult("failed", model=getattr(llm, "model", ""))

    def done():
        result.latency_s = time.perf_counter() - t0
        session.record(text, result.turn_type, result.status)
        _log(log_path, {"type": "turn", "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "prompt_version": prompts.PROMPT_VERSION,
                        "user_message": text, **{k: v for k, v in result.to_dict().items() if k != "validation"},
                        "validation_status": result.validation.status if result.validation else None,
                        "raw_replies": [t[:3000] for t in result.raw_replies]})
        return result

    if not text:
        result.message = "Please type a message."
        return done()
    if len(text) > C.MAX_QUESTION_CHARS:
        result.message = f"Please keep the message under {C.MAX_QUESTION_CHARS:,} characters."
        return done()

    cat = _catalog_for(catalog, df)

    # ---------------------------------------------------------------- first message: the Step 8 parser
    if not session.has_context:
        p = parse_question(text, llm, catalog=cat, df=df, log_path=None, max_attempts=max_attempts)
        result.status, result.message, result.attempts = p.status, p.message, p.attempts
        result.problems, result.raw_replies, result.validation = p.problems, p.raw_replies, p.validation
        if p.status in ("ok", "needs_clarification"):
            result.turn_type, result.spec = "new", p.spec
            session.set_spec(p.spec, _pending(p.validation))
            result.changes = diff_specs(None, p.spec)
        return done()

    # ---------------------------------------------------------------- follow-up: an edit of the current request
    system, schema = prompts.build_followup_system_prompt(cat), followup_json_schema()
    state = prompts.followup_user_prompt(session.current(), session.questions, text)
    user, last_validation = state, None

    for attempt in range(1, max_attempts + 1):
        try:
            raw = llm.generate_json(system, user, schema)
        except LLMUnavailable as e:
            result.status, result.message, result.attempts = "llm_unavailable", str(e), attempt
            return done()
        result.attempts = attempt
        result.raw_replies.append(raw)
        reply, problem = read_followup(raw)

        if reply is not None:
            if reply.intent == "out_of_scope":
                result.status = "out_of_scope"
                result.message = (reply.message or "").strip() or OUT_OF_SCOPE_MESSAGE
                return done()
            if reply.intent == "reset":
                _apply_reset(session, result)
                return done()
            if reply.intent == "undo":
                _apply_undo(session, result, df, cat)
                return done()

            new_spec, notes, edit_problems = apply_edit(session.spec, reply)
            if edit_problems:
                problem = " ".join(edit_problems)
            else:
                v = validate_spec(new_spec, df=df, catalog=cat)
                if v.status == "invalid":
                    last_validation = v
                    problem = "The edited request contradicts itself: " + " ".join(v.errors)
                else:
                    old = session.spec
                    result.turn_type = "new" if (reply.fresh or old is None) else "update"
                    result.status, result.spec, result.validation, result.notes = v.status, new_spec, v, notes
                    result.changes = diff_specs(old, new_spec)
                    session.set_spec(new_spec, _pending(v))
                    if not result.changes and v.status == "ok":
                        result.message = NO_CHANGE
                    return done()

        result.problems.append(problem)
        if attempt < max_attempts:
            user = prompts.followup_correction_prompt(state, raw, [problem])

    # out of attempts: the session keeps the request the user already had
    if last_validation is not None:
        result.status, result.validation = "invalid", last_validation
        result.message = " ".join(last_validation.errors)
    else:
        result.status, result.message = "failed", FAILED_MESSAGE
    return done()
