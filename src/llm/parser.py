"""English question -> checked request (or the question to ask the user).

    result = parse_question("what sells together on Sundays?", llm)
    result.status     ok | needs_clarification | invalid | out_of_scope | failed | llm_unavailable
    result.validation the ValidationResult (request / questions / warnings / notes) when there is one
    result.message    plain-language text for the user when there is no request to run

Flow:  model -> JSON -> LLMReply -> validate_spec().
If the JSON is malformed, or validation finds a contradiction, the problem is sent back to
the model ONCE for self-correction (the technique from the text-to-SQL papers in the
literature survey). If it still fails the user is asked to rephrase. A question that merely
needs clarification is NOT retried: it goes to the user, who knows the answer.
"""
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from src.core import config as C
from src.core.catalog import Catalog, get_catalog
from src.core.errors import LLMUnavailable
from src.core.spec import Spec
from src.core.validation import ValidationResult, validate_spec
from src.llm import prompts
from src.llm.reply import LLMReply, llm_json_schema

LOG_DEFAULT = object()          # "use the configured log file" (looked up at call time, so tests can redirect it)

OUT_OF_SCOPE_MESSAGE = ("I can only help with questions about which products sell together, unusual product "
                        "combinations, and which combinations earn the most money.")
FAILED_MESSAGE = ("I could not turn that into a request I can run. Could you rephrase it? For example: "
                  "\"Which products are often bought together on Sundays?\"")


@dataclass
class ParseResult:
    status: str
    message: str | None = None
    spec: Spec | None = None
    validation: ValidationResult | None = None
    attempts: int = 0
    raw_replies: list = field(default_factory=list)
    problems: list = field(default_factory=list)       # what was wrong with replies we had to retry
    model: str = ""
    latency_s: float = 0.0

    def to_dict(self) -> dict:
        return {"status": self.status, "message": self.message,
                "spec": self.spec.model_dump(exclude_defaults=True) if self.spec else None,
                "validation": self.validation.to_dict() if self.validation else None,
                "attempts": self.attempts, "problems": self.problems,
                "model": self.model, "latency_s": round(self.latency_s, 3)}


# ------------------------------------------------------------------ reading the reply
def _prune_nulls(node):
    """Models often write null for 'not given'. For every field of the spec, null means the same as absent."""
    if isinstance(node, dict):
        return {k: _prune_nulls(v) for k, v in node.items() if v is not None}
    if isinstance(node, list):
        return [_prune_nulls(v) for v in node]
    return node


def _extract_json(raw: str):
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def read_reply(raw: str):
    """-> (LLMReply or None, problem text or None)"""
    if not raw or not raw.strip():
        return None, "The reply was empty."
    try:
        data = _extract_json(raw)
    except json.JSONDecodeError as e:
        return None, f"The reply was not valid JSON ({e.msg})."
    if not isinstance(data, dict):
        return None, "The reply must be a single JSON object."
    data = _prune_nulls(data)
    try:
        return LLMReply.model_validate(data), None
    except ValidationError as e:
        parts = [f"{'.'.join(str(p) for p in err['loc']) or 'reply'}: {err['msg']}" for err in e.errors()[:5]]
        return None, "The JSON did not match the required format: " + "; ".join(parts) + "."


# ------------------------------------------------------------------ logging
def _log(path, record):
    if path is None:
        return
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError:
        pass                                              # logging must never break an answer


def _record(question, r: ParseResult):
    return {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "question": question,
            "prompt_version": prompts.PROMPT_VERSION,
            "status": r.status, "attempts": r.attempts, "model": r.model, "latency_s": round(r.latency_s, 3),
            "problems": r.problems, "raw_replies": [t[:3000] for t in r.raw_replies],
            "spec": r.spec.model_dump(exclude_defaults=True) if r.spec else None,
            "validation": r.validation.to_dict() if r.validation else None, "message": r.message}


# ------------------------------------------------------------------ the main function
def parse_question(question: str, llm, catalog: Catalog | None = None, df=None,
                   log_path=LOG_DEFAULT, max_attempts: int = 2) -> ParseResult:
    t0 = time.perf_counter()
    if log_path is LOG_DEFAULT:
        log_path = C.LLM_LOG_FILE
    result = ParseResult("failed", model=getattr(llm, "model", ""))

    def done():
        result.latency_s = time.perf_counter() - t0
        _log(log_path, _record(question, result))
        return result

    text = (question or "").strip()
    if not text:
        result.message = "Please type a question."
        return done()
    if len(text) > C.MAX_QUESTION_CHARS:
        result.message = f"Please keep the question under {C.MAX_QUESTION_CHARS:,} characters."
        return done()

    if catalog is None and df is not None:
        catalog = Catalog(df)
    cat = catalog if catalog is not None else (get_catalog() if df is None else None)
    system, schema = prompts.build_system_prompt(cat), llm_json_schema()
    user, previous, problems = prompts.user_prompt(text), "", []

    for attempt in range(1, max_attempts + 1):
        try:
            raw = llm.generate_json(system, user, schema)
        except LLMUnavailable as e:
            result.status, result.message = "llm_unavailable", str(e)
            result.attempts = attempt
            return done()
        result.attempts = attempt
        result.raw_replies.append(raw)
        previous = raw

        reply, problem = read_reply(raw)
        validation = None
        if reply is not None and reply.intent == "out_of_scope":
            result.status = "out_of_scope"
            result.message = (reply.message or "").strip() or OUT_OF_SCOPE_MESSAGE
            return done()
        if reply is not None:
            result.spec = reply.spec
            validation = validate_spec(reply.spec, df=df, catalog=cat)
            if validation.status != "invalid":
                result.status, result.validation = validation.status, validation
                return done()
            problem = "The request contradicts itself: " + " ".join(validation.errors)
            result.validation = validation

        result.problems.append(problem)
        if attempt < max_attempts:                         # ask the model to fix exactly this problem
            user = prompts.correction_prompt(text, previous, [problem])

    # out of attempts
    if result.validation is not None and result.validation.status == "invalid":
        result.status = "invalid"
        result.message = " ".join(result.validation.errors)
    else:
        result.status, result.message = "failed", FAILED_MESSAGE
    return done()
