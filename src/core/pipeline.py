"""One message in, one complete answer out:

    body = Pipeline(llm, df).chat(session, "which products sell together on Sundays?")

    converse (Gemini: sentence -> edit)  ->  validate  ->  mine (exact)  ->  explain (Gemini, verified)

Everything the API returns comes from here, in ONE stable JSON shape, so the interface never has to
guess. Domain outcomes (a question to answer, a request that is too heavy, an off-topic message) are
normal results with a `status` and/or `error`, not exceptions. Only genuine bugs raise.
"""
import threading
import time

from src.core import config as C
from src.core.catalog import Catalog
from src.core.errors import EngineUnavailable, InvalidRequest, LimitExceeded, LLMUnavailable, MiningError
from src.core.validation import validate_spec
from src.llm.conversation import TurnResult, converse, reset_session, undo_session
from src.llm.explainer import explain_result
from src.llm.session import Session
from src.mining.engine import run_mining

_CODES = {InvalidRequest: "invalid_request", LimitExceeded: "limit_exceeded",
          EngineUnavailable: "engine_unavailable", LLMUnavailable: "llm_unavailable"}
RESULT_KEYS = ("kind", "n_baskets_selected", "n_baskets_mined", "total_found", "shown", "itemsets", "hint",
               "notes", "filter_trace", "params", "timings")


def error_code(e: Exception) -> str:
    return next((code for cls, code in _CODES.items() if isinstance(e, cls)), "error")


def _dedupe(items):
    return list(dict.fromkeys(items))


class Pipeline:
    def __init__(self, llm, df, catalog: Catalog | None = None, mining_slots: int = C.MAX_CONCURRENT_MINING,
                 wait_s: float = C.MINING_WAIT_S):
        self.llm, self.df = llm, df
        self.catalog = catalog if catalog is not None else Catalog(df)
        self._slots = threading.BoundedSemaphore(mining_slots)
        self._wait_s = wait_s

    # ------------------------------------------------------------------ the operations
    def chat(self, session: Session, message: str, explain: bool = True) -> dict:
        t0 = time.perf_counter()
        turn = converse(session, message, self.llm, catalog=self.catalog, df=self.df)
        return self._respond(session, turn, message if turn.turn_type == "new" else None, explain, t0)

    def undo(self, session: Session, explain: bool = True) -> dict:
        t0 = time.perf_counter()
        return self._respond(session, undo_session(session, catalog=self.catalog, df=self.df), None, explain, t0)

    def reset(self, session: Session) -> dict:
        t0 = time.perf_counter()
        return self._respond(session, reset_session(session), None, False, t0)

    def run_current(self, session: Session, explain: bool = True) -> dict:
        """Mine the current request again, without a new message (for example after a busy server)."""
        t0 = time.perf_counter()
        if session.spec is None:
            turn = TurnResult("nothing_to_run", message="There is no request yet. Ask a question first.")
        else:
            v = validate_spec(session.spec, df=self.df, catalog=self.catalog)
            turn = TurnResult(v.status, turn_type="rerun", spec=session.spec, validation=v)
        return self._respond(session, turn, None, explain, t0)

    # ------------------------------------------------------------------ building the response
    def _respond(self, session: Session, turn: TurnResult, question, explain: bool, t0: float) -> dict:
        v = turn.validation
        body = {
            "status": turn.status, "turn_type": turn.turn_type, "message": turn.message,
            "request": v.request.model_dump(exclude_defaults=True) if v and v.request else None,
            "changes": turn.changes,
            "notes": _dedupe(list(turn.notes) + (v.notes if v else [])),
            "warnings": _dedupe(v.warnings) if v else [],
            "questions": [q.to_dict() for q in v.questions] if v else [],
            "errors": list(v.errors) if v else [],
            "result": None, "answer": None, "error": None,
            "session": {"current": session.current(), "turns": len(session.turns),
                        "can_undo": bool(session.history), "pending_questions": len(session.questions)},
            "timings": {"language_s": round(turn.latency_s, 3)},
        }
        if turn.status == "ok" and v is not None and v.request is not None:
            self._mine_into(body, v.request, question, explain)
        body["timings"]["total_s"] = round(time.perf_counter() - t0, 3)
        return body

    def _mine_into(self, body: dict, request, question, explain: bool) -> None:
        if not self._slots.acquire(timeout=self._wait_s):
            body["error"] = {"code": "server_busy", "message": "The server is busy with other requests. "
                                                              "Please try again in a moment."}
            return
        try:
            t = time.perf_counter()
            result = run_mining(request, df=self.df)
            body["timings"]["mining_s"] = round(time.perf_counter() - t, 3)
        except MiningError as e:
            body["error"] = {"code": error_code(e), "message": str(e)}
            return
        finally:
            self._slots.release()                      # the model call below must not hold a mining slot

        d = result.to_dict()
        body["result"] = {k: d[k] for k in RESULT_KEYS}
        body["warnings"] = _dedupe(body["warnings"] + result.warnings)
        if explain:
            t = time.perf_counter()
            e = explain_result(result, self.llm, question=question)
            body["answer"] = {"text": e.text, "source": e.source, "attempts": e.attempts, "reason": e.reason}
            body["timings"]["explain_s"] = round(time.perf_counter() - t, 3)
