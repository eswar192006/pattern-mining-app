import json

import pytest

from src.core import config as C
from src.core.errors import LLMUnavailable
from src.core.pipeline import Pipeline, error_code
from src.core.sessions import RateLimiter, SessionStore
from src.llm.fake import FakeLLM
from src.llm.session import Session
from tests.test_validation import BUNT, DF, REG

J = json.dumps
FREQ = J({"intent": "mining_request", "spec": {"kind": "frequent", "min_support": 0.3}})
ANSWER = (f"The pair bought together most often is {BUNT} and {REG}, in about 46% of orders (30 orders). "
          "Consider offering them as a bundle.")


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


# ======================================================================= session store
def test_sessions_are_created_found_and_deleted():
    store = SessionStore()
    a, b = store.create(), store.create()
    assert a != b and len(a) >= 20 and len(store) == 2                      # long, unguessable, unique ids
    assert store.get(a).session is not store.get(b).session
    assert store.get("nope") is None
    assert store.delete(a) and not store.delete(a) and store.get(a) is None and len(store) == 1


def test_idle_sessions_expire_and_use_keeps_them_alive():
    clock, store = Clock(), None
    store = SessionStore(ttl_s=100, clock=clock)
    keep, drop = store.create(), store.create()
    clock.t += 60
    assert store.get(keep) is not None                                       # touched at t+60
    clock.t += 60                                                            # drop is now 120 s idle, keep 60 s
    assert store.get(drop) is None and store.get(keep) is not None


def test_the_least_recently_used_session_is_dropped_when_full():
    clock = Clock()
    store = SessionStore(max_sessions=3, clock=clock)
    ids = []
    for _ in range(3):
        ids.append(store.create())
        clock.t += 1
    store.get(ids[0])                                                        # ids[0] becomes the most recent
    clock.t += 1
    new = store.create()
    assert len(store) == 3 and store.get(ids[1]) is None                      # ids[1] was the least recently used
    assert store.get(ids[0]) is not None and store.get(ids[2]) is not None and store.get(new) is not None


# ======================================================================= rate limiter
def test_rate_limiter_allows_a_burst_then_blocks_then_recovers():
    clock = Clock()
    limiter = RateLimiter(limit=3, window_s=60, clock=clock)
    assert [limiter.check("a") for _ in range(3)] == [None, None, None]
    assert limiter.check("a") == 60.0                                         # must wait the whole window
    assert limiter.check("b") is None                                         # other clients are unaffected
    clock.t += 45
    assert limiter.check("a") == 15.0
    clock.t += 16
    assert limiter.check("a") is None


# ======================================================================= pipeline
def pipeline(replies, **kw):
    return Pipeline(FakeLLM(replies), DF, **kw)


def test_one_message_produces_the_complete_answer():
    p, s = pipeline([FREQ, ANSWER]), Session()
    body = p.chat(s, "what sells together?")
    assert body["status"] == "ok" and body["turn_type"] == "new" and body["error"] is None
    assert body["request"] == {"kind": "frequent", "min_support": 0.3}
    assert body["changes"] == ["type of pattern: (not set) -> frequent", "minimum support: (not set) -> 0.3"]
    top = body["result"]["itemsets"][0]
    assert top["itemset"] == [BUNT, REG] and top["count"] == 30 and top["rank"] == 1
    assert body["answer"] == {"text": ANSWER, "source": "llm", "attempts": 1, "reason": None}
    assert set(body["timings"]) == {"language_s", "mining_s", "explain_s", "total_s"}
    assert body["session"] == {"current": {"kind": "frequent", "min_support": 0.3}, "turns": 1,
                               "can_undo": False, "pending_questions": 0}
    assert json.dumps(body)


def test_every_outcome_has_the_same_shape():
    shapes = []
    p, s = pipeline([FREQ, ANSWER, J({"intent": "out_of_scope", "message": "No."}), "garbage", "garbage",
                     J({"intent": "mining_request", "spec": {"kind": "frequent", "filters": {"include_products": ["cakestand"]}}}),
                     LLMUnavailable("down")]), Session()
    for text in ("a", "weather", "???", "cakestand", "down"):
        shapes.append(set(p.chat(s, text)))
    shapes.append(set(p.undo(s, explain=False))), shapes.append(set(p.reset(s))), shapes.append(set(p.run_current(Session())))
    assert len({frozenset(x) for x in shapes}) == 1                           # the interface never has to guess


def test_explain_can_be_switched_off():
    p, s = pipeline([FREQ]), Session()
    body = p.chat(s, "q", explain=False)
    assert body["result"] is not None and body["answer"] is None and "explain_s" not in body["timings"]


def test_a_follow_up_keeps_the_request_and_mines_the_new_data():
    sunday = J({"intent": "mining_request", "spec": {"filters": {"weekdays": ["Sunday"]}}})
    answer2 = f"On Sundays {BUNT} and {REG} are bought together in 100% of orders (30 orders)."
    p, s = pipeline([FREQ, ANSWER, sunday, answer2]), Session()
    p.chat(s, "q")
    body = p.chat(s, "only Sundays")
    assert body["turn_type"] == "update" and body["changes"] == ["weekdays: (not set) -> Sunday"]
    assert body["request"]["min_support"] == 0.3 and body["request"]["filters"]["weekdays"] == ["Sunday"]
    assert body["result"]["n_baskets_selected"] == 30 and body["answer"]["source"] == "llm"
    assert body["session"]["can_undo"] is True


def test_the_explanation_of_a_follow_up_is_not_asked_a_question_it_never_heard():
    llm = FakeLLM([FREQ, ANSWER, J({"intent": "mining_request", "spec": {"top_k": 3}}), ANSWER])
    p, s = Pipeline(llm, DF), Session()
    p.chat(s, "what sells together?")
    p.chat(s, "just three")
    texts = [c for c in llm.calls if c["mode"] == "text"]
    assert texts[0]["user"].startswith("Question: what sells together?")      # a new question: asked as typed
    assert not texts[1]["user"].startswith("Question:")                        # a follow-up: described by its settings


def test_questions_and_errors_never_start_mining():
    ask = J({"intent": "mining_request", "spec": {"kind": "frequent", "filters": {"include_products": ["cakestand"]}}})
    body = pipeline([ask]).chat(Session(), "cakestand")
    assert body["status"] == "needs_clarification" and body["result"] is None and body["answer"] is None
    assert body["questions"][0]["options"] and "mining_s" not in body["timings"]


def test_undo_reset_and_run_current():
    sunday = J({"intent": "mining_request", "spec": {"filters": {"weekdays": ["Sunday"]}}})
    p, s = pipeline([FREQ, sunday]), Session()
    assert p.run_current(s)["status"] == "nothing_to_run"

    p.chat(s, "q", explain=False)
    sundays = p.chat(s, "only Sundays", explain=False)
    assert sundays["result"]["n_baskets_selected"] == 30

    undone = p.undo(s, explain=False)
    assert undone["status"] == "ok" and undone["turn_type"] == "undo"
    assert undone["changes"] == ["weekdays: Sunday -> (not set)"]
    assert "filters" not in undone["request"]                                  # the Sunday filter is gone
    assert undone["result"]["n_baskets_selected"] == 65                        # ... and all days are mined again

    again = p.run_current(s, explain=False)
    assert again["turn_type"] == "rerun" and again["result"]["n_baskets_selected"] == 65

    reset = p.reset(s)
    assert reset["status"] == "reset" and reset["result"] is None and reset["request"] is None
    assert reset["session"]["current"] is None and reset["session"]["can_undo"] is True
    restored = p.undo(s, explain=False)                                        # a reset can be undone
    assert restored["status"] == "ok" and restored["request"]["kind"] == "frequent"
    assert p.undo(Session())["status"] == "nothing_to_undo"


def test_undo_also_writes_the_answer_when_asked():
    sunday = J({"intent": "mining_request", "spec": {"filters": {"weekdays": ["Sunday"]}}})
    p, s = pipeline([FREQ, sunday, ANSWER]), Session()
    p.chat(s, "q", explain=False), p.chat(s, "only Sundays", explain=False)
    body = p.undo(s)                                                           # all-days numbers: 46.15%, 30 orders
    assert body["answer"]["source"] == "llm" and body["answer"]["text"] == ANSWER


def test_mining_problems_are_reported_in_the_body_not_raised(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "SPMF_JAR", tmp_path / "missing.jar")
    body = pipeline([J({"intent": "mining_request", "spec": {"kind": "high_utility", "min_utility": 50}})]).chat(Session(), "q")
    assert body["status"] == "ok" and body["request"]["kind"] == "high_utility" and body["result"] is None
    assert body["error"]["code"] == "engine_unavailable" and "spmf.jar" in body["error"]["message"]

    monkeypatch.setattr(C, "MAX_RARE_RESULTS", 1)
    rare = J({"intent": "mining_request", "spec": {"kind": "rare", "max_support": 0.3, "min_count": 3, "min_strength": 0.2}})
    body = pipeline([rare]).chat(Session(), "q")
    assert body["error"]["code"] == "limit_exceeded" and body["result"] is None and body["answer"] is None


def test_a_busy_server_leaves_the_request_in_place_for_a_rerun():
    p, s = pipeline([FREQ, ANSWER], mining_slots=1, wait_s=0.05), Session()
    p._slots.acquire()                                                         # another job occupies the only slot
    body = p.chat(s, "q")
    assert body["status"] == "ok" and body["error"]["code"] == "server_busy" and body["result"] is None
    assert s.spec is not None                                                  # the request itself was kept
    p._slots.release()
    assert p.run_current(s)["result"]["total_found"] > 0


def test_warnings_are_listed_once():
    spec = J({"intent": "mining_request", "spec": {"kind": "frequent", "min_support": 0.3, "filters": {"weekend": True}}})
    sunday_answer = f"On weekends {BUNT} and {REG} are bought together in 100% of orders (30 orders)."
    body = pipeline([spec, sunday_answer]).chat(Session(), "weekend")
    assert body["answer"]["source"] == "llm"
    assert len(body["warnings"]) == len(set(body["warnings"])) and any("only Sunday" in w for w in body["warnings"])


def test_error_codes():
    from src.core.errors import EngineUnavailable, InvalidRequest, LimitExceeded
    assert [error_code(e) for e in (InvalidRequest("x"), LimitExceeded("x"), EngineUnavailable("x"),
                                    LLMUnavailable("x"), ValueError("x"))] == \
           ["invalid_request", "limit_exceeded", "engine_unavailable", "llm_unavailable", "error"]
