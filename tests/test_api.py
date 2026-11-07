import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from src.core import config as C
from src.core.errors import LLMUnavailable
from src.core.sessions import RateLimiter
from src.llm.client import GeminiClient
from src.llm.fake import FakeLLM
from tests.test_llm_client import KEY, server                      # noqa: F401  (the fake Google server fixture)
from tests.test_pipeline import ANSWER, FREQ, Clock
from tests.test_validation import BUNT, DF, REG

J = json.dumps
SUNDAY = J({"intent": "mining_request", "spec": {"filters": {"weekdays": ["Sunday"]}}})
ASK = J({"intent": "mining_request", "spec": {"kind": "frequent", "filters": {"include_products": ["cakestand"]}}})


def make(replies=(), **kw):
    llm = FakeLLM(list(replies))
    app = create_app(llm=llm, df=DF, **kw)
    return TestClient(app, raise_server_exceptions=False), llm, app


def error_of(r, status, code):
    assert r.status_code == status, r.text
    assert set(r.json()) == {"error"} and r.json()["error"]["code"] == code and r.json()["error"]["message"]


# ======================================================================= health and catalog
def test_health_and_catalog():
    c, _, _ = make()
    h = c.get("/health").json()
    assert h["status"] == "ok" and h["data"] == {"ready": True, "baskets": 65, "error": None}
    assert h["llm"]["ready"] and h["llm"]["model"] == "fake" and h["sessions"] == 0
    assert set(h["mining"]) == {"java", "spmf_jar", "high_utility_available"}
    cat = c.get("/catalog").json()
    assert cat["baskets"] == 65 and cat["products"] == 6 and "Saturday" not in cat["weekdays_with_orders"]
    assert set(cat["kinds"]) == {"frequent", "rare", "high_utility"} and cat["presets"]["frequent"]["strict"]
    assert cat["limits"]["min_support_floor"] == C.MIN_SUPPORT_FLOOR


# ======================================================================= a conversation over HTTP
def test_a_conversation_over_http():
    c, _, _ = make([FREQ, ANSWER, SUNDAY])
    r1 = c.post("/chat", json={"message": "what sells together?"})
    b1 = r1.json()
    sid = b1["session_id"]
    assert r1.status_code == 200 and b1["status"] == "ok" and b1["result"]["itemsets"][0]["itemset"] == [BUNT, REG]
    assert b1["answer"]["source"] == "llm" and len(sid) >= 20

    b2 = c.post("/chat", json={"message": "only Sundays", "session_id": sid, "explain": False}).json()
    assert b2["session_id"] == sid and b2["turn_type"] == "update" and b2["answer"] is None
    assert b2["request"] == {"kind": "frequent", "min_support": 0.3, "filters": {"weekdays": ["Sunday"]}}

    state = c.get(f"/sessions/{sid}").json()
    assert state["current"] == {"kind": "frequent", "min_support": 0.3, "filters": {"weekdays": ["Sunday"]}}
    assert state["can_undo"] is True and len(state["turns"]) == 2

    undone = c.post(f"/sessions/{sid}/undo", params={"explain": "false"}).json()
    assert undone["status"] == "ok" and "filters" not in undone["request"]
    reset = c.post(f"/sessions/{sid}/reset").json()
    assert reset["status"] == "reset" and reset["session"]["current"] is None
    assert c.post(f"/sessions/{sid}/run").json()["status"] == "nothing_to_run"
    assert c.delete(f"/sessions/{sid}").json() == {"deleted": True}
    error_of(c.get(f"/sessions/{sid}"), 404, "session_not_found")


def test_sessions_can_be_created_first_and_unknown_ids_are_404():
    c, _, _ = make([FREQ])
    created = c.post("/sessions")
    assert created.status_code == 201
    sid = created.json()["session_id"]
    assert c.post("/chat", json={"message": "q", "session_id": sid, "explain": False}).json()["session_id"] == sid
    for method, path in (("get", "/sessions/bogus"), ("delete", "/sessions/bogus"), ("post", "/sessions/bogus/undo"),
                         ("post", "/sessions/bogus/reset"), ("post", "/sessions/bogus/run")):
        error_of(getattr(c, method)(path), 404, "session_not_found")
    error_of(c.post("/chat", json={"message": "q", "session_id": "bogus"}), 404, "session_not_found")


def test_a_clarifying_question_is_answered_on_the_next_message():
    answer = J({"intent": "mining_request", "spec": {"filters": {"include_products": [REG]}}})
    c, _, _ = make([ASK, answer])
    b1 = c.post("/chat", json={"message": "what sells with the cakestand?", "explain": False}).json()
    assert b1["status"] == "needs_clarification" and b1["result"] is None and b1["questions"][0]["options"][0] == REG
    assert b1["session"]["pending_questions"] == 1
    b2 = c.post("/chat", json={"message": "the regency one", "session_id": b1["session_id"], "explain": False}).json()
    assert b2["status"] == "ok" and b2["result"] is not None and b2["session"]["pending_questions"] == 0


def test_off_topic_messages_are_a_normal_result():
    c, _, _ = make([J({"intent": "out_of_scope", "message": "Only sales patterns, sorry."})])
    r = c.post("/chat", json={"message": "weather?"})
    assert r.status_code == 200 and r.json()["status"] == "out_of_scope" and r.json()["message"] == "Only sales patterns, sorry."
    assert r.json()["result"] is None


# ======================================================================= bad input
@pytest.mark.parametrize("body", [{}, {"message": ""}, {"message": "x" * (C.MAX_QUESTION_CHARS + 1)},
                                  {"message": "hi", "surprise": 1}, {"message": 5}, {"message": "hi", "explain": "maybe"}])
def test_bad_input_is_rejected_before_anything_is_spent(body):
    c, llm, _ = make()
    error_of(c.post("/chat", json=body), 422, "invalid_input")
    assert llm.calls == []                                                      # no model call, no money


def test_a_body_that_is_not_json_is_rejected():
    c, llm, _ = make()
    error_of(c.post("/chat", content="not json", headers={"Content-Type": "application/json"}), 422, "invalid_input")


# ======================================================================= things that go wrong
def test_the_model_being_down_gives_503_and_keeps_the_conversation():
    c, _, _ = make([FREQ, LLMUnavailable("Gemini is down.")])
    sid = c.post("/chat", json={"message": "q", "explain": False}).json()["session_id"]
    r = c.post("/chat", json={"message": "only Sundays", "session_id": sid})
    assert r.status_code == 503 and r.json()["status"] == "llm_unavailable" and r.json()["message"] == "Gemini is down."
    assert c.get(f"/sessions/{sid}").json()["current"] == {"kind": "frequent", "min_support": 0.3}


def test_a_missing_key_or_missing_data_degrades_gracefully(monkeypatch, tmp_path):
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(C, "ROOT", tmp_path)
    c = TestClient(create_app(df=DF), raise_server_exceptions=False)             # no model client could be created
    h = c.get("/health").json()
    assert h["status"] == "degraded" and not h["llm"]["ready"] and "GEMINI_API_KEY" in h["llm"]["error"]
    r = c.post("/chat", json={"message": "q"})
    error_of(r, 503, "service_unavailable")
    assert "GEMINI_API_KEY" in r.json()["error"]["message"] and "Never paste" in r.json()["error"]["message"]
    error_of(c.get("/catalog"), 503, "service_unavailable")
    assert c.post("/sessions").status_code == 201                                  # not everything needs the model

    import src.mining.engine as engine

    def missing():
        raise FileNotFoundError("Run `python -m src.prepare_data` first.")
    monkeypatch.setattr(engine, "get_data", missing)
    c = TestClient(create_app(llm=FakeLLM([])), raise_server_exceptions=False)
    assert c.get("/health").json()["data"] == {"ready": False, "baskets": None, "error": "Run `python -m src.prepare_data` first."}
    assert "prepare_data" in c.post("/chat", json={"message": "q"}).json()["error"]["message"]


def test_mining_that_cannot_run_is_reported_in_the_body(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "SPMF_JAR", tmp_path / "missing.jar")
    hu = J({"intent": "mining_request", "spec": {"kind": "high_utility", "min_utility": 50}})
    c, _, _ = make([hu])
    r = c.post("/chat", json={"message": "earners"})
    assert r.status_code == 503 and r.json()["status"] == "ok" and r.json()["error"]["code"] == "engine_unavailable"
    assert r.json()["request"]["kind"] == "high_utility" and r.json()["result"] is None

    monkeypatch.setattr(C, "MAX_RARE_RESULTS", 1)
    rare = J({"intent": "mining_request", "spec": {"kind": "rare", "max_support": 0.3, "min_count": 3, "min_strength": 0.2}})
    c, _, _ = make([rare])
    r = c.post("/chat", json={"message": "rare"})
    assert r.status_code == 200 and r.json()["error"]["code"] == "limit_exceeded"        # the user can fix this one


def test_a_busy_server_says_so_and_the_request_can_be_rerun():
    c, _, app = make([FREQ, ANSWER])
    pipeline = app.state.pipeline
    pipeline._wait_s = 0.05
    for _ in range(C.MAX_CONCURRENT_MINING):
        pipeline._slots.acquire()
    r = c.post("/chat", json={"message": "q"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "server_busy" and r.json()["request"]["kind"] == "frequent"
    for _ in range(C.MAX_CONCURRENT_MINING):
        pipeline._slots.release()
    again = c.post(f"/sessions/{r.json()['session_id']}/run")
    assert again.status_code == 200 and again.json()["result"]["total_found"] > 0 and again.json()["answer"]["source"] == "llm"


def test_rate_limit():
    clock = Clock()
    c, _, _ = make([FREQ] + [J({"intent": "mining_request", "spec": {"top_k": n}}) for n in (3, 4, 5)],
                   limiter=RateLimiter(limit=3, window_s=60, clock=clock))
    sid = c.post("/chat", json={"message": "q", "explain": False}).json()["session_id"]
    for _ in range(2):
        assert c.post("/chat", json={"message": "fewer", "session_id": sid, "explain": False}).status_code == 200
    r = c.post("/chat", json={"message": "fewer", "session_id": sid, "explain": False})
    error_of(r, 429, "rate_limited")
    assert int(r.headers["Retry-After"]) >= 1 and "wait" in r.json()["error"]["message"]
    clock.t += 61
    assert c.post("/chat", json={"message": "fewer", "session_id": sid, "explain": False}).status_code == 200


def test_a_bug_never_leaks_internal_details(monkeypatch):
    c, _, app = make()

    def boom(*a, **k):
        raise RuntimeError("secret internal detail /home/user/file.py")
    monkeypatch.setattr(app.state.pipeline, "chat", boom)
    r = c.post("/chat", json={"message": "q"})
    error_of(r, 500, "internal_error")
    assert "secret" not in r.text and "file.py" not in r.text


def test_requests_in_one_conversation_are_handled_one_at_a_time():
    state = {"active": 0, "max": 0}
    guard = threading.Lock()

    def slow(reply):
        def f(system, user):
            with guard:
                state["active"] += 1
                state["max"] = max(state["max"], state["active"])
            time.sleep(0.15)
            with guard:
                state["active"] -= 1
            return reply
        return f

    c, _, app = make([slow(FREQ), slow(J({"intent": "mining_request", "spec": {"top_k": 3}}))])
    sid = c.post("/sessions").json()["session_id"]
    results = []

    def post():
        results.append(TestClient(app).post("/chat", json={"message": "m", "session_id": sid, "explain": False}).status_code)

    threads = [threading.Thread(target=post) for _ in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results == [200, 200] and state["max"] == 1


# ======================================================================= secrets, CORS, docs
def test_the_api_key_is_never_returned(server):
    server.script = [(200, FREQ, 0), (200, ANSWER, 0)]
    c = TestClient(create_app(llm=GeminiClient(api_key=KEY, base_url=server.url), df=DF), raise_server_exceptions=False)
    r = c.post("/chat", json={"message": "what sells together?"})
    assert r.status_code == 200 and r.json()["answer"]["source"] == "llm" and len(server.requests) == 2
    assert KEY not in r.text and KEY not in json.dumps(dict(r.headers))
    assert KEY not in c.get("/health").text and KEY not in c.get("/catalog").text

    server.script = [(403, f"API key {KEY} not valid", 0)]
    r = c.post("/chat", json={"message": "again", "session_id": r.json()["session_id"]})
    assert r.status_code == 503 and "API key" in r.json()["message"] and KEY not in r.text


def test_cors_allows_the_interface_origin_only():
    c, _, _ = make()
    ok = c.get("/health", headers={"Origin": "http://localhost:8501"})
    assert ok.headers["access-control-allow-origin"] == "http://localhost:8501"
    assert "access-control-allow-origin" not in c.get("/health", headers={"Origin": "http://evil.example"}).headers


def test_documentation_is_available():
    c, _, _ = make()
    paths = set(c.get("/openapi.json").json()["paths"])
    assert {"/chat", "/health", "/catalog", "/sessions", "/sessions/{session_id}/undo",
            "/sessions/{session_id}/reset", "/sessions/{session_id}/run"} <= paths
    assert c.get("/docs").status_code == 200


# ======================================================================= real data
real = pytest.mark.skipif(not C.CLEAN_FILE.exists(), reason="run prepare_data first")


@real
def test_a_real_conversation_over_http():
    def honest(system, user):                                   # a model that writes only from the facts it is given
        facts = json.loads(user.split("FACTS:\n", 1)[1])
        row = facts["results"][0]
        return (f"The most common pairing is {' and '.join(row['products'])}, found in about "
                f"{row['share_of_orders_pct']:.1f}% of orders ({row['orders']} orders).")

    llm = FakeLLM([J({"intent": "mining_request", "spec": {"kind": "frequent", "strictness": "strict"}}), honest,
                   J({"intent": "mining_request", "spec": {"filters": {"weekdays": ["Sunday"]}}}), honest])
    c = TestClient(create_app(llm=llm), raise_server_exceptions=False)
    assert c.get("/health").json()["data"]["baskets"] == 19618
    b1 = c.post("/chat", json={"message": "which products sell together?"}).json()
    assert b1["result"]["itemsets"][0]["itemset"] == ["JUMBO BAG PINK POLKADOT", "JUMBO BAG RED RETROSPOT"]
    assert b1["result"]["n_baskets_mined"] == 18141 and b1["answer"]["source"] == "llm"
    b2 = c.post("/chat", json={"message": "only on Sundays", "session_id": b1["session_id"]}).json()
    assert b2["result"]["n_baskets_selected"] == 2190 and b2["request"]["min_support"] == 0.02
    assert b2["result"]["itemsets"][0]["itemset"] != b1["result"]["itemsets"][0]["itemset"]
    assert b2["answer"]["source"] == "llm" and json.dumps(b2)
