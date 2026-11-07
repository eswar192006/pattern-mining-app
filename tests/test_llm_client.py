"""GeminiClient against a FAKE local Google server. The real google-genai SDK is used, so
this checks the actual HTTP request we send (against Google's documented Interactions API)
and every failure path, with no network and no API key."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from src.core import config as C
from src.core.errors import LLMUnavailable
from src.llm.client import GeminiClient
from src.llm.reply import llm_json_schema

KEY = "SECRET-KEY-XYZ-123"
GOOD = '{"intent":"mining_request","spec":{"kind":"frequent"}}'


class FakeGemini:
    """Replies from `script` in order (status, text/error message, delay_s); 200 + GOOD when empty."""

    def __init__(self):
        self.requests, self.script = [], []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                status, text, delay = outer.script.pop(0) if outer.script else (200, GOOD, 0)
                if delay:
                    time.sleep(delay)
                if status == 200:
                    payload = {"id": "int_1", "status": "completed", "steps": [
                        {"type": "model_output", "status": "done", "content": [{"type": "text", "text": text}]}]
                        if text is not None else []}
                else:
                    payload = {"error": {"code": status, "message": text, "status": "ERROR"}}
                data = json.dumps(payload).encode()
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def client(self, **kw):
        return GeminiClient(api_key=KEY, base_url=self.url, **kw)


@pytest.fixture
def server():
    s = FakeGemini()
    yield s
    s.httpd.shutdown()


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(C, "ROOT", tmp_path)            # so a real .env of the user is never read


def test_request_matches_googles_documentation(server):
    text = server.client().generate_json("SYSTEM TEXT", "Question: hi", llm_json_schema())
    assert text == GOOD
    req = server.requests[0]
    assert req["path"].endswith("/interactions")
    assert req["headers"]["x-goog-api-key"] == KEY
    body = req["body"]
    assert body["model"] == C.GEMINI_MODEL_DEFAULT
    # the instructions travel at the start of the input (system_instruction was ignored by the live API)
    assert body["input"] == "SYSTEM TEXT\n\nQuestion: hi" and "system_instruction" not in body
    assert body["store"] is False                                   # do not keep the conversation server-side
    assert body["response_format"] == {"type": "text", "mime_type": "application/json", "schema": llm_json_schema()}
    assert body["generation_config"]["temperature"] == 0
    assert body["generation_config"]["thinking_level"] == "low"


def test_model_name_comes_from_the_environment(server, monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "my-future-model")
    c = server.client()
    c.generate_json("s", "u", None)
    assert c.model == "my-future-model" and server.requests[0]["body"]["model"] == "my-future-model"
    assert "schema" not in server.requests[0]["body"]["response_format"]       # schema=None -> JSON mode only


def test_falls_back_when_model_rejects_generation_settings(server):
    server.script = [(400, "thinking_level is not supported by this model", 0)]
    c = server.client()
    assert c.generate_json("s", "u", llm_json_schema()) == GOOD
    assert "generation_config" in server.requests[0]["body"]
    assert "generation_config" not in server.requests[1]["body"] and "schema" in server.requests[1]["body"]["response_format"]
    c.generate_json("s", "u", llm_json_schema())                      # remembers what works: no second 400
    assert len(server.requests) == 3 and "generation_config" not in server.requests[2]["body"]


def test_falls_back_to_json_mode_without_schema(server):
    server.script = [(400, "schema too complex", 0), (400, "schema too complex", 0)]
    assert server.client().generate_json("s", "u", llm_json_schema()) == GOOD
    assert "schema" not in server.requests[2]["body"]["response_format"]


def test_everything_rejected_gives_a_friendly_error(server):
    server.script = [(400, "bad request A", 0)] * 3
    with pytest.raises(LLMUnavailable, match="rejected the request"):
        server.client().generate_json("s", "u", llm_json_schema())


def test_bad_key_message_never_leaks_the_key(server):
    server.script = [(403, f"API key {KEY} not valid", 0)]
    with pytest.raises(LLMUnavailable) as e:
        server.client().generate_json("s", "u", None)
    assert "API key" in str(e.value) and KEY not in str(e.value) and ".env" in str(e.value)


def test_unknown_model_tells_how_to_fix_it(server):
    server.script = [(404, "model not found", 0)]
    with pytest.raises(LLMUnavailable) as e:
        server.client(model="old-model").generate_json("s", "u", None)
    assert "old-model" in str(e.value) and "GEMINI_MODEL" in str(e.value)


def test_busy_server_is_retried_by_the_sdk_then_reported(server):
    server.script = [(429, "slow down", 0)]                           # one hiccup: the SDK retries, we never notice
    assert server.client().generate_json("s", "u", None) == GOOD and len(server.requests) == 2
    server.requests.clear()
    server.script = [(429, "slow down", 0)] * 8                       # keeps failing: friendly message
    with pytest.raises(LLMUnavailable, match="limit or quota"):
        server.client().generate_json("s", "u", None)
    assert len(server.requests) >= 2


def test_server_errors(server):
    server.script = [(503, "overloaded", 0)] * 8
    with pytest.raises(LLMUnavailable, match="temporarily unavailable"):
        server.client().generate_json("s", "u", None)


def test_no_connection_and_timeout(server):
    c = GeminiClient(api_key=KEY, base_url="http://127.0.0.1:9")      # nothing listens there
    with pytest.raises(LLMUnavailable, match="Could not reach Gemini"):
        c.generate_json("s", "u", None)
    server.script = [(200, GOOD, 3)]
    slow = server.client(timeout_s=0.5)
    t = time.perf_counter()
    with pytest.raises(LLMUnavailable, match="Gemini did not respond within 0.5 seconds"):
        slow.generate_json("s", "u", None)
    assert time.perf_counter() - t < 2.5                                # the timeout really applies


def test_empty_answer_is_an_empty_string_not_a_crash(server):
    server.script = [(200, None, 0)]
    assert server.client().generate_json("s", "u", None) == ""


def test_missing_key_explains_what_to_do():
    with pytest.raises(LLMUnavailable) as e:
        GeminiClient()
    msg = str(e.value)
    assert "GEMINI_API_KEY" in msg and ".env" in msg and "Never paste" in msg


def test_whole_chain_with_the_real_sdk(server):
    """sentence -> (real SDK -> fake Google) -> spec -> validator -> request -> mining engine -> itemsets"""
    from src.llm.parser import parse_question
    from src.mining.engine import run_mining
    from tests.test_validation import CAT, DF

    server.script = [(200, '{"intent":"mining_request","spec":{"kind":"frequent","min_support":0.3,'
                           '"filters":{"weekdays":["Sunday"]}}}', 0)]
    parsed = parse_question("what do people buy together on Sundays?", server.client(), catalog=CAT, df=DF, log_path=None)
    assert parsed.status == "ok" and parsed.attempts == 1
    sent = server.requests[0]["body"]
    assert sent["input"].endswith("\n\nQuestion: what do people buy together on Sundays?")
    assert "No orders on Wednesday, Thursday, Saturday" in sent["input"] and "system_instruction" not in sent

    result = run_mining(parsed.validation.request, df=DF)
    assert result.n_baskets_selected == 30                              # the 30 Sunday baskets
    top = result.itemsets.iloc[0]
    assert top["itemset"] == ("PARTY BUNTING", "REGENCY CAKESTAND 3 TIER") and top["count"] == 30


# ---------------------------------------------------------------- plain-text answers (explanation layer)
def test_plain_text_request_has_no_json_format(server):
    server.script = [(200, "A short plain answer.", 0)]
    assert server.client().generate_text("SYSTEM", "FACTS") == "A short plain answer."
    body = server.requests[0]["body"]
    assert "response_format" not in body                                  # free text, not JSON mode
    assert body["input"] == "SYSTEM\n\nFACTS" and "system_instruction" not in body and body["store"] is False
    assert body["generation_config"]["temperature"] == 0


def test_plain_text_falls_back_when_the_model_rejects_settings(server):
    server.script = [(400, "thinking_level is not supported", 0), (200, "ok text", 0)]
    c = server.client()
    assert c.generate_text("s", "u") == "ok text"
    assert "generation_config" in server.requests[0]["body"] and "generation_config" not in server.requests[1]["body"]
    c.generate_text("s", "u")                                             # remembered: no repeated 400
    assert len(server.requests) == 3


def test_plain_text_errors_are_friendly(server):
    server.script = [(403, f"key {KEY} denied", 0)]
    with pytest.raises(LLMUnavailable) as e:
        server.client().generate_text("s", "u")
    assert "API key" in str(e.value) and KEY not in str(e.value)


def test_question_to_verified_answer_through_the_real_sdk(server):
    """sentence -> Gemini(JSON) -> spec -> validator -> engine -> Gemini(text) -> number check -> answer"""
    from src.llm.explainer import explain_result
    from src.llm.parser import parse_question
    from src.mining.engine import run_mining
    from tests.test_validation import BUNT, CAT, DF, REG

    answer = (f"The pair bought together most often is {BUNT} and {REG}, in about 46% of orders (30 orders). "
              "Consider offering them as a bundle.")
    server.script = [(200, '{"intent":"mining_request","spec":{"kind":"frequent","min_support":0.3}}', 0), (200, answer, 0)]
    client = server.client()
    parsed = parse_question("what is bought together?", client, catalog=CAT, df=DF, log_path=None)
    result = run_mining(parsed.validation.request, df=DF)
    explanation = explain_result(result, client, question="what is bought together?", log_path=None)

    assert explanation.source == "llm" and explanation.text == answer and explanation.attempts == 1
    json_call, text_call = (r["body"] for r in server.requests)
    assert "response_format" in json_call and "response_format" not in text_call
    assert text_call["input"].startswith("You write a short answer for a store manager")
    assert "\n\nQuestion: what is bought together?\n\nFACTS:" in text_call["input"]
    assert "46.15" in text_call["input"]                                 # the verified numbers the model may use


def test_a_two_turn_conversation_through_the_real_sdk(server):
    """turn 1 uses the first-question prompt; turn 2 sends the follow-up prompt WITH the current request."""
    from src.llm.conversation import converse
    from src.llm.session import Session
    from tests.test_validation import CAT, DF

    server.script = [
        (200, '{"intent":"mining_request","spec":{"kind":"frequent","min_support":0.3}}', 0),
        (200, '{"intent":"mining_request","spec":{"filters":{"weekdays":["Sunday"]}}}', 0),
    ]
    client, session = server.client(), Session()
    t1 = converse(session, "what is bought together?", client, catalog=CAT, df=DF, log_path=None)
    t2 = converse(session, "only on Sundays", client, catalog=CAT, df=DF, log_path=None)

    assert (t1.status, t1.turn_type, t2.status, t2.turn_type) == ("ok", "new", "ok", "update")
    first, second = (r["body"] for r in server.requests)
    assert "follow-up handler" not in first["input"] and first["input"].endswith("Question: what is bought together?")
    assert "follow-up handler" in second["input"]
    assert second["input"].endswith('CURRENT REQUEST: {"kind":"frequent","min_support":0.3}\nPENDING QUESTIONS: none\n'
                               "Message: only on Sundays")
    assert "clear" in second["response_format"]["schema"]["properties"]
    assert t2.validation.request.min_support == 0.3 and t2.validation.request.filters.weekdays == ["Sunday"]
