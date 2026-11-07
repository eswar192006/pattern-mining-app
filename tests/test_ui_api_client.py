"""The interface's API client (ui/api_client.py): friendly failures, and the real network code.

Three layers: a fake transport (no network), a real local HTTP server (the standard-library code path,
including a closed port), and, when FastAPI is installed, the real API with a scripted model.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from tests.ui_samples import HEALTH_OK, reply
from ui.api_client import START_COMMAND, ApiClient, ApiProblem


class Fake:
    """A transport that records requests and plays back scripted (status, headers, body) answers."""

    def __init__(self, *answers):
        self.answers, self.calls = list(answers), []

    def __call__(self, method, url, payload, timeout):
        self.calls.append((method, url, payload, timeout))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def client(*answers):
    transport = Fake(*answers)
    return ApiClient("http://api.test/", transport), transport


def problem_of(call):
    try:
        call()
    except ApiProblem as e:
        return e
    raise AssertionError("expected an ApiProblem")


# ------------------------------------------------------------------ requests it sends
def test_chat_sends_the_documented_request():
    api, t = client((200, {}, reply()))
    assert api.chat("hello", None, explain=False)["status"] == "ok"
    method, url, payload, timeout = t.calls[0]
    assert (method, url) == ("POST", "http://api.test/chat") and payload == {"message": "hello", "explain": False}
    api, t = client((200, {}, reply()))
    api.chat("again", "sid-9")
    assert t.calls[0][2] == {"message": "again", "explain": True, "session_id": "sid-9"}


def test_the_other_endpoints_use_the_right_paths():
    api, t = client((200, {}, reply()), (200, {}, reply()), (200, {}, reply()), (200, {}, HEALTH_OK), (200, {}, {"deleted": True}))
    api.undo("s1", explain=False)
    api.run("s1")
    api.reset("s1")
    api.health()
    api.delete("s1")
    assert [(c[0], c[1].replace("http://api.test", "")) for c in t.calls] == [
        ("POST", "/sessions/s1/undo?explain=false"), ("POST", "/sessions/s1/run?explain=true"),
        ("POST", "/sessions/s1/reset"), ("GET", "/health"), ("DELETE", "/sessions/s1")]


def test_slow_calls_wait_longer_than_quick_ones():
    api, t = client((200, {}, HEALTH_OK), (200, {}, reply()))
    api.health()
    api.chat("x")
    assert t.calls[0][3] < t.calls[1][3] and t.calls[1][3] >= 120


# ------------------------------------------------------------------ replies that are not failures
def test_a_503_with_a_normal_body_is_returned_not_raised():
    body = reply(status="llm_unavailable", message="Gemini is busy")
    api, _ = client((503, {}, body))
    assert api.chat("x")["status"] == "llm_unavailable"


def test_probe_never_raises():
    api, _ = client((200, {}, HEALTH_OK))
    health, problem = api.probe()
    assert health["status"] == "ok" and problem is None
    api, _ = client(ConnectionRefusedError())
    health, problem = api.probe()
    assert health is None and problem.kind == "server_down"


# ------------------------------------------------------------------ friendly failures
def test_server_not_running_says_how_to_start_it():
    e = problem_of(lambda: client(ConnectionRefusedError())[0].chat("x"))
    assert e.kind == "server_down" and START_COMMAND in e.detail and "http://api.test" in e.detail


def test_timeouts_are_named_as_such():
    e = problem_of(lambda: client(TimeoutError())[0].chat("x"))
    assert e.kind == "timeout" and "seconds" in e.detail


def test_expired_conversation_asks_to_start_again():
    body = {"error": {"code": "session_not_found", "message": "gone"}}
    e = problem_of(lambda: client((404, {}, body))[0].chat("x", "old"))
    assert e.kind == "session_expired" and "expired" in e.headline


def test_rate_limit_keeps_the_wait_time():
    body = {"error": {"code": "rate_limited", "message": "Too many requests. Please wait 12 seconds."}}
    e = problem_of(lambda: client((429, {"Retry-After": "13"}, body))[0].chat("x"))
    assert e.kind == "rate_limited" and e.retry_after == 13 and "12 seconds" in e.detail


def test_not_ready_shows_the_servers_reason():
    body = {"error": {"code": "service_unavailable", "message": "No Gemini API key found."}}
    e = problem_of(lambda: client((503, {}, body))[0].chat("x"))
    assert e.kind == "not_ready" and "No Gemini API key found." in e.detail


def test_bad_input_and_unexpected_errors():
    e = problem_of(lambda: client((422, {}, {"error": {"code": "invalid_input", "message": "message: too long"}}))[0].chat("x"))
    assert e.kind == "bad_input" and "too long" in e.detail
    e = problem_of(lambda: client((500, {}, {"error": {"code": "internal_error", "message": "Something went wrong."}}))[0].chat("x"))
    assert e.kind == "server_error"
    e = problem_of(lambda: client((502, {}, None))[0].chat("x"))                         # a proxy page, not our JSON
    assert e.kind == "server_error" and "502" in e.detail


def test_deleting_a_missing_conversation_is_silent():
    api, _ = client((404, {}, {"error": {"code": "session_not_found", "message": "gone"}}))
    assert api.delete("old") is None


# ------------------------------------------------------------------ the real network code, against a local server
class _Handler(BaseHTTPRequestHandler):
    def _send(self, status, body, headers=()):
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        for k, v in headers:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        self._send(200, HEALTH_OK)

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if payload.get("message") == "limit":
            return self._send(429, {"error": {"code": "rate_limited", "message": "slow down"}}, [("Retry-After", "5")])
        self._send(200, reply(message=payload.get("message")))

    def log_message(self, *args):
        pass


def _serve():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_real_http_round_trip_and_error_statuses():
    server = _serve()
    try:
        api = ApiClient(f"http://127.0.0.1:{server.server_port}")
        assert api.health()["llm"]["model"] == "gemini-3.6-flash"
        assert api.chat("hi there")["message"] == "hi there"
        e = problem_of(lambda: api.chat("limit"))
        assert e.kind == "rate_limited" and e.retry_after == 5
    finally:
        server.shutdown()


def test_real_closed_port_is_reported_as_server_down():
    server = _serve()
    port = server.server_port
    server.shutdown()
    server.server_close()
    e = problem_of(lambda: ApiClient(f"http://127.0.0.1:{port}").health())
    assert e.kind == "server_down"


# ------------------------------------------------------------------ the real API with a scripted model
def test_through_the_real_api_with_a_fake_model():
    pytest.importorskip("fastapi")
    pytest.importorskip("pydantic")
    from fastapi.testclient import TestClient

    from tests.test_api import make
    from tests.test_pipeline import ANSWER, FREQ

    tc, _, _ = make([FREQ, ANSWER])

    def transport(method, url, payload, timeout):
        r = tc.request(method, url.replace("http://api.test", ""), json=payload)
        return r.status_code, dict(r.headers), r.json()

    api = ApiClient("http://api.test", transport)
    assert api.health()["status"] == "ok" and api.catalog()["baskets"] == 65
    first = api.chat("what sells together?")
    assert first["status"] == "ok" and first["answer"]["source"] == "llm" and first["result"]["itemsets"]
    assert api.reset(first["session_id"])["status"] == "reset"
    gone = problem_of(lambda: api.chat("again", "no-such-session"))
    assert gone.kind == "session_expired"
