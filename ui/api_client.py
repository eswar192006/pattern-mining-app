"""The interface's connection to the API server (api/main.py). No Streamlit in this file.

    api = ApiClient("http://127.0.0.1:8000")
    reply = api.chat("which products sell together on Sundays?", session_id=None)

Every failure becomes an ApiProblem with a plain-words headline and detail that are safe to show to a
user: the server is not running, the conversation expired, too many requests, the server is not ready
(no key, data not prepared), and so on. Only the standard library is used for the network, and the
transport can be replaced, so the client is tested without a network (see tests/test_ui_api_client.py).

A normal answer from the API is a dict in the shape of src/core/pipeline.py. Domain outcomes such as a
question to answer or an off-topic message are normal replies (status 200), and so is "the model is
unavailable" (status 503 with a normal body), so those are returned, not raised.
"""
import json
import os
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8000"
START_COMMAND = "uvicorn api.main:create_app --factory --port 8000"
QUICK_S, SLOW_S = 10.0, 240.0          # health and catalog are quick; a chat turn may wait for the model and Java


class ApiProblem(Exception):
    """A failure the user can understand. `kind` lets the page choose a screen or a button."""

    def __init__(self, kind: str, headline: str, detail: str = "", retry_after: float | None = None):
        super().__init__(f"{headline}: {detail}" if detail else headline)
        self.kind, self.headline, self.detail, self.retry_after = kind, headline, detail, retry_after


def default_url() -> str:
    return os.getenv("API_URL", DEFAULT_URL).rstrip("/")


def urllib_transport(method: str, url: str, payload, timeout: float):
    """-> (status, headers, parsed JSON or None). HTTP error statuses are returned, not raised."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status, headers, raw = resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:                    # 4xx / 5xx still carry a JSON body from our API
        status, headers, raw = e.code, dict(e.headers or {}), e.read()
    try:
        return status, headers, (json.loads(raw.decode("utf-8")) if raw else None)
    except ValueError:
        return status, headers, None


class ApiClient:
    def __init__(self, base_url: str | None = None, transport=None):
        self.base_url = (base_url or default_url()).rstrip("/")
        self._transport = transport or urllib_transport

    # ------------------------------------------------------------------ the endpoints
    def health(self) -> dict:
        return self._request("GET", "/health", timeout=QUICK_S)

    def catalog(self) -> dict:
        return self._request("GET", "/catalog", timeout=QUICK_S)

    def chat(self, message: str, session_id: str | None = None, explain: bool = True) -> dict:
        payload = {"message": message, "explain": bool(explain)}
        if session_id:
            payload["session_id"] = session_id
        return self._request("POST", "/chat", payload, timeout=SLOW_S)

    def undo(self, session_id: str, explain: bool = True) -> dict:
        return self._request("POST", f"/sessions/{session_id}/undo?explain={str(bool(explain)).lower()}", timeout=SLOW_S)

    def reset(self, session_id: str) -> dict:
        return self._request("POST", f"/sessions/{session_id}/reset", timeout=QUICK_S)

    def run(self, session_id: str, explain: bool = True) -> dict:
        return self._request("POST", f"/sessions/{session_id}/run?explain={str(bool(explain)).lower()}", timeout=SLOW_S)

    def delete(self, session_id: str) -> None:
        try:
            self._request("DELETE", f"/sessions/{session_id}", timeout=QUICK_S)
        except ApiProblem:
            pass                                          # forgetting a conversation that is already gone is fine

    def probe(self) -> tuple[dict | None, ApiProblem | None]:
        """Health check that never raises: (health, None) or (None, problem). For the page's first screen."""
        try:
            return self.health(), None
        except ApiProblem as e:
            return None, e

    # ------------------------------------------------------------------ one place for every failure
    def _request(self, method: str, path: str, payload=None, timeout: float = QUICK_S) -> dict:
        try:
            status, headers, body = self._transport(method, self.base_url + path, payload, timeout)
        except TimeoutError:
            raise self._timeout(timeout) from None
        except urllib.error.URLError as e:
            if isinstance(e.reason, TimeoutError):
                raise self._timeout(timeout) from None
            raise self._down() from None
        except OSError:                                    # connection refused / reset
            raise self._down() from None

        if isinstance(body, dict) and status in (200, 201):
            return body
        if isinstance(body, dict) and status == 503 and "status" in body:
            return body                                    # "the model is unavailable" is a normal reply
        raise self._problem(status, headers, body)

    def _down(self) -> ApiProblem:
        return ApiProblem("server_down", "The pattern-mining server is not running",
                          f"Start it in a terminal:  {START_COMMAND}   (looking for it at {self.base_url})")

    @staticmethod
    def _timeout(timeout: float) -> ApiProblem:
        return ApiProblem("timeout", "The server took too long to answer",
                          f"No answer within {timeout:g} seconds. Try again; heavy money-based questions can be slow.")

    @staticmethod
    def _problem(status: int, headers: dict, body) -> ApiProblem:
        error = (body or {}).get("error") if isinstance(body, dict) else None
        code = (error or {}).get("code", "")
        message = (error or {}).get("message", "")
        if code == "session_not_found":
            return ApiProblem("session_expired", "This conversation has expired",
                              "Conversations are forgotten after two hours of silence or when the server restarts. "
                              "Ask your question again to start a new one.")
        if code == "rate_limited" or status == 429:
            wait = None
            for key in ("Retry-After", "retry-after"):
                if key in (headers or {}):
                    try:
                        wait = float(headers[key])
                    except ValueError:
                        pass
            return ApiProblem("rate_limited", "Too many requests", message or "Please wait a moment and try again.", wait)
        if code == "service_unavailable" or status == 503:
            return ApiProblem("not_ready", "The server is not ready", message or "It is still starting up or missing something.")
        if code == "invalid_input" or status == 422:
            return ApiProblem("bad_input", "That message could not be used", message)
        return ApiProblem("server_error", "Something went wrong on the server",
                          message or f"The server answered with status {status}. Please try again.")
