"""The web API.   Start it:

    uvicorn api.main:create_app --factory --port 8000        (or: python -m api.main)
    then open  http://127.0.0.1:8000/docs  for an interactive page where you can try every endpoint.

Every response that is not a normal result has the same shape:  {"error": {"code": ..., "message": ...}}.
The API keeps working when something is missing (Ollama unavailable, data not prepared, no Java): it starts,
/health says what is wrong, and only the endpoints that need it answer 503 with the reason.
Provider credentials, when used, stay on the server and are never returned.
"""
import logging
import os
import shutil

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from src.core import config as C
from src.core.catalog import Catalog
from src.core.errors import LLMUnavailable
from src.core.pipeline import Pipeline
from src.core.presets import KIND_DESCRIPTIONS, PRESETS
from src.core.sessions import RateLimiter, SessionStore
from src.llm.client import create_llm_client
from src.llm.replay import ReplayLLM

log = logging.getLogger("api")
BUSY_CODES = ("server_busy", "engine_unavailable")


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, headers: dict | None = None):
        self.status, self.code, self.message, self.headers = status, code, message, headers or {}


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1, max_length=C.MAX_QUESTION_CHARS, description="What the user typed.")
    session_id: str | None = Field(None, max_length=64, description="Omit to start a new conversation.")
    explain: bool = Field(True, description="Also write the plain-English answer (one more model call).")


def _error(status: int, code: str, message: str, headers=None) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status, headers=headers)


def create_app(llm=None, df=None, store: SessionStore | None = None, limiter: RateLimiter | None = None) -> FastAPI:
    # ---- start up even when something is missing; remember why
    data_error = llm_error = None
    if df is None:
        try:
            from src.mining.engine import get_data
            df = get_data()
        except FileNotFoundError as e:
            data_error = str(e)
    if llm is None:
        try:
            from dotenv import load_dotenv
            load_dotenv(C.ROOT / ".env")
            replay_enabled = os.getenv("DEMO_REPLAY", "").strip().lower() in {"1", "true", "yes", "on"}
            llm = ReplayLLM() if replay_enabled else create_llm_client()
        except LLMUnavailable as e:
            llm_error = str(e)
    replay_mode = isinstance(llm, ReplayLLM)
    pipeline = Pipeline(llm, df) if (df is not None and llm is not None) else None
    store = store or SessionStore()
    limiter = limiter or RateLimiter()

    app = FastAPI(title="Hybrid LLM pattern mining: sales", version="1.0",
                  description="Ask questions about retail sales patterns in plain English. A language model "
                              "translates and explains; exact algorithms do all the counting.")
    origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()] or C.CORS_ORIGINS_DEFAULT
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST", "DELETE"],
                       allow_headers=["Content-Type"])

    # ---- one error shape everywhere
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, e: ApiError):
        return _error(e.status, e.code, e.message, e.headers)

    @app.exception_handler(RequestValidationError)
    async def _invalid(request: Request, e: RequestValidationError):
        first = e.errors()[0]
        where = ".".join(str(p) for p in first["loc"] if p != "body")
        return _error(422, "invalid_input", f"{where}: {first['msg']}" if where else first["msg"])

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, e: Exception):
        log.exception("Unexpected error on %s %s", request.method, request.url.path)
        return _error(500, "internal_error", "Something went wrong on the server. Please try again.")

    # ---- helpers
    def need_pipeline() -> Pipeline:
        if pipeline is None:
            raise ApiError(503, "service_unavailable", data_error or llm_error or "The service is not ready.")
        return pipeline

    def need_session(sid: str):
        entry = store.get(sid)
        if entry is None:
            raise ApiError(404, "session_not_found", "This conversation was not found or has expired. "
                                                     "Start a new one with POST /sessions.")
        return entry

    def throttle(request: Request) -> None:
        wait = limiter.check(request.client.host if request.client else "unknown")
        if wait is not None:
            raise ApiError(429, "rate_limited", f"Too many requests. Please wait {wait:g} seconds.",
                           {"Retry-After": str(int(wait) + 1)})

    def reply(body: dict, sid: str) -> JSONResponse:
        body = {"session_id": sid, **body}
        down = body["status"] == "llm_unavailable" or (body["error"] or {}).get("code") in BUSY_CODES
        return JSONResponse(body, status_code=503 if down else 200)

    # ---- endpoints
    @app.get("/health")
    def health():
        java, jar = shutil.which("java") is not None, C.SPMF_JAR.exists()
        ready = pipeline is not None
        return {"status": "ok" if ready else "degraded",
                "demo_mode": "replay" if replay_mode else "live",
                "data": {"ready": df is not None, "baskets": int(df["basket_id"].nunique()) if df is not None else None,
                         "error": data_error},
                "llm": {"ready": llm is not None, "model": getattr(llm, "model", None), "error": llm_error},
                "mining": {"java": java, "spmf_jar": jar, "high_utility_available": java and jar},
                "sessions": len(store)}

    @app.get("/catalog")
    def catalog():
        cat = need_pipeline().catalog
        return {"baskets": cat.n_baskets, "products": len(cat.item_names), "date_range": cat.date_range,
                "weekdays_with_orders": sorted(cat.weekdays_present), "countries": list(cat.countries.index[:15]),
                "example_products": cat.item_names[:20], "kinds": KIND_DESCRIPTIONS, "presets": PRESETS,
                "limits": {"min_support_floor": C.MIN_SUPPORT_FLOOR, "max_message_chars": C.MAX_QUESTION_CHARS,
                           "high_utility_floor_by_size": C.HUIM_MIN_UTILITY_PCT_FLOOR}}

    @app.post("/sessions", status_code=201)
    def new_session():
        return {"session_id": store.create()}

    @app.get("/sessions/{session_id}")
    def get_session(session_id: str):
        s = need_session(session_id).session
        return {"session_id": session_id, "current": s.current(), "pending_questions": s.questions,
                "can_undo": bool(s.history), "turns": s.turns}

    @app.delete("/sessions/{session_id}")
    def delete_session(session_id: str):
        need_session(session_id)
        store.delete(session_id)
        return {"deleted": True}

    @app.post("/chat")
    def chat(req: ChatRequest, request: Request):
        pl = need_pipeline()
        throttle(request)
        sid = req.session_id or store.create()
        entry = need_session(sid)
        with entry.lock:
            return reply(pl.chat(entry.session, req.message, req.explain), sid)

    @app.post("/sessions/{session_id}/undo")
    def undo(session_id: str, request: Request, explain: bool = True):
        pl = need_pipeline()
        entry = need_session(session_id)
        throttle(request)
        with entry.lock:
            return reply(pl.undo(entry.session, explain), session_id)

    @app.post("/sessions/{session_id}/reset")
    def reset(session_id: str):
        pl = need_pipeline()
        entry = need_session(session_id)
        with entry.lock:
            return reply(pl.reset(entry.session), session_id)

    @app.post("/sessions/{session_id}/run")
    def run(session_id: str, request: Request, explain: bool = True):
        pl = need_pipeline()
        entry = need_session(session_id)
        throttle(request)
        with entry.lock:
            return reply(pl.run_current(entry.session, explain), session_id)

    app.state.pipeline, app.state.store, app.state.limiter = pipeline, store, limiter
    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(), host=os.getenv("API_HOST", "127.0.0.1"), port=int(os.getenv("API_PORT", "8000")))
