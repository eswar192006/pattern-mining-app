"""Providers for the shared language-model interface.

LLMClient is a tiny interface: generate_json(system, user, schema) -> text.
  * OllamaClient  - a local model through Ollama's HTTP API.
  * GeminiClient  - optional Google Gemini compatibility.
  * FakeLLM       - scripted answers for tests (src/llm/fake.py), no network, no key.

Provider-specific behavior stays here; parsing, mining, and UI use the shared interface.

Safety and privacy:
  * the key is read from the environment / .env and never logged or put in messages
  * store=False asks Google not to keep the conversation server-side
  * every failure becomes LLMUnavailable with a message that is safe to show a user
"""
import os
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.core import config as C
from src.core.errors import LLMUnavailable

# Settings tried in order. Newer models accept all of them, but if a model rejects one
# (HTTP 400) we fall back to a simpler request instead of failing.
#   0: constrained JSON schema + deterministic, low-effort generation
#   1: constrained JSON schema only
#   2: JSON mode without a schema (the prompt describes the format; pydantic validates)
_PROFILES = [
    {"schema": True, "gen": {"temperature": 0, "thinking_level": "low", "max_output_tokens": 2048}},
    {"schema": True, "gen": None},
    {"schema": False, "gen": None},
]


class OllamaClient:
    """Local Ollama client implementing generate_json/generate_text."""

    def __init__(self, model=None, base_url=None, timeout_s=None):
        try:
            from dotenv import load_dotenv
            load_dotenv(C.ROOT / ".env")
        except ImportError:
            pass
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.timeout_s = timeout_s or float(os.getenv("LLM_TIMEOUT_S", C.LLM_TIMEOUT_S))
        self._check_model()

    def _check_model(self):
        req = Request(self.base_url + "/api/tags")
        try:
            with urlopen(req, timeout=min(self.timeout_s, 4)) as response:
                models = json.loads(response.read().decode("utf-8")).get("models", [])
            names = {m.get("name") for m in models}
            if self.model not in names and f"{self.model}:latest" not in names:
                raise LLMUnavailable(f"Ollama is running, but model '{self.model}' is not installed. Run: ollama pull {self.model}")
        except LLMUnavailable:
            raise
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, TypeError, AttributeError,
                json.JSONDecodeError, UnicodeDecodeError):
            raise LLMUnavailable(f"Could not reach Ollama at {self.base_url}. Start the Ollama app and run: ollama pull {self.model}") from None

    def generate_json(self, system: str, user: str, schema: dict | None = None) -> str:
        return self._call(system, user, schema or "json")

    def generate_text(self, system: str, user: str) -> str:
        return self._call(system, user)

    def _call(self, system, user, fmt=None):
        body = {"model": self.model,
                "messages": ([{"role": "system", "content": system}] if system else [])
                            + [{"role": "user", "content": user}],
                "stream": False, "think": False,
                "options": {"temperature": 0}}
        if fmt is not None:
            body["format"] = fmt
        req = Request(self.base_url + "/api/chat", data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(req, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return (payload.get("message") or {}).get("content", "")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:240]
            raise LLMUnavailable(f"Ollama request failed ({exc.code}): {detail}. Check that '{self.model}' is installed.") from None
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            if isinstance(exc, TimeoutError) or getattr(exc, "reason", None).__class__ is TimeoutError:
                msg = f"Ollama did not respond within {self.timeout_s:g} seconds."
            else:
                msg = f"Could not reach Ollama at {self.base_url}. Start the Ollama app and make sure '{self.model}' is installed."
            raise LLMUnavailable(msg) from None
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
            raise LLMUnavailable("Ollama returned an invalid response. Check the Ollama service and model.") from None


def create_llm_client(provider=None):
    """Create the configured live provider; Ollama is the local default."""
    try:
        from dotenv import load_dotenv
        load_dotenv(C.ROOT / ".env")
    except ImportError:
        pass
    provider = (provider or os.getenv("LLM_PROVIDER", "ollama")).strip().lower()
    if provider == "ollama":
        return OllamaClient()
    if provider == "gemini":
        return GeminiClient()
    raise LLMUnavailable(f"Unknown LLM_PROVIDER '{provider}'. Choose 'ollama' or 'gemini'.")


def _status(e):
    """HTTP status of an SDK error. The Interactions API uses .status_code, the older API .code."""
    return getattr(e, "status_code", None) or getattr(e, "code", None)


class GeminiClient:
    def __init__(self, api_key=None, model=None, timeout_s=None, base_url=None, client=None):
        try:
            from dotenv import load_dotenv
            load_dotenv(C.ROOT / ".env")
        except ImportError:                       # python-dotenv is optional
            pass
        self.model = model or os.getenv("GEMINI_MODEL") or C.GEMINI_MODEL_DEFAULT
        self._key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
        self._timeout_s = timeout_s or float(os.getenv("LLM_TIMEOUT_S", C.LLM_TIMEOUT_S))
        self._profile = 0                         # remembers the first profile the server accepted (JSON)
        self._text_profile = 0                    # same, for plain-text answers

        if client is None:
            if not self._key:
                raise LLMUnavailable(
                    "No Gemini API key found. Copy .env.example to .env and put your key in it as "
                    "GEMINI_API_KEY=... (get a key at https://aistudio.google.com/apikey). "
                    "Never paste the key into a chat or commit it to git.")
            from google import genai
            from google.genai import types
            options = {"timeout": int(self._timeout_s * 1000)}
            if base_url:
                options["base_url"] = base_url
            client = genai.Client(api_key=self._key, http_options=types.HttpOptions(**options))
        self._client = client

    # ---------------------------------------------------------------- public
    def generate_json(self, system: str, user: str, schema: dict | None = None) -> str:
        """Ask the model for a JSON answer; returns the raw text ('' if it sent nothing)."""
        last = None
        i = self._profile
        while i < len(_PROFILES):
            profile = _PROFILES[i]
            try:
                text = self._call(system, user, schema if profile["schema"] else None, profile["gen"])
                self._profile = i
                return text
            except Exception as e:                # mapped below
                if _status(e) == 400 and i + 1 < len(_PROFILES):
                    last, i = e, i + 1            # this model rejected a setting: try a simpler request
                    continue
                raise self._friendly(e) from None
        raise self._friendly(last) from None

    def generate_text(self, system: str, user: str) -> str:
        """Ask the model for a plain-text answer (used by the explanation layer)."""
        gens = [_PROFILES[0]["gen"], None]
        i = self._text_profile
        while i < len(gens):
            try:
                text = self._call(system, user, None, gens[i], as_json=False)
                self._text_profile = i
                return text
            except Exception as e:
                if _status(e) == 400 and i + 1 < len(gens):
                    i += 1                        # this model rejected the generation settings: simpler request
                    continue
                raise self._friendly(e) from None

    # --------------------------------------------------------------- private
    def _call(self, system, user, schema, gen, as_json=True):
        # The Interactions API did not follow `system_instruction` reliably in live tests (filters and
        # thresholds in the prompt were ignored; the model only used the schema). So the instructions
        # travel at the start of the input instead, which was verified to work.
        combined = f"{system}\n\n{user}" if system else user
        kwargs = dict(model=self.model, input=combined, store=False)
        if as_json:
            fmt = {"type": "text", "mime_type": "application/json"}
            if schema is not None:
                fmt["schema"] = schema
            kwargs["response_format"] = fmt
        if gen:
            kwargs["generation_config"] = gen
        # (the SDK itself retries 429 / 5xx a few times before raising)
        interaction = self._client.interactions.create(**kwargs)
        return getattr(interaction, "output_text", None) or ""

    def _scrub(self, text: str) -> str:
        return text.replace(self._key, "***") if self._key else text

    def _friendly(self, e) -> LLMUnavailable:
        if isinstance(e, LLMUnavailable):
            return e
        code = _status(e)
        detail = self._scrub(str(getattr(e, "message", "") or e))[:200]
        if code in (401, 403):
            msg = ("Gemini rejected the API key. Check GEMINI_API_KEY in your .env file "
                   "(the key may be wrong, revoked or not allowed for this model).")
        elif code == 404:
            msg = (f"Gemini does not know the model '{self.model}'. Model names change often: pick a current one "
                   "from https://ai.google.dev/gemini-api/docs/models and set GEMINI_MODEL in your .env file.")
        elif code == 429:
            msg = "Gemini says the request limit or quota was reached. Wait a minute and try again."
        elif code == 504:
            msg = ("Gemini did not finish before its server deadline. Retry the request or increase "
                   "LLM_TIMEOUT_S in your .env file.")
        elif type(e).__name__ == "APITimeoutError":
            msg = (f"Gemini did not respond within {self._timeout_s:g} seconds. Increase LLM_TIMEOUT_S "
                   "in your .env file or try again later.")
        elif code and code >= 500:
            msg = "The Gemini service is temporarily unavailable. Please try again in a moment."
        elif code == 400:
            msg = f"Gemini rejected the request: {detail}"
        else:
            msg = ("Could not reach Gemini (network problem or timeout). "
                   "Check your internet connection and try again.")
        return LLMUnavailable(msg)
