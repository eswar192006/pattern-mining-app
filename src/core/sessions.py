"""Server-side conversation storage and rate limiting for the API. In memory, thread-safe, and bounded:
an idle conversation expires, and the oldest is dropped when there are too many, so memory cannot grow
without limit. Session ids are unguessable random strings (they are the only credential)."""
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from src.core import config as C
from src.llm.session import Session


@dataclass
class Entry:
    session: Session
    lock: threading.Lock = field(default_factory=threading.Lock)     # one request at a time per conversation
    last_used: float = 0.0


class SessionStore:
    def __init__(self, max_sessions: int = C.MAX_SESSIONS, ttl_s: float = C.SESSION_TTL_S, clock=time.monotonic):
        self._max, self._ttl, self._clock = max_sessions, ttl_s, clock
        self._items: dict[str, Entry] = {}
        self._guard = threading.RLock()

    def _expire(self) -> None:
        now = self._clock()
        for sid in [k for k, e in self._items.items() if now - e.last_used > self._ttl]:
            del self._items[sid]

    def create(self) -> str:
        with self._guard:
            self._expire()
            while len(self._items) >= self._max:                      # drop the least recently used
                del self._items[min(self._items, key=lambda k: self._items[k].last_used)]
            sid = secrets.token_urlsafe(16)
            self._items[sid] = Entry(Session(), last_used=self._clock())
            return sid

    def get(self, sid: str) -> Entry | None:
        with self._guard:
            self._expire()
            entry = self._items.get(sid)
            if entry is not None:
                entry.last_used = self._clock()
            return entry

    def delete(self, sid: str) -> bool:
        with self._guard:
            return self._items.pop(sid, None) is not None

    def __len__(self) -> int:
        with self._guard:
            self._expire()
            return len(self._items)


class RateLimiter:
    """At most `limit` requests per `window_s` per key (sliding window)."""

    def __init__(self, limit: int = C.RATE_LIMIT_PER_MIN, window_s: float = 60.0, clock=time.monotonic):
        self._limit, self._window, self._clock = limit, window_s, clock
        self._hits: dict[str, deque] = {}
        self._guard = threading.Lock()

    def check(self, key: str) -> float | None:
        """Record a request. Returns None if allowed, else the seconds to wait before trying again."""
        now = self._clock()
        with self._guard:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= self._window:
                hits.popleft()
            if len(hits) >= self._limit:
                return round(self._window - (now - hits[0]), 1)
            hits.append(now)
            if len(self._hits) > 5000:                                 # forget idle clients
                for k in [k for k, h in self._hits.items() if not h or now - h[-1] >= self._window]:
                    self._hits.pop(k, None)
            return None
