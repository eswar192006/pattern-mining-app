"""A scripted stand-in for the language model: no network, no key, fully deterministic.
Used by the tests (and handy for offline demos)."""


class FakeLLM:
    model = "fake"

    def __init__(self, replies):
        """replies: list of items, used one per call. An item is a str (returned),
        an Exception (raised), or a callable(system, user) -> str."""
        self._replies = list(replies)
        self.calls = []

    def generate_json(self, system: str, user: str, schema=None) -> str:
        return self._next("json", system, user, schema)

    def generate_text(self, system: str, user: str) -> str:
        return self._next("text", system, user, None)

    def _next(self, mode, system, user, schema):
        self.calls.append({"mode": mode, "system": system, "user": user, "schema": schema})
        if not self._replies:
            raise AssertionError("FakeLLM was called more times than it has replies")
        item = self._replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item(system, user) if callable(item) else item
