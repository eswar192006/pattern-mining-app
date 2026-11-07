"""Errors the app can explain to a user in plain words.

All carry a human-readable message. They also subclass the built-in error that
the miners used before (ValueError / RuntimeError), so older code keeps working.
"""


class MiningError(Exception):
    """Base class: anything the engine can turn into a friendly message."""


class InvalidRequest(MiningError, ValueError):
    """The request itself is wrong or contradictory (fixable by the user)."""


class LimitExceeded(MiningError, ValueError):
    """The request is valid but too heavy: below a safety floor, too many results,
    or it ran too long. The message says which setting to relax."""


class EngineUnavailable(MiningError, RuntimeError):
    """A needed component is missing or failed (for example Java or spmf.jar)."""


class LLMUnavailable(MiningError, RuntimeError):
    """The language model could not be used (no key, rejected key, no network, quota, timeout...).
    The message is safe to show to a user: it never contains the API key."""
