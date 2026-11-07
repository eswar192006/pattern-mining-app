"""Shared test setup."""
import pytest

from src.core import config as C


@pytest.fixture(autouse=True)
def isolated_llm_log(tmp_path, monkeypatch):
    """No test may ever write to the real logs/llm_requests.jsonl (Step 13's evaluation reads it)."""
    monkeypatch.setattr(C, "LLM_LOG_FILE", tmp_path / "llm_requests.jsonl")
