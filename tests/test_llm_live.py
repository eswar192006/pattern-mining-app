"""OPTIONAL: calls the REAL Gemini. Skipped unless you switch it on:

    RUN_LIVE_LLM=1 python -m pytest tests/test_llm_live.py -q -s

Needs GEMINI_API_KEY in .env and the cleaned data. Uses ~7 requests (paced to respect free-tier limits).
A model can occasionally phrase things differently, so treat a single failure as a prompt to look
(python -m src.demo_parse "<that question>"), not necessarily as a bug."""
import os
import time

import pytest

from src.core import config as C
from src.core.catalog import get_catalog
from src.llm.client import GeminiClient
from src.llm.parser import parse_question

pytestmark = [
    pytest.mark.skipif(os.getenv("RUN_LIVE_LLM") != "1", reason="set RUN_LIVE_LLM=1 to call the real Gemini"),
    pytest.mark.skipif(not C.CLEAN_FILE.exists(), reason="run prepare_data first"),
]

CASES = [
    ("Which products are usually bought together?", "ok", lambda v: v.request.kind == "frequent"),
    ("Which product combinations earn the most money on Sundays?", "ok",
     lambda v: v.request.kind == "high_utility" and v.request.filters.weekdays == ["Sunday"]),
    ("Show me rare but strongly linked pairs of products in Germany", "ok",
     lambda v: v.request.kind == "rare" and v.request.filters.countries == ["Germany"]),
    ("What sells together with the regency cakestand?", "ok",
     lambda v: v.request.filters.include_items == ["REGENCY CAKESTAND 3 TIER"]),
    ("Strongest bundles of three products on weekdays, leave out the bunting", "ok",
     lambda v: v.request.max_len == 3 and v.request.filters.weekend is False and len(v.request.filters.exclude_items) == 18),
    ("Tell me something interesting about my sales", "needs_clarification",
     lambda v: v.questions[0].field == "kind"),
    ("What's the weather in Paris today?", "out_of_scope", None),
]


@pytest.fixture(scope="module")
def llm():
    return GeminiClient()


@pytest.mark.parametrize("question,status,check", CASES)
def test_real_gemini_translates_correctly(llm, tmp_path, question, status, check):
    time.sleep(float(os.getenv("LIVE_PACING_S", "4")))
    r = parse_question(question, llm, catalog=get_catalog(), log_path=tmp_path / "live.jsonl")
    assert r.status == status, (r.status, r.message, r.problems, r.raw_replies)
    if check:
        assert check(r.validation), r.validation.to_dict()
