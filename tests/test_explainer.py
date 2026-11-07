import json
import shutil

import pytest

from src.core import config as C
from src.core.errors import LLMUnavailable
from src.llm import numbercheck as nc
from src.llm.explainer import SYSTEM_PROMPT, build_facts, explain_result, render_template
from src.llm.fake import FakeLLM
from src.llm.parser import parse_question
from src.mining.engine import run_mining
from tests.test_validation import BUNT, CAT, DF, REG

needs_spmf = pytest.mark.skipif(shutil.which("java") is None or not C.SPMF_JAR.exists(), reason="needs Java and tools/spmf.jar")

FREQ = {"kind": "frequent", "min_support": 0.3}                       # (BUNT, REG): 30 of 65 baskets = 46.15%
RARE = {"kind": "rare", "max_support": 0.3, "min_count": 3, "min_strength": 0.2, "max_len": 3}
HU = {"kind": "high_utility", "min_utility": 50, "max_len": 3}

GOOD = (f"The pair bought together most often is {BUNT} and {REG}, in about 46% of orders (30 orders). "
        "Consider offering them as a bundle.")


def result(req):
    return run_mining(req, df=DF)


def explain(req, replies, question="what sells together?", **kw):
    llm = FakeLLM(replies)
    return explain_result(result(req), llm, question=question, log_path=kw.pop("log_path", None), **kw), llm


# ---------------------------------------------------------------- the fact sheet
def test_facts_are_compact_rounded_and_complete():
    f = build_facts(result({**FREQ, "filters": {"weekdays": ["Sunday"]}}), "what sells on Sundays?")
    d = f.data
    assert d["question"] == "what sells on Sundays?" and d["looking_for"].startswith("combinations of products")
    assert d["orders_analysed"] == 30 and d["combinations_found"] == d["combinations_shown"] == 1
    assert d["results"] == [{"products": [BUNT, REG], "orders": 30, "share_of_orders_pct": 100.0}]
    assert any("only orders placed on Sunday" in s for s in d["settings"])
    assert "2010" not in json.dumps(d)                                   # no dates or years for the model to repeat
    assert f.names == sorted([BUNT, REG])
    assert any(n.pct and abs(n.value - 100.0) < 1e-9 for n in f.allowed)


def test_facts_row_limit_and_size():
    r = result({"kind": "frequent", "min_support": 0.05})
    assert r.total_found > 3
    f = build_facts(r, max_rows=3)
    assert len(f.data["results"]) == 3 and f.data["combinations_found"] == r.total_found
    assert len(json.dumps(f.data)) < 4000


def test_facts_per_kind_fields():
    rare = build_facts(result(RARE)).data["results"][0]
    assert set(rare) == {"products", "orders", "share_of_orders_pct", "come_together_pct"}


@needs_spmf
def test_facts_high_utility_fields():
    f = build_facts(result(HU))
    row = f.data["results"][0]
    assert set(row) == {"products", "orders", "earned", "share_of_sales_pct"} and f.data["total_sales"] == 140
    assert any("revenue" in c for c in f.data["caveats"])                # revenue is not profit


# ---------------------------------------------------------------- the fixed template is always clean
@pytest.mark.parametrize("req", [FREQ, RARE, pytest.param(HU, marks=needs_spmf),
                                 {"kind": "frequent", "min_support": 0.9},                    # empty: hint
                                 {"kind": "frequent", "filters": {"weekdays": ["Saturday"]}},  # empty: no baskets
                                 {"kind": "frequent", "min_support": 0.3, "filters": {"weekend": True}}])   # warning
def test_template_passes_its_own_number_check(req):
    f = build_facts(result(req))
    text = render_template(f)
    assert text and nc.check_text(text, f.allowed, f.names) == [], text


def test_template_content():
    t = render_template(build_facts(result(FREQ)))
    assert f"{BUNT} and {REG}" in t and "46.15%" in t and "30 orders" in t and "bundle" in t
    assert "9.0%" not in t
    t = render_template(build_facts(result({"kind": "frequent", "min_support": 0.9})))
    assert "46.15%" in t and f"{BUNT} + {REG}" in t


# ---------------------------------------------------------------- the model's answer
def test_a_clean_answer_is_used():
    e, llm = explain(FREQ, [GOOD])
    assert e.source == "llm" and e.text == GOOD and e.attempts == 1 and e.problems == [] and e.reason is None
    assert llm.calls[0]["mode"] == "text" and llm.calls[0]["system"] == SYSTEM_PROMPT
    user = llm.calls[0]["user"]
    assert user.startswith("Question: what sells together?") and "FACTS:" in user and BUNT in user


def test_rounding_and_name_case_are_accepted():
    text = f"People most often buy {BUNT.title()} together with {REG.lower()}, in roughly 46% of orders."
    e, _ = explain(FREQ, [text])
    assert e.source == "llm", e.problems


def test_an_invented_number_is_rejected_and_the_model_gets_one_more_try():
    bad = GOOD.replace("46%", "52%")
    e, llm = explain(FREQ, [bad, GOOD])
    assert e.source == "llm" and e.attempts == 2 and e.text == GOOD
    assert "numbers that are not in the data: 52%" in e.problems[0]
    retry = llm.calls[1]["user"]
    assert "rejected" in retry and "52%" in retry and "calculate nothing" in retry and "FACTS:" in retry


def test_a_calculated_figure_is_rejected():
    e, _ = explain(FREQ, [GOOD + " That is twice as often as the next pair.", GOOD])
    assert e.attempts == 2 and "comparisons or calculations" in e.problems[0] and "twice" in e.problems[0]
    e, _ = explain(FREQ, [GOOD + " That is 17% more than the next pair.", GOOD])
    assert e.attempts == 2 and "numbers that are not in the data: 17%" in e.problems[0]


def test_an_invented_product_is_rejected():
    e, _ = explain(FREQ, [GOOD + " Also try the MEGA MUG SET.", GOOD])
    assert e.attempts == 2 and "MEGA MUG SET" in e.problems[0]


def test_dates_are_rejected():
    e, _ = explain(FREQ, [GOOD + " This held throughout 2011.", GOOD])
    assert e.attempts == 2 and "2011" in e.problems[0]


def test_after_two_bad_answers_the_template_is_used():
    bad = GOOD.replace("46%", "52%")
    e, llm = explain(FREQ, [bad, bad])
    f = build_facts(result(FREQ), "what sells together?")
    assert e.source == "template" and e.text == render_template(f) and e.attempts == 2 and len(llm.calls) == 2
    assert "failed the number check" in e.reason and "52%" in e.reason


def test_an_empty_answer_counts_as_a_problem():
    e, _ = explain(FREQ, ["   ", GOOD])
    assert e.source == "llm" and e.attempts == 2 and e.problems == ["The answer was empty."]


def test_a_model_that_is_down_still_gives_an_answer():
    e, llm = explain(FREQ, [LLMUnavailable("Gemini rejected the API key.")])
    assert e.source == "template" and e.attempts == 1 and "46.15%" in e.text
    assert "unavailable" in e.reason and "rejected the API key" in e.reason


def test_nothing_to_explain_never_calls_the_model():
    e, llm = explain({"kind": "frequent", "min_support": 0.9}, [])
    assert e.source == "template" and llm.calls == [] and e.reason == "no results to explain"
    assert "46.15%" in e.text


# ---------------------------------------------------------------- logging, serialisation, the whole chain
def test_explanations_are_logged(tmp_path):
    log = tmp_path / "log.jsonl"
    explain(FREQ, [GOOD.replace("46%", "52%"), GOOD], log_path=log)
    (line,) = [json.loads(x) for x in log.read_text().splitlines()]
    assert line["type"] == "explanation" and line["source"] == "llm" and line["attempts"] == 2
    assert line["question"] == "what sells together?" and line["problems"] and line["kind"] == "frequent"


def test_explanation_is_json_serialisable():
    for replies in ([GOOD], ["bad 99%", "bad 99%"]):
        assert json.dumps(explain(FREQ, replies)[0].to_dict())


def test_sentence_to_answer():
    """question -> model -> spec -> validator -> engine -> model -> verified answer"""
    spec_reply = '{"intent":"mining_request","spec":{"kind":"frequent","min_support":0.3}}'
    parsed = parse_question("what do people buy together?", FakeLLM([spec_reply]), catalog=CAT, df=DF, log_path=None)
    assert parsed.status == "ok"
    res = run_mining(parsed.validation.request, df=DF)
    e = explain_result(res, FakeLLM([GOOD]), question="what do people buy together?", log_path=None)
    assert e.source == "llm" and "46%" in e.text


# ---------------------------------------------------------------- real data
real = pytest.mark.skipif(not C.CLEAN_FILE.exists(), reason="run prepare_data first")


@real
def test_real_data_answers_are_verified():
    r = run_mining({"kind": "frequent", "min_support": 0.01})
    f = build_facts(r)
    a, b = f.data["results"][0]["products"]
    honest = f"The most common pairing is {a} with {b}, found in about 4.5% of orders (808 orders)."
    wrong = f"The most common pairing is {a} with {b}, found in 7.8% of orders (808 orders)."
    assert nc.check_text(honest, f.allowed, f.names) == []
    assert "7.8%" in nc.check_text(wrong, f.allowed, f.names)[0]
    assert nc.check_text(render_template(f), f.allowed, f.names) == []
    e = explain_result(r, FakeLLM([wrong, honest]), log_path=None)
    assert e.source == "llm" and e.attempts == 2 and e.text == honest


@real
@needs_spmf
def test_real_data_high_utility_template_is_clean():
    r = run_mining({"kind": "high_utility", "min_utility": 10000})
    f = build_facts(r)
    text = render_template(f)
    assert nc.check_text(text, f.allowed, f.names) == [] and "PARTY BUNTING" in text and "revenue" in text


# ---------------------------------------------------------------- the interactive demo, with a fake model
@real
def test_demo_prints_a_verified_answer(capsys):
    from src.demo_parse import show

    def honest(system, user):                              # a 'model' that writes only from the facts it is given
        row = json.loads(user.split("FACTS:\n", 1)[1])["results"][0]
        return (f"The most common pairing is {' and '.join(row['products'])}, found in about "
                f"{row['share_of_orders_pct']:.1f}% of orders. Consider a bundle.")

    spec = '{"intent":"mining_request","spec":{"kind":"frequent","strictness":"strict"}}'
    show("what sells together?", FakeLLM([spec, honest]), run=True, explain=True)
    out = capsys.readouterr().out
    assert "STATUS: ok" in out and "ANSWER  (written by: llm; model calls: 1" in out
    assert "JUMBO BAG PINK POLKADOT and JUMBO BAG RED RETROSPOT" in out


@real
def test_demo_shows_why_the_template_was_used(capsys):
    from src.demo_parse import show
    spec = '{"intent":"mining_request","spec":{"kind":"frequent","strictness":"strict"}}'
    show("what sells together?", FakeLLM([spec, "It is in 99% of orders.", "It is in 99% of orders."]), run=True, explain=True)
    out = capsys.readouterr().out
    assert "written by: template; model calls: 2" in out
    assert "[fixed template used because: the model's answer failed the number check" in out
    assert "[rejected model answer: numbers that are not in the data: 99%]" in out
    assert "JUMBO BAG PINK POLKADOT" in out and "4.45%" in out          # the template's verified numbers
