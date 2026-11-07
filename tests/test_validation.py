import json

import pandas as pd
import pytest

from src.core import config as C
from src.core.catalog import Catalog, get_catalog
from src.core.spec import spec_json_schema
from src.core.validation import validate_spec
from src.mining.engine import run_mining

REG, SWEET = "REGENCY CAKESTAND 3 TIER", "SWEETHEART CAKESTAND 3 TIER"
BUNT, RED, PINK, STORE = "PARTY BUNTING", "JUMBO BAG RED RETROSPOT", "JUMBO BAG PINK POLKADOT", "JUMBO STORAGE BAG SUKI"


def make_df():
    """65 baskets: 30 Sunday/UK/Dec {REG,BUNT}, 20 Monday/UK/Jan {RED,PINK},
    10 Tuesday/France/Feb {SWEET,STORE,BUNT}, 5 Friday/Germany/Mar {REG,RED}."""
    spec = [("a", 30, "Sunday", "UK", 12, [REG, BUNT]), ("b", 20, "Monday", "UK", 1, [RED, PINK]),
            ("c", 10, "Tuesday", "France", 2, [SWEET, STORE, BUNT]), ("d", 5, "Friday", "Germany", 3, [REG, RED])]
    rows = [(f"{t}{k}", it, 1.0, day, day in ("Saturday", "Sunday"), month, country)
            for t, n, day, country, month, items in spec for k in range(n) for it in items]
    return pd.DataFrame(rows, columns=["basket_id", "item", "utility", "weekday", "is_weekend", "month", "country"])


DF = make_df()
CAT = Catalog(DF)


def v(spec):
    return validate_spec(spec, df=DF, catalog=CAT)


# ---------- catalogue ----------
def test_catalog_matching():
    assert CAT.exact_product("regency cakestand 3-tier") == REG            # case and punctuation ignored
    assert CAT.match_products("cakestand") == [REG, SWEET]                 # most popular first
    assert CAT.match_products("bags") == [RED, PINK, STORE]                # plural handled
    assert CAT.match_products("red retrospot jumbo") == [RED]              # word order ignored
    assert CAT.match_products("teapot") == []
    assert CAT.similar_products("regancy cakestand 3 tier")[0] == REG      # typo -> suggestion


def test_catalog_countries_and_summary():
    assert CAT.match_country("uk") == ("UK", [])                             # the data's own name, any case
    assert CAT.match_country("france") == ("France", [])
    alias = Catalog(DF.assign(country=DF["country"].replace({"UK": "United Kingdom"})))
    assert alias.match_country("uk") == ("United Kingdom", [])              # common abbreviation -> data's name
    assert CAT.match_country("Frnce") == (None, ["France"])
    s = CAT.prompt_summary()
    assert "65 baskets" in s and "No orders on Wednesday, Thursday, Saturday" in s


# ---------- presets and the happy path ----------
@pytest.mark.parametrize("kind,strictness,field,value", [
    ("frequent", "loose", "min_support", 0.007), ("frequent", "strict", "min_support", 0.02),
    ("rare", "normal", "min_strength", 0.3), ("high_utility", "strict", "min_utility_pct", 0.002)])
def test_strictness_becomes_numbers(kind, strictness, field, value):
    spec = {"kind": kind, "strictness": strictness}
    if kind == "rare":
        spec.update(max_support=0.3, min_count=3)          # the tiny dataset needs a workable 'rare' band
    r = v(spec)
    assert r.status == "ok", (r.questions, r.errors)
    assert getattr(r.request, field) == value
    assert any("Strictness" in n for n in r.notes)


def test_request_contains_only_relevant_fields():
    r = v({"kind": "frequent", "min_strength": 0.5})
    assert r.status == "ok" and any("min_strength" in w for w in r.warnings)
    assert "min_strength" not in r.request.model_fields_set
    res = run_mining(r.request, df=DF)                                       # the engine accepts it...
    assert not any("Ignored settings" in w for w in res.warnings)            # ...and has nothing left to ignore


def test_explicit_numbers_override_the_word():
    r = v({"kind": "frequent", "strictness": "strict", "min_support": 0.4})
    assert r.request.min_support == 0.4


# ---------- questions ----------
def test_missing_kind_asks_with_three_options():
    r = v({})
    assert r.status == "needs_clarification" and r.request is None
    q = r.questions[0]
    assert q.field == "kind" and len(q.options) == 3


def test_ambiguous_product_asks_and_never_guesses():
    r = v({"kind": "frequent", "filters": {"include_products": ["cakestand"]}})
    assert r.status == "needs_clarification" and r.request is None
    assert r.questions[0].options == [REG, SWEET]


def test_unique_product_is_resolved_with_a_note():
    r = v({"kind": "frequent", "filters": {"include_products": ["regency cakestand"]}})
    assert r.status == "ok" and r.request.filters.include_items == [REG]
    assert any("-> " + REG in n for n in r.notes)


def test_exclude_keyword_covers_a_group():
    r = v({"kind": "frequent", "filters": {"exclude_products": ["jumbo"]}})
    assert r.status == "ok" and set(r.request.filters.exclude_items) == {RED, PINK, STORE}
    assert any("matched 3" in n for n in r.notes)


def test_unknown_product_and_country_offer_suggestions():
    r = v({"kind": "frequent", "filters": {"include_products": ["regancy cakestand 3 tier"], "countries": ["Frnce"]}})
    assert r.status == "needs_clarification"
    by_field = {q.field: q for q in r.questions}
    assert REG in by_field["filters.include_products[0]"].options
    assert by_field["filters.countries[0]"].options == ["France"]


def test_all_problems_reported_at_once():
    r = v({"filters": {"include_products": ["cakestand"], "countries": ["Frnce"]}})
    assert {q.field for q in r.questions} == {"kind", "filters.include_products[0]", "filters.countries[0]"}


def test_weekday_without_data_asks_for_another_day():
    r = v({"kind": "frequent", "filters": {"weekdays": ["saturday"]}})
    assert r.status == "needs_clarification"
    assert "Saturday" in r.questions[0].question and "Sunday" in r.questions[0].options
    assert "Saturday" not in r.questions[0].options


def test_filters_that_remove_everything_ask_which_to_relax():
    r = v({"kind": "frequent", "filters": {"months": [7]}})
    assert r.status == "needs_clarification" and r.questions[0].field == "filters"
    assert "month in [7]" in r.questions[0].question


def test_rare_is_impossible_on_a_tiny_selection():
    r = v({"kind": "rare"})                                  # 0.5% of 65 baskets = 0 baskets
    assert r.status == "needs_clarification" and "cannot reach" in r.questions[0].question
    assert "Switch to frequent patterns" in r.questions[0].options


# ---------- errors ----------
def test_contradictions_are_errors():
    r = v({"kind": "frequent", "filters": {"include_products": ["regency cakestand"], "exclude_products": [REG]}})
    assert r.status == "invalid" and "both included and excluded" in r.errors[0]
    r = v({"kind": "high_utility", "min_utility": 5, "min_utility_pct": 0.1})
    assert r.status == "invalid" and "not both" in r.errors[0]
    r = v({"kind": "frequent", "algorithm": "efim"})
    assert r.status == "invalid" and "not available" in r.errors[0]
    r = v({"kind": "rare", "max_support": 0.3, "min_count": 3,
           "filters": {"include_products": [REG, SWEET, BUNT]}})
    assert r.status == "invalid" and "limited to 2" in r.errors[0]


def test_malformed_spec_is_invalid():
    for bad in ({"kind": "nope"}, {"surprise": 1}, {"kind": "frequent", "min_support": 3}, {"filters": {"months": [13]}}):
        r = v(bad)
        assert r.status == "invalid" and r.errors and r.request is None


# ---------- warnings: values changed, never silently ----------
def test_safety_floors_raise_values_with_a_warning():
    r = v({"kind": "frequent", "min_support": 0.001})
    assert r.request.min_support == C.MIN_SUPPORT_FLOOR and any("raised to 0.5%" in w for w in r.warnings)
    r = v({"kind": "high_utility", "min_utility": 0.0001})
    assert r.request.min_utility is None and r.request.min_utility_pct == C.HUIM_MIN_UTILITY_PCT_FLOOR[2]
    assert any("lowest allowed" in w for w in r.warnings)
    r = v({"kind": "high_utility", "min_utility_pct": 0.0005, "max_len": 3})          # below the triples floor
    assert r.request.min_utility_pct == C.HUIM_MIN_UTILITY_PCT_FLOOR[3]


def test_size_and_count_limits_are_clamped_with_a_warning():
    r = v({"kind": "frequent", "top_k": 1000})
    assert r.request.top_k == 500 and any("top_k" in w for w in r.warnings)
    r = v({"kind": "high_utility", "max_len": 4})
    assert r.request.max_len == 3 and any("lowered from 4 to 3" in w for w in r.warnings)


def test_other_warnings():
    assert any("EFIM" in w for w in v({"kind": "high_utility", "algorithm": "efim"}).warnings)
    r = v({"kind": "rare", "algorithm": "fpgrowth", "max_support": 0.3, "min_count": 3})
    assert any("algorithm" in w for w in r.warnings) and r.request.algorithm is None
    assert any("coincidence" in w for w in v({"kind": "frequent", "min_support": 0.01}).warnings)


# ---------- plumbing ----------
def test_ok_request_runs_in_the_engine():
    r = v({"kind": "frequent", "min_support": 0.3})
    res = run_mining(r.request, df=DF)
    assert res.itemsets.iloc[0]["itemset"] == (BUNT, REG) and res.itemsets.iloc[0]["count"] == 30


def test_everything_is_json_serialisable():
    for spec in ({"kind": "frequent"}, {}, {"kind": "frequent", "filters": {"include_products": [REG], "exclude_products": [REG]}}):
        assert json.dumps(v(spec).to_dict())
    schema = spec_json_schema()
    assert "strictness" in schema["properties"] and "filters" in schema["properties"]


# ---------- real data ----------
real = pytest.mark.skipif(not C.CLEAN_FILE.exists(), reason="run prepare_data first")


@real
def test_real_data_validation():
    r = validate_spec({"kind": "high_utility"})
    assert r.status == "ok" and r.request.min_utility_pct == 0.001 and r.mineable_baskets == 18141

    r = validate_spec({"kind": "frequent", "filters": {"include_products": ["cakestand"]}})
    assert r.status == "needs_clarification" and r.questions[0].options[0] == REG

    r = validate_spec({"kind": "frequent", "filters": {"exclude_products": ["bunting"], "countries": ["uk"],
                                                          "include_products": ["regency cakestand"]}})
    assert r.status == "ok" and len(r.request.filters.exclude_items) == 18
    assert r.request.filters.countries == ["United Kingdom"]

    r = validate_spec({"kind": "rare", "filters": {"weekdays": ["Sunday"]}})
    assert r.status == "ok" and any("exactly 10 baskets" in w for w in r.warnings)
    assert "No orders on Saturday" in get_catalog().prompt_summary()
