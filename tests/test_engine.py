import json
import shutil

import pandas as pd
import pytest

from src.core import config as C
from src.core.errors import EngineUnavailable, InvalidRequest, LimitExceeded, MiningError
from src.mining.engine import run_mining

needs_spmf = pytest.mark.skipif(
    shutil.which("java") is None or not C.SPMF_JAR.exists(),
    reason="needs Java and tools/spmf.jar",
)

TEA, MUG, CUP, JAM = "TEA", "MUG", "CUP", "JAM"
PRICE = {TEA: 5.0, MUG: 6.0, CUP: 4.0, JAM: 10.0}
WEEKEND = {"Saturday", "Sunday"}


def make_df():
    """15 baskets with 2+ products (+1 single-product basket).
    6 Sunday/UK/Dec: TEA+MUG      4 Monday/UK/Jan: TEA+CUP
    3 Tuesday/France/Feb: MUG+CUP+JAM      2 Friday/UK/Mar: TEA+MUG+CUP     1 Monday: TEA only"""
    spec = ([("s", 6, "Sunday", "UK", 12, [TEA, MUG])] + [("m", 4, "Monday", "UK", 1, [TEA, CUP])]
            + [("t", 3, "Tuesday", "France", 2, [MUG, CUP, JAM])] + [("f", 2, "Friday", "UK", 3, [TEA, MUG, CUP])]
            + [("x", 1, "Monday", "UK", 1, [TEA])])
    rows = []
    for tag, n, day, country, month, items in spec:
        for k in range(n):
            for it in items:
                rows.append((f"{tag}{k}", it, PRICE[it], day, day in WEEKEND, month, country))
    return pd.DataFrame(rows, columns=["basket_id", "item", "utility", "weekday", "is_weekend", "month", "country"])


DF = make_df()


def sets(result):
    return {s: c for s, c in zip(result.itemsets["itemset"], result.itemsets["count"])}


# ---------- the three kinds, hand-counted ----------
def test_frequent_hand_counted():
    r = run_mining({"kind": "frequent", "min_support": 0.2}, df=DF)
    got = sets(r)
    assert r.n_baskets_selected == 16 and r.n_baskets_mined == 15
    assert got[(MUG, TEA)] == 8 and got[(CUP, TEA)] == 6 and got[(CUP, MUG)] == 5
    assert got[(CUP, JAM, MUG)] == 3                      # triple (no max_len for frequent)
    assert r.itemsets.iloc[0]["rank"] == 1 and r.itemsets.iloc[0]["itemset"] == (MUG, TEA)


def test_rare_hand_counted():
    r = run_mining({"kind": "rare", "max_support": 0.3, "min_count": 3, "min_strength": 0.2, "max_len": 3}, df=DF)
    assert sets(r) == {(CUP, JAM): 3, (JAM, MUG): 3, (CUP, JAM, MUG): 3}


@needs_spmf
def test_high_utility_hand_counted():
    r = run_mining({"kind": "high_utility", "min_utility": 50, "max_len": 3}, df=DF)
    util = {s: u for s, u in zip(r.itemsets["itemset"], r.itemsets["utility"])}
    assert util == {(MUG, TEA): 88.0, (CUP, TEA): 54.0, (CUP, MUG): 50.0, (CUP, JAM, MUG): 60.0}
    assert r.params["total_utility"] == 192.0
    assert any("revenue" in n for n in r.notes)


# ---------- basket filters ----------
def test_weekday_and_weekend_filters():
    r = run_mining({"kind": "frequent", "min_support": 0.5, "filters": {"weekdays": ["sunday"]}}, df=DF)
    assert r.n_baskets_selected == 6 and sets(r) == {(MUG, TEA): 6}
    assert r.filter_trace[-1][1] == 6
    w = run_mining({"kind": "frequent", "min_support": 0.5, "filters": {"weekend": True}}, df=DF)
    assert w.n_baskets_selected == 6
    assert any("only Sunday" in x for x in w.warnings)       # no Saturday data
    wd = run_mining({"kind": "frequent", "min_support": 0.2, "filters": {"weekend": False}}, df=DF)
    assert wd.n_baskets_selected == 10


def test_month_and_country_filters_case_insensitive():
    assert run_mining({"kind": "frequent", "min_support": 0.2, "filters": {"months": [12]}}, df=DF).n_baskets_selected == 6
    r = run_mining({"kind": "frequent", "min_support": 0.5, "filters": {"countries": ["france"]}}, df=DF)
    assert r.n_baskets_selected == 3 and (CUP, JAM, MUG) in sets(r)
    with pytest.raises(InvalidRequest, match="Did you mean: France"):
        run_mining({"kind": "frequent", "filters": {"countries": ["Frnce"]}}, df=DF)


def test_filter_that_removes_everything_explains_itself():
    r = run_mining({"kind": "frequent", "filters": {"weekdays": ["Saturday"]}}, df=DF)
    assert len(r.itemsets) == 0 and r.n_baskets_selected == 0
    assert "No baskets remain" in r.hint and any("no orders on Saturday" in w for w in r.warnings)


# ---------- product filters ----------
def test_exclude_items_removes_product_before_mining():
    r = run_mining({"kind": "frequent", "min_support": 0.2, "filters": {"exclude_items": ["tea"]}}, df=DF)
    assert all(TEA not in s for s in r.itemsets["itemset"])
    assert (CUP, MUG) in sets(r)
    assert r.n_baskets_selected == 15                         # the TEA-only basket disappears


def test_include_items_keeps_only_results_with_them():
    full = run_mining({"kind": "frequent", "min_support": 0.2}, df=DF)
    r = run_mining({"kind": "frequent", "min_support": 0.2, "filters": {"include_items": ["JAM"]}}, df=DF)
    assert r.total_found < full.total_found
    assert r.total_found > 0 and all(JAM in s for s in r.itemsets["itemset"])
    assert any("all selected baskets" in n for n in r.notes)


def test_include_item_that_matches_nothing_explains_why():
    r = run_mining({"kind": "frequent", "min_support": 0.5, "filters": {"include_items": ["JAM"]}}, df=DF)
    assert r.total_found == 0
    assert "none contain JAM" in r.hint


def test_bad_product_filters():
    with pytest.raises(InvalidRequest, match="Did you mean: MUG"):
        run_mining({"kind": "frequent", "filters": {"include_items": ["MUGG"]}}, df=DF)
    with pytest.raises(InvalidRequest, match="both included and excluded"):
        run_mining({"kind": "frequent", "filters": {"include_items": ["MUG"], "exclude_items": ["MUG"]}}, df=DF)
    with pytest.raises(InvalidRequest, match="limited to 2"):
        run_mining({"kind": "rare", "max_len": 2, "filters": {"include_items": ["MUG", "TEA", "CUP"]}}, df=DF)


# ---------- result shaping ----------
def test_top_k_and_total_found():
    r = run_mining({"kind": "frequent", "min_support": 0.2, "top_k": 2}, df=DF)
    assert len(r.itemsets) == 2 and r.total_found > 2
    assert list(r.itemsets["rank"]) == [1, 2]


def test_json_round_trip():
    r = run_mining({"kind": "rare", "max_support": 0.3, "min_count": 3, "min_strength": 0.2}, df=DF)
    d = json.loads(json.dumps(r.to_dict()))
    assert d["kind"] == "rare" and d["shown"] == len(d["itemsets"]) > 0
    assert isinstance(d["itemsets"][0]["itemset"], list) and "strength" in d["itemsets"][0]


# ---------- hints for empty results ----------
def test_frequent_empty_result_hint_names_best_pair():
    r = run_mining({"kind": "frequent", "min_support": 0.9}, df=DF)
    assert r.total_found == 0
    assert "MUG + TEA" in r.hint and "53.33%" in r.hint


def test_rare_empty_result_hint():
    r = run_mining({"kind": "rare", "max_support": 0.3, "min_count": 3, "min_strength": 0.9}, df=DF)
    assert r.total_found == 0 and "strongest rare pair" in r.hint


@needs_spmf
def test_high_utility_empty_result_hint():
    r = run_mining({"kind": "high_utility", "min_utility": 100}, df=DF)
    assert r.total_found == 0 and "MUG + TEA" in r.hint and "88" in r.hint


# ---------- request validation and typed errors ----------
def test_invalid_requests():
    for bad in ({"kind": "nope"}, {"kind": "frequent", "surprise": 1}, {"kind": "frequent", "min_support": 2},
                {"kind": "rare", "max_len": 4}, {"kind": "frequent", "algorithm": "efim"},
                {"kind": "high_utility", "min_utility": 5, "min_utility_pct": 0.1}):
        with pytest.raises(InvalidRequest):
            run_mining(bad, df=DF)


def test_ignored_settings_produce_a_warning():
    r = run_mining({"kind": "frequent", "min_support": 0.2, "min_strength": 0.5}, df=DF)
    assert any("min_strength" in w for w in r.warnings)


def test_safety_limits_surface_as_typed_errors():
    with pytest.raises(LimitExceeded) as e:
        run_mining({"kind": "frequent", "min_support": 0.001}, df=DF)
    assert isinstance(e.value, MiningError) and isinstance(e.value, ValueError)


def test_missing_spmf_is_a_typed_error(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "SPMF_JAR", tmp_path / "missing.jar")
    with pytest.raises(EngineUnavailable, match="spmf.jar"):
        run_mining({"kind": "high_utility", "min_utility": 50}, df=DF)


# ---------- real data: engine must equal the miners it calls ----------
real = pytest.mark.skipif(not C.CLEAN_FILE.exists(), reason="run prepare_data first")


@real
def test_real_frequent_and_sunday_filter_match_direct_calls():
    from src.mining.data_loader import load_clean, to_baskets
    from src.mining.frequent import mine_frequent
    df = load_clean()

    r = run_mining({"kind": "frequent", "min_support": 0.01, "top_k": 500}, df=df)
    direct = mine_frequent(to_baskets(df), 0.01, "fpgrowth")
    assert r.total_found == len(direct.itemsets) == 1601
    assert list(r.itemsets["itemset"]) == list(direct.itemsets["itemset"].head(500))

    s = run_mining({"kind": "frequent", "min_support": 0.01, "top_k": 500, "filters": {"weekdays": ["Sunday"]}}, df=df)
    sub = df[df["weekday"] == "Sunday"]
    direct_s = mine_frequent(to_baskets(sub), 0.01, "fpgrowth")
    assert s.n_baskets_selected == sub["basket_id"].nunique() == 2190
    assert s.total_found == len(direct_s.itemsets)
    assert list(s.itemsets["itemset"]) == list(direct_s.itemsets["itemset"].head(500))


@real
def test_real_rare_and_empty_hint():
    from src.mining.data_loader import load_clean, to_baskets
    from src.mining.rare import mine_rare
    df = load_clean()
    r = run_mining({"kind": "rare"}, df=df)
    assert r.total_found == mine_rare(to_baskets(df)).params["total_found"] == 2051

    e = run_mining({"kind": "frequent", "min_support": 0.05}, df=df)
    assert e.total_found == 0
    assert "JUMBO BAG PINK POLKADOT" in e.hint and "4.45%" in e.hint


@real
@needs_spmf
def test_real_high_utility_through_engine():
    from src.mining.data_loader import load_clean
    r = run_mining({"kind": "high_utility", "min_utility": 10000}, df=load_clean())
    assert r.total_found == 1615
    assert r.itemsets.iloc[0]["itemset"] == ("PARTY BUNTING", "REGENCY CAKESTAND 3 TIER")
    assert json.dumps(r.to_dict())
