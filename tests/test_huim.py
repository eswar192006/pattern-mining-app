import shutil
import sys
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src.core import config as C
from src.mining import huim
from src.mining.huim import java_available, mine_high_utility

needs_spmf = pytest.mark.skipif(
    not java_available() or not C.SPMF_JAR.exists(),
    reason="needs Java and tools/spmf.jar (download from the official SPMF site)",
)


def make_df(baskets):
    """baskets: {basket_id: {item: utility}}"""
    return pd.DataFrame([(b, i, u) for b, items in baskets.items() for i, u in items.items()],
                        columns=["basket_id", "item", "utility"])


HAND = make_df({
    "b1": {"a": 5.0, "b": 6.0, "c": 4.0},
    "b2": {"a": 5.0, "b": 6.0},
    "b3": {"b": 6.0, "c": 4.0},
    "b4": {"a": 5.0, "c": 4.0},
    "b5": {"a": 5.0, "b": 6.0, "c": 4.0},
})   # total utility = 60


# ---------- no Java needed: validation and safety ----------
def test_bad_arguments():
    with pytest.raises(ValueError):
        mine_high_utility(HAND, min_utility=None, min_utility_pct=None)
    with pytest.raises(ValueError):
        mine_high_utility(HAND, min_utility=0)
    with pytest.raises(ValueError):
        mine_high_utility(HAND, min_utility=10, max_len=4)
    with pytest.raises(ValueError):
        mine_high_utility(HAND, min_utility=10, algorithm="nope")


def test_safety_floor():
    with pytest.raises(ValueError, match="safety floor"):
        mine_high_utility(HAND, min_utility=0.001)          # far below 0.02% of total


def test_empty_input():
    r = mine_high_utility(HAND.iloc[0:0], min_utility_pct=0.1)
    assert len(r.itemsets) == 0


def test_watchdog_stops_slow_process(tmp_path):
    cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
    with pytest.raises(ValueError, match="longer than"):
        huim._run_spmf(cmd, str(tmp_path / "o.txt"), str(tmp_path / "l.txt"), timeout_s=1)


def test_watchdog_stops_huge_output(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "MAX_HUIM_OUTPUT_MB", 1)
    out = str(tmp_path / "o.txt")
    code = f"import time; f=open({out!r},'w'); f.write('x'*3000000); f.flush(); time.sleep(30)"
    with pytest.raises(ValueError, match="MB"):
        huim._run_spmf([sys.executable, "-c", code], out, str(tmp_path / "l.txt"), timeout_s=20)


# ---------- needs Java + SPMF ----------
@needs_spmf
def test_hand_counted_pairs_and_triple():
    r = mine_high_utility(HAND, min_utility=25.0, max_len=3, min_len=2)
    got = {s: (u, c) for s, u, c in zip(r.itemsets["itemset"], r.itemsets["utility"], r.itemsets["count"])}
    assert got == {
        ("a", "b"): (33.0, 3), ("a", "c"): (27.0, 3),
        ("b", "c"): (30.0, 3), ("a", "b", "c"): (30.0, 2),
    }
    assert r.n_baskets == 5 and r.params["total_utility"] == 60.0
    top = r.itemsets.iloc[0]
    assert top["itemset"] == ("a", "b") and top["utility_share"] == pytest.approx(33 / 60)


@needs_spmf
def test_max_len_limits_size():
    r = mine_high_utility(HAND, min_utility=25.0, max_len=2)
    assert set(r.itemsets["length"]) == {2}


@needs_spmf
def test_min_utility_pct_matches_money():
    a = mine_high_utility(HAND, min_utility=25.0, max_len=3)
    b = mine_high_utility(HAND, min_utility_pct=25 / 60, max_len=3)
    assert set(a.itemsets["itemset"]) == set(b.itemsets["itemset"])


def brute_force_cents(df, min_cents, min_len, max_len):
    out = {}
    for _, g in df.groupby("basket_id"):
        cents = {i: int(round(u * 100)) for i, u in zip(g["item"], g["utility"])}
        for k in range(min_len, max_len + 1):
            for combo in combinations(sorted(cents), k):
                out[combo] = out.get(combo, 0) + sum(cents[i] for i in combo)
    return {k: v for k, v in out.items() if v >= min_cents}


@needs_spmf
@pytest.mark.parametrize("seed,max_len,algorithm", [(1, 2, "fhm+"), (2, 3, "fhm+"), (3, 3, "efim")])
def test_matches_brute_force(seed, max_len, algorithm):
    rnd = np.random.default_rng(seed)
    items = [f"i{j:02d}" for j in range(14)]
    rows = {}
    for b in range(150):
        k = int(rnd.integers(2, 7))
        chosen = rnd.choice(items, size=k, replace=False)
        rows[f"b{b}"] = {str(i): float(rnd.integers(1, 41)) * 0.25 for i in chosen}
    df = make_df(rows)

    total = df["utility"].sum()
    min_money = 0.02 * total                      # well above the safety floor
    r = mine_high_utility(df, min_utility=min_money, max_len=max_len, min_len=2, algorithm=algorithm)
    got = {s: int(round(u * 100)) for s, u in zip(r.itemsets["itemset"], r.itemsets["utility"])}
    expected = brute_force_cents(df, int(np.ceil(min_money * 100 - 1e-6)), 2, max_len)

    assert got == expected
    assert len(expected) > 5


# ---------- real data, independent calculation (no Java involved in the reference) ----------
@needs_spmf
@pytest.mark.skipif(not C.CLEAN_FILE.exists(), reason="run prepare_data first")
def test_real_data_pairs_match_independent_matrix_calculation():
    from src.mining.data_loader import load_clean
    df = load_clean()
    r = mine_high_utility(df, min_utility=10000, max_len=2, min_len=2)
    got = {s: int(round(u * 100)) for s, u in zip(r.itemsets["itemset"], r.itemsets["utility"])}

    d = df.groupby(["basket_id", "item"], as_index=False)["utility"].sum()
    d = d[d.groupby("basket_id")["item"].transform("size") >= 2]
    b_codes, _ = pd.factorize(d["basket_id"])
    i_codes, i_names = pd.factorize(d["item"])
    cents = (d["utility"] * 100).round().astype("int64").to_numpy()
    shape = (b_codes.max() + 1, len(i_names))
    U = sp.csr_matrix((cents, (b_codes, i_codes)), shape=shape)
    B = sp.csr_matrix((np.ones(len(d), dtype=np.int64), (b_codes, i_codes)), shape=shape)
    M = (U.T @ B).tocsr()                         # M[a,b] = utility of a in baskets that contain b
    T = sp.triu(M + M.T, k=1).tocoo()             # pair utility = M[a,b] + M[b,a]
    keep = T.data >= 10000 * 100
    expected = {tuple(sorted((i_names[a], i_names[b]))): int(v)
                for a, b, v in zip(T.row[keep], T.col[keep], T.data[keep])}

    assert len(expected) > 1000
    assert got == expected
