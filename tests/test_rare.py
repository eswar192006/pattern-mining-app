import random
from collections import Counter
from itertools import combinations

import pytest

from src.core import config as C
from src.mining.rare import mine_rare


# ---------- hand-counted case ----------
HAND = (
    [["a", "b"]] * 4 + [["a", "c"]] + [["d", "e", "f"]] * 4 + [["x", "y"]] * 20
)  # n = 29


def test_hand_counted_pairs_and_triple():
    r = mine_rare(HAND, max_support=0.2, min_count=3, min_strength=0.5, max_len=3)
    got = {s: (c, round(st, 3)) for s, c, st in
           zip(r.itemsets["itemset"], r.itemsets["count"], r.itemsets["strength"])}
    assert r.n_baskets == 29
    assert got == {
        ("a", "b"): (4, 0.8),            # a in 5 baskets, b in 4 -> 4/5
        ("d", "e"): (4, 1.0), ("d", "f"): (4, 1.0), ("e", "f"): (4, 1.0),
        ("d", "e", "f"): (4, 1.0),
    }
    # (x,y) is too common to be rare (20 > 0.2 x 29); (a,c) appears only once
    assert ("x", "y") not in got and ("a", "c") not in got


def test_lift_value():
    r = mine_rare(HAND, max_support=0.2, min_count=3, min_strength=0.5, max_len=2)
    ab = r.itemsets[r.itemsets["itemset"] == ("a", "b")].iloc[0]
    assert ab["lift"] == pytest.approx(29 * 4 / (5 * 4))


def test_pairs_only_by_default():
    r = mine_rare(HAND, max_support=0.2, min_count=3, min_strength=0.5)
    assert set(r.itemsets["length"]) == {2}


# ---------- brute-force cross-check on random data ----------
def brute_force(baskets, max_support, min_count, min_strength, max_len):
    baskets = [sorted(set(b)) for b in baskets if len(set(b)) >= 2]
    n = len(baskets)
    item_cnt = Counter(i for b in baskets for i in b)
    out = {}
    for k in range(2, max_len + 1):
        cnt = Counter()
        for b in baskets:
            for combo in combinations(b, k):
                cnt[combo] += 1
        for combo, c in cnt.items():
            strength = c / max(item_cnt[i] for i in combo)
            if c >= min_count and c <= max_support * n and strength >= min_strength:
                out[combo] = c
    return out


@pytest.mark.parametrize("seed,max_len", [(1, 2), (2, 3), (3, 3)])
def test_matches_brute_force(seed, max_len):
    rnd = random.Random(seed)
    items = [f"i{j:02d}" for j in range(25)]
    weights = [1 + (j % 5) * 3 for j in range(25)]          # mix of rare and common items
    baskets = [rnd.choices(items, weights, k=rnd.randint(1, 7)) for _ in range(400)]
    # plant a strong rare triple so triples are actually exercised
    baskets += [["i00", "i05", "i10"]] * 6

    kw = dict(max_support=0.08, min_count=3, min_strength=0.2, max_len=max_len)
    fast = mine_rare(baskets, **kw)
    got = dict(zip(fast.itemsets["itemset"], fast.itemsets["count"]))
    expected = brute_force(baskets, **kw)

    assert got == expected
    assert len(expected) > 0
    if max_len == 3:
        assert any(len(k) == 3 for k in expected)


# ---------- behaviour and safety ----------
def test_single_item_baskets_dropped():
    r = mine_rare(HAND + [["a"]] * 5, max_support=0.2, min_count=3, min_strength=0.5)
    assert r.n_baskets == 29


def test_top_k_keeps_best_and_reports_total():
    r = mine_rare(HAND, max_support=0.2, min_count=3, min_strength=0.5, max_len=3, top_k=2)
    assert len(r.itemsets) == 2
    assert r.params["total_found"] == 5
    assert (r.itemsets["strength"] == 1.0).all()


def test_bad_inputs():
    for bad in (dict(max_support=0), dict(min_count=1), dict(min_strength=0),
                dict(min_strength=1.5), dict(max_len=4)):
        with pytest.raises(ValueError):
            mine_rare(HAND, **bad)


def test_result_cap(monkeypatch):
    monkeypatch.setattr(C, "MAX_RARE_RESULTS", 2)
    with pytest.raises(ValueError, match="limit"):
        mine_rare(HAND, max_support=0.2, min_count=3, min_strength=0.5, max_len=3)


def test_empty_input():
    r = mine_rare([], max_support=0.1)
    assert len(r.itemsets) == 0 and r.n_baskets == 0
