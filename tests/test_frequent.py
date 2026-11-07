import pytest
from src.mining.frequent import mine_frequent

# 5 baskets, hand-countable
BASKETS = [
    ["bread", "butter", "milk"],
    ["bread", "butter"],
    ["bread", "milk"],
    ["butter", "jam"],
    ["bread", "butter", "jam"],
]


@pytest.mark.parametrize("algo", ["apriori", "fpgrowth"])
def test_hand_computed_supports(algo):
    r = mine_frequent(BASKETS, min_support=0.4, algorithm=algo, min_len=2)
    got = {s: c for s, c in zip(r.itemsets["itemset"], r.itemsets["count"])}
    assert r.n_baskets == 5
    assert got[("bread", "butter")] == 3      # baskets 1, 2, 5
    assert got[("bread", "milk")] == 2        # baskets 1, 3
    assert got[("butter", "jam")] == 2        # baskets 4, 5
    assert ("butter", "milk") not in got      # only basket 1 -> count 1 < 2
    assert ("jam", "milk") not in got         # never together
    assert all(c >= 2 for c in got.values())  # 0.4 x 5 = 2


def test_algorithms_agree():
    a = mine_frequent(BASKETS, 0.2, "apriori", min_len=1)
    f = mine_frequent(BASKETS, 0.2, "fpgrowth", min_len=1)
    assert set(a.itemsets["itemset"]) == set(f.itemsets["itemset"])


def test_single_item_baskets_dropped():
    r = mine_frequent(BASKETS + [["bread"]], 0.4, "fpgrowth", drop_single_item_baskets=True)
    assert r.n_baskets == 5
    r2 = mine_frequent(BASKETS + [["bread"]], 0.4, "fpgrowth", drop_single_item_baskets=False)
    assert r2.n_baskets == 6


def test_max_len_and_bad_inputs():
    r = mine_frequent(BASKETS, 0.2, "fpgrowth", max_len=2)
    assert r.itemsets["length"].max() <= 2
    with pytest.raises(ValueError):
        mine_frequent(BASKETS, 0.0)
    with pytest.raises(ValueError):
        mine_frequent(BASKETS, 0.001)          # below safety floor
    with pytest.raises(ValueError):
        mine_frequent(BASKETS, 0.1, algorithm="nope")


def test_nothing_frequent_returns_empty():
    r = mine_frequent(BASKETS, 1.0, "fpgrowth")
    assert len(r.itemsets) == 0
