"""Cross-check on the REAL data (skipped if data/processed/clean_transactions.csv
is missing). The rare miner and the frequent miner are independent implementations;
where their ranges overlap they must return identical pairs and counts."""
import pytest

from src.core import config as C
from src.mining.data_loader import load_clean, to_baskets
from src.mining.frequent import mine_frequent
from src.mining.rare import mine_rare

pytestmark = pytest.mark.skipif(not C.CLEAN_FILE.exists(), reason="run prepare_data first")


def test_rare_pairs_match_frequent_pairs_in_overlap():
    baskets = to_baskets(load_clean())

    # frequent miner: all pairs with support >= 0.5%  (count >= 91 of 18,141 baskets)
    f = mine_frequent(baskets, min_support=0.005, algorithm="fpgrowth", min_len=2, max_len=2)
    n = f.n_baskets
    upper = int(0.02 * n)                                   # keep counts 91..362
    expected = {s: c for s, c in zip(f.itemsets["itemset"], f.itemsets["count"]) if c <= upper}

    # rare miner: same count window, strength filter effectively off
    r = mine_rare(baskets, max_support=0.02, min_count=91, min_strength=1e-6, max_len=2)
    got = dict(zip(r.itemsets["itemset"], r.itemsets["count"]))

    assert len(expected) > 1000
    assert got == expected
