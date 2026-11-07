"""Frequent itemset mining (Apriori / FP-Growth) behind one function.

    result = mine_frequent(baskets, min_support=0.01, algorithm="fpgrowth")
    result.itemsets   -> DataFrame: itemset (tuple), support, count, length

Support = fraction of baskets containing the whole itemset. The denominator is
`result.n_baskets`, i.e. the baskets actually mined (see drop_single_item_baskets).
"""
from dataclasses import dataclass, field
import math
import time

import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth

from src.core import config as C
from src.core.errors import InvalidRequest, LimitExceeded

ALGORITHMS = {"apriori": apriori, "fpgrowth": fpgrowth}


@dataclass
class MiningResult:
    itemsets: pd.DataFrame
    n_baskets: int
    algorithm: str
    min_support: float
    runtime_s: float
    params: dict = field(default_factory=dict)


def build_basket_matrix(baskets: list[list[str]], min_support: float) -> pd.DataFrame:
    """One-hot (baskets x items) boolean matrix, keeping only items that can
    possibly reach `min_support` (any itemset containing a rarer item can't)."""
    n = len(baskets)
    min_count = max(1, math.ceil(min_support * n))

    counts: dict[str, int] = {}
    for b in baskets:
        for item in set(b):
            counts[item] = counts.get(item, 0) + 1
    keep = sorted(i for i, c in counts.items() if c >= min_count)
    col = {item: j for j, item in enumerate(keep)}

    import numpy as np
    matrix = np.zeros((n, len(keep)), dtype=bool)
    for r, b in enumerate(baskets):
        for item in set(b):
            j = col.get(item)
            if j is not None:
                matrix[r, j] = True
    return pd.DataFrame(matrix, columns=keep)


def mine_frequent(
    baskets: list[list[str]],
    min_support: float = 0.01,
    algorithm: str = "fpgrowth",
    min_len: int = 2,
    max_len: int | None = None,
    drop_single_item_baskets: bool = True,
) -> MiningResult:
    if algorithm not in ALGORITHMS:
        raise InvalidRequest(f"algorithm must be one of {list(ALGORITHMS)}")
    if not 0 < min_support <= 1:
        raise InvalidRequest("min_support must be in (0, 1]")
    if min_support < C.MIN_SUPPORT_FLOOR:
        raise LimitExceeded(
            f"min_support {min_support} is below the safety floor "
            f"{C.MIN_SUPPORT_FLOOR}: the number of itemsets explodes and mining "
            "would take minutes or run out of memory. Use a higher value."
        )

    if drop_single_item_baskets:
        baskets = [b for b in baskets if len(set(b)) >= 2]
    n = len(baskets)

    start = time.perf_counter()
    matrix = build_basket_matrix(baskets, min_support)
    if matrix.shape[1] == 0:
        found = pd.DataFrame(columns=["support", "itemsets"])
    else:
        extra = {"low_memory": True} if algorithm == "apriori" else {}
        found = ALGORITHMS[algorithm](
            matrix, min_support=min_support, use_colnames=True, max_len=max_len, **extra
        )
    runtime = time.perf_counter() - start

    out = pd.DataFrame({
        "itemset": [tuple(sorted(s)) for s in found["itemsets"]],
        "support": found["support"].astype(float),
    })
    out["count"] = (out["support"] * n).round().astype(int)
    out["length"] = out["itemset"].map(len)
    out = out[out["length"] >= min_len]
    out = out.sort_values(["support", "length"], ascending=[False, False]).reset_index(drop=True)
    return MiningResult(out, n, algorithm, min_support, runtime)
