"""Rare-but-meaningful itemset mining (pairs and triples), exact.

An itemset is reported when ALL of these hold:
  * rare:       support <= max_support           (uncommon)
  * not noise:  count   >= min_count             (appears in enough baskets)
  * meaningful: strength >= min_strength         (its items really belong together)

strength = ALL-CONFIDENCE = count(itemset) / count(most popular item in it).
Plain words: "of the baskets that contain the most popular item, at least this
fraction also contain all the other items." For a pair this is the weaker of
the two directions of confidence, so both items must pull each other along.

Why not lift? On this data >800,000 pairs are rare and almost all have a high
lift purely by coincidence (big wholesale baskets). All-confidence cuts that to
a few thousand real relationships (colour variants, matching sets).

All-confidence never rises when an item is added (downward closed), so every
triple's three pairs must already be strong pairs. That makes triples cheap to
find exactly without enumerating the huge space of all combinations.
"""
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.core import config as C
from src.core.errors import InvalidRequest, LimitExceeded
from src.mining.frequent import MiningResult


def _incidence(baskets, min_count):
    """Sparse basket x item 0/1 matrix, keeping only items with count >= min_count
    (an itemset containing a rarer item can never reach min_count)."""
    counts: dict[str, int] = {}
    for b in baskets:
        for item in set(b):
            counts[item] = counts.get(item, 0) + 1
    keep = sorted(i for i, c in counts.items() if c >= min_count)
    col = {item: j for j, item in enumerate(keep)}
    rows, cols = [], []
    for r, b in enumerate(baskets):
        for item in set(b):
            j = col.get(item)
            if j is not None:
                rows.append(r)
                cols.append(j)
    X = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.int32), (rows, cols)),
        shape=(len(baskets), len(keep)),
    )
    cnt = np.array([counts[i] for i in keep], dtype=np.int64)
    return X, keep, cnt


def _triples(X, cnt, pr, pc, min_count, min_strength, max_count):
    """Exact triples whose three pairs are all strong pairs (pr[i] < pc[i])."""
    nbr: dict[int, set] = {}
    for a, b in zip(pr.tolist(), pc.tolist()):
        nbr.setdefault(a, set()).add(b)

    work, total = [], 0
    for a, b in zip(pr.tolist(), pc.tolist()):
        cand = nbr.get(a, set()) & nbr.get(b, set())   # c > b, (a,c) and (b,c) strong
        if cand:
            total += len(cand)
            if total > C.MAX_TRIPLE_CANDIDATES:
                raise LimitExceeded(
                    f"Too many triple candidates (> {C.MAX_TRIPLE_CANDIDATES:,}). "
                    "Raise min_strength or min_count, or use max_len=2."
                )
            work.append((a, b, sorted(cand)))
    if not work:
        return []

    used = sorted({i for a, b, cs in work for i in (a, b, *cs)})
    Xc = X.tocsc()
    tid = {i: Xc[:, i].toarray().ravel().astype(bool) for i in used}

    found = []
    for a, b, cs in work:
        ab = tid[a] & tid[b]
        for c in cs:
            count = int(np.count_nonzero(ab & tid[c]))
            if min_count <= count <= max_count:
                strength = count / max(cnt[a], cnt[b], cnt[c])
                if strength >= min_strength:
                    found.append(((a, b, c), count, strength))
    return found


def mine_rare(
    baskets: list[list[str]],
    max_support: float = 0.005,
    min_count: int = 10,
    min_strength: float = 0.3,
    max_len: int = 2,
    top_k: int | None = None,
    drop_single_item_baskets: bool = True,
) -> MiningResult:
    if not 0 < max_support <= 1:
        raise InvalidRequest("max_support must be in (0, 1]")
    if min_count < 2:
        raise InvalidRequest("min_count must be at least 2")
    if not 0 < min_strength <= 1:
        raise InvalidRequest("min_strength must be in (0, 1]")
    if max_len not in (2, 3):
        raise InvalidRequest("max_len must be 2 (pairs) or 3 (pairs and triples)")

    if drop_single_item_baskets:
        baskets = [b for b in baskets if len(set(b)) >= 2]
    n = len(baskets)
    start = time.perf_counter()

    records = []   # (index tuple, count, strength)
    names: list[str] = []
    cnt = np.zeros(0, dtype=np.int64)
    if n:
        max_count = max_support * n
        X, names, cnt = _incidence(baskets, min_count)
        if X.shape[1] >= 2:
            co = sp.triu(X.T @ X, k=1).tocoo()
            r, c, v = co.row, co.col, co.data.astype(np.int64)
            strength = v / np.maximum(cnt[r], cnt[c])
            strong = (v >= min_count) & (strength >= min_strength)
            pr, pc, pv, ps = r[strong], c[strong], v[strong], strength[strong]

            rare = pv <= max_count
            records += [((a, b), int(k), float(s)) for a, b, k, s in
                        zip(pr[rare].tolist(), pc[rare].tolist(), pv[rare].tolist(), ps[rare].tolist())]
            if max_len == 3:
                records += _triples(X, cnt, pr, pc, min_count, min_strength, max_count)

    if len(records) > C.MAX_RARE_RESULTS:
        raise LimitExceeded(
            f"{len(records):,} itemsets match, more than the limit of {C.MAX_RARE_RESULTS:,}. "
            "Raise min_strength or min_count (or lower max_support)."
        )

    rows = []
    for idx, count, strength in records:
        lift = (n ** (len(idx) - 1)) * count / float(np.prod([cnt[i] for i in idx]))
        rows.append((tuple(names[i] for i in idx), count / n, count, len(idx), strength, lift))
    out = pd.DataFrame(rows, columns=["itemset", "support", "count", "length", "strength", "lift"])
    out = out.sort_values(["strength", "count"], ascending=[False, False]).reset_index(drop=True)
    total_found = len(out)
    if top_k is not None:
        out = out.head(top_k)

    runtime = time.perf_counter() - start
    return MiningResult(
        out, n, "rare", (min_count / n) if n else 0.0, runtime,
        params={"max_support": max_support, "min_count": min_count,
                "min_strength": min_strength, "max_len": max_len, "total_found": total_found},
    )
