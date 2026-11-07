"""Cheap exact look-ups used to explain an EMPTY result ("nothing reaches 5%, but
the best pair reaches 4.45%"). Each takes about a second on the full data."""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.mining.rare import _incidence


def best_pair(baskets):
    """Most common pair of products: {'pair', 'count', 'support', 'n'} or None."""
    baskets = [b for b in baskets if len(set(b)) >= 2]
    n = len(baskets)
    if n == 0:
        return None
    X, names, _ = _incidence(baskets, 1)
    if X.shape[1] < 2:
        return None
    co = sp.triu(X.T @ X, k=1).tocoo()
    if co.nnz == 0:
        return None
    k = int(np.argmax(co.data))
    pair = tuple(sorted((names[co.row[k]], names[co.col[k]])))
    return {"pair": pair, "count": int(co.data[k]), "support": float(co.data[k]) / n, "n": n}


def best_pair_utility(df):
    """Pair that earned the most money: {'pair', 'utility'} or None.
    Pair utility = money from both products, summed over baskets containing both
    (computed with matrices; independent of Java)."""
    d = df.groupby(["basket_id", "item"], as_index=False)["utility"].sum()
    d = d[d.groupby("basket_id")["item"].transform("size") >= 2]
    if d.empty:
        return None
    b_codes, _ = pd.factorize(d["basket_id"])
    i_codes, i_names = pd.factorize(d["item"])
    cents = (d["utility"] * 100).round().astype("int64").to_numpy()
    shape = (int(b_codes.max()) + 1, len(i_names))
    U = sp.csr_matrix((cents, (b_codes, i_codes)), shape=shape)
    B = sp.csr_matrix((np.ones(len(d), dtype=np.int64), (b_codes, i_codes)), shape=shape)
    M = (U.T @ B).tocsr()
    T = sp.triu(M + M.T, k=1).tocoo()
    if T.nnz == 0:
        return None
    k = int(np.argmax(T.data))
    pair = tuple(sorted((i_names[T.row[k]], i_names[T.col[k]])))
    return {"pair": pair, "utility": float(T.data[k]) / 100.0}


def baskets_containing(df, items):
    """Number of baskets that contain ALL the given products."""
    sub = df[df["item"].isin(items)]
    if sub.empty:
        return 0
    return int((sub.groupby("basket_id")["item"].nunique() == len(set(items))).sum())
