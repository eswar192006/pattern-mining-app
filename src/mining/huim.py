"""High-utility itemset mining (HUIM) through the SPMF Java library.

Utility = quantity x unit price (money). An itemset's utility is the money its
items brought in, summed over the baskets that contain the whole itemset.

    result = mine_high_utility(df, min_utility_pct=0.001, max_len=2)

* `df` is the cleaned table from data_loader (needs basket_id, item, utility).
* Thresholds can be given as money (`min_utility`) or as a share of total sales
  (`min_utility_pct`, e.g. 0.001 = 0.1% of all sales).

Algorithms (both from SPMF, both exact):
  "fhm+" (default)  length-constrained variant of the HUI-Miner / FHM family.
  "efim"            EFIM. Fast, but it CANNOT limit itemset length.

Why length limits matter: utility has no simple pruning rule (adding an item can
raise or lower it), so when one giant wholesale order is worth 50,000, every large
subset of its ~100 items whose combined value reaches the threshold is "high utility",
and there are astronomically many. On this data EFIM wrote 22 million itemsets (and
filled the disk) at a threshold of 50,000. FHM+ with max_len=2 or 3 avoids this. A watchdog also kills the Java
process if it runs too long or writes too much.

SPMF needs integer utilities, so money is converted to cents internally.
"""
import math
import os
import shutil
import subprocess
import tempfile
import time

import numpy as np
import pandas as pd

from src.core import config as C
from src.core.errors import EngineUnavailable, InvalidRequest, LimitExceeded
from src.mining.frequent import MiningResult

ALGORITHMS = ("fhm+", "efim")
JAR_HELP = (
    f"SPMF jar not found at {C.SPMF_JAR}. Download spmf.jar from the official page "
    "https://www.philippe-fournier-viger.com/spmf/download.php and save it as tools/spmf.jar."
)


def find_java() -> str | None:
    """Find a usable java executable path, or None if unavailable."""
    if shutil.which("java") is not None:
        try:
            proc = subprocess.run(["java", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if proc.returncode == 0:
                return "java"
        except OSError:
            pass

    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        cand = os.path.join(java_home, "bin", "java")
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand

    for cand in ("/opt/homebrew/opt/openjdk/bin/java", "/usr/local/opt/openjdk/bin/java"):
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand

    return None


def java_available() -> bool:
    """Return True only if java binary exists AND can execute successfully (not macOS stub)."""
    return find_java() is not None


def _validate(min_utility, min_utility_pct, max_len, min_len, algorithm):
    if algorithm not in ALGORITHMS:
        raise InvalidRequest(f"algorithm must be one of {ALGORITHMS}")
    if (min_utility is None) == (min_utility_pct is None):
        raise InvalidRequest("Give exactly one of min_utility (money) or min_utility_pct (share of total sales).")
    if min_utility is not None and min_utility <= 0:
        raise InvalidRequest("min_utility must be positive")
    if min_utility_pct is not None and not 0 < min_utility_pct <= 1:
        raise InvalidRequest("min_utility_pct must be in (0, 1]")
    if max_len not in (2, 3):
        raise InvalidRequest("max_len must be 2 (pairs) or 3 (pairs and triples)")
    if not 1 <= min_len <= max_len:
        raise InvalidRequest("min_len must be between 1 and max_len")


def _write_spmf_input(df: pd.DataFrame, path: str, ids: dict) -> None:
    """One line per basket:  item ids : total utility : each item's utility (cents)."""
    d = df.assign(iid=df["item"].map(ids), cents=(df["utility"] * 100).round().astype("int64"))
    with open(path, "w") as f:
        for _, g in d.groupby("basket_id", sort=False):
            f.write(" ".join(map(str, g["iid"])) + ":" + str(int(g["cents"].sum())) + ":"
                    + " ".join(map(str, g["cents"])) + "\n")


def _run_spmf(cmd, out_path, log_path, timeout_s):
    """Run Java with a watchdog on time and output size."""
    limit = C.MAX_HUIM_OUTPUT_MB * 1024 * 1024
    start = time.perf_counter()
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            time.sleep(0.25)
            if time.perf_counter() - start > timeout_s:
                proc.kill()
                raise LimitExceeded(f"High-utility mining took longer than {timeout_s}s and was stopped. "
                                 "Raise min_utility or lower max_len.")
            if os.path.exists(out_path) and os.path.getsize(out_path) > limit:
                proc.kill()
                raise LimitExceeded(f"High-utility mining was producing more than {C.MAX_HUIM_OUTPUT_MB} MB of "
                                 "itemsets and was stopped. Raise min_utility or lower max_len.")
    elapsed = time.perf_counter() - start
    log_text = open(log_path, errors="replace").read()
    if proc.returncode != 0 or "error while trying" in log_text.lower() or not os.path.exists(out_path):
        raise EngineUnavailable("SPMF failed:\n" + log_text[-1500:])
    return elapsed


def _parse(out_path, names_by_id):
    found = []
    with open(out_path) as f:
        for line in f:
            if "#UTIL:" not in line:
                continue
            items, util = line.rsplit("#UTIL:", 1)
            ids = [int(t) for t in items.split()]
            found.append((tuple(sorted(names_by_id[i] for i in ids)), int(util)))
    return found


def mine_high_utility(
    df: pd.DataFrame,
    min_utility: float | None = None,
    min_utility_pct: float | None = 0.001,
    max_len: int = 2,
    min_len: int = 2,
    algorithm: str = "fhm+",
    top_k: int | None = None,
    drop_single_item_baskets: bool = True,
    java_memory: str | None = None,
    timeout_s: int | None = None,
) -> MiningResult:
    if min_utility is not None:
        min_utility_pct = None            # an explicit money amount wins over the default pct
    _validate(min_utility, min_utility_pct, max_len, min_len, algorithm)

    d = df[["basket_id", "item", "utility"]]
    d = d.groupby(["basket_id", "item"], as_index=False)["utility"].sum()
    if drop_single_item_baskets:
        d = d[d.groupby("basket_id")["item"].transform("size") >= 2]
    n = d["basket_id"].nunique()
    total = float(d["utility"].sum())
    if n == 0 or total <= 0:
        return MiningResult(pd.DataFrame(columns=["itemset", "utility", "count", "support", "length", "utility_share"]),
                            0, algorithm, 0.0, 0.0, params={"total_utility": 0.0})

    money = min_utility if min_utility is not None else min_utility_pct * total
    floor = C.HUIM_MIN_UTILITY_PCT_FLOOR[max_len]
    if money < floor * total:
        raise LimitExceeded(
            f"min_utility {money:,.0f} is below the safety floor {floor * total:,.0f} "
            f"({floor:.2%} of total sales {total:,.0f}) for max_len={max_len}: mining would take too long "
            "or return too many itemsets. Use a higher value or a smaller max_len.")

    if not C.SPMF_JAR.exists():
        raise EngineUnavailable(JAR_HELP)
    if not java_available():
        raise EngineUnavailable("Java was not found. Install a JDK (version 11 or newer) and try again.")

    start = time.perf_counter()
    names = sorted(d["item"].unique())
    ids = {nm: i + 1 for i, nm in enumerate(names)}
    names_by_id = {i: nm for nm, i in ids.items()}
    minutil_cents = max(1, math.ceil(money * 100 - 1e-6))

    with tempfile.TemporaryDirectory() as tmp:
        inp, out, log = (os.path.join(tmp, x) for x in ("in.txt", "out.txt", "log.txt"))
        _write_spmf_input(d, inp, ids)
        java_bin = find_java() or "java"
        jvm = [java_bin] + ([f"-Xmx{java_memory}"] if java_memory else []) + ["-jar", str(C.SPMF_JAR), "run"]
        if algorithm == "fhm+":
            cmd = jvm + ["FHM+", inp, out, str(minutil_cents), str(min_len), str(max_len)]
        else:
            cmd = jvm + ["EFIM", inp, out, str(minutil_cents)]
        java_s = _run_spmf(cmd, out, log, timeout_s or C.HUIM_TIMEOUT_S)
        found = _parse(out, names_by_id)

    found = [(s, u) for s, u in found if min_len <= len(s) <= max_len]
    if len(found) > C.MAX_RARE_RESULTS:
        raise LimitExceeded(f"{len(found):,} itemsets match; raise min_utility or lower max_len.")

    # basket counts for the results (how many baskets contain the whole itemset)
    tid: dict[str, np.ndarray] = {}
    basket_index = {b: j for j, b in enumerate(d["basket_id"].unique())}
    rows_idx = d["basket_id"].map(basket_index).to_numpy()
    by_item = d.assign(r=rows_idx).groupby("item")["r"].apply(lambda s: s.to_numpy())

    def tidset(item):
        a = tid.get(item)
        if a is None:
            a = np.zeros(n, dtype=bool)
            a[by_item[item]] = True
            tid[item] = a
        return a

    rows = []
    for itemset, cents in found:
        mask = tidset(itemset[0])
        for it in itemset[1:]:
            mask = mask & tidset(it)
        count = int(np.count_nonzero(mask))
        rows.append((itemset, cents / 100.0, count, count / n, len(itemset), (cents / 100.0) / total))
    out_df = pd.DataFrame(rows, columns=["itemset", "utility", "count", "support", "length", "utility_share"])
    out_df = out_df.sort_values(["utility", "count"], ascending=[False, False]).reset_index(drop=True)
    total_found = len(out_df)
    if top_k is not None:
        out_df = out_df.head(top_k)

    return MiningResult(
        out_df, n, algorithm, money / total, time.perf_counter() - start,
        params={"min_utility": money, "total_utility": total, "max_len": max_len, "min_len": min_len,
                "total_found": total_found, "java_s": java_s},
    )
