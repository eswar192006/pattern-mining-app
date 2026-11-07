"""Try the unified engine: describe what you want as JSON.

    python -m src.demo_engine '{"kind":"frequent","min_support":0.02}'
    python -m src.demo_engine '{"kind":"frequent","min_support":0.01,"filters":{"weekdays":["Sunday"]}}'
    python -m src.demo_engine '{"kind":"rare","max_len":3}'
    python -m src.demo_engine '{"kind":"high_utility","min_utility":10000,"filters":{"include_items":["REGENCY CAKESTAND 3 TIER"]}}'

(high_utility needs Java and tools/spmf.jar.) With no argument, runs the first example.
"""
import json
import sys

from src.core.errors import MiningError
from src.mining.engine import run_mining

METRICS = {"frequent": ["support", "count"], "rare": ["strength", "count"], "high_utility": ["utility", "count"]}

def print_result(r):
    t = r.timings
    print(f"{r.kind} | baskets selected={r.n_baskets_selected} mined={r.n_baskets_mined} | "
          f"found={r.total_found} shown={len(r.itemsets)} | {t['total_s']:.1f}s total")
    for label, n in r.filter_trace:
        print(f"  filter: {label:<40} -> {n} baskets")
    for w in r.warnings:
        print("WARNING:", w)
    if r.hint:
        print("HINT:", r.hint)
    print()
    cols = METRICS[r.kind]
    for _, row in r.itemsets.iterrows():
        vals = "  ".join(f"{c}={row[c]:.4g}" for c in cols)
        print(f"{int(row['rank']):>3}. {vals}   " + "  +  ".join(row["itemset"]))


if __name__ == "__main__":
    raw = sys.argv[1] if len(sys.argv) > 1 else '{"kind":"frequent","min_support":0.02}'
    try:
        result = run_mining(json.loads(raw))
    except json.JSONDecodeError as e:
        sys.exit(f"That is not valid JSON ({e}). Wrap the whole request in single quotes.")
    except MiningError as e:
        sys.exit(f"Could not run: {e}")
    print_result(result)
