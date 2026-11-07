"""Try rare-but-meaningful mining on the real data.

    python -m src.demo_rare                       # defaults: pairs, strength >= 0.3
    python -m src.demo_rare 0.5 10 3 15           # min_strength, min_count, max_len, rows to show
"""
import sys
from src.mining.data_loader import load_clean, to_baskets
from src.mining.rare import mine_rare

if __name__ == "__main__":
    strength = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
    min_count = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    max_len = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    top = int(sys.argv[4]) if len(sys.argv) > 4 else 10

    r = mine_rare(to_baskets(load_clean()), min_strength=strength,
                  min_count=min_count, max_len=max_len)
    print(f"rare | max_support={r.params['max_support']} min_count={min_count} "
          f"min_strength={strength} max_len={max_len} | baskets mined={r.n_baskets} | "
          f"{r.params['total_found']} itemsets | {r.runtime_s:.2f}s\n")
    for _, row in r.itemsets.head(top).iterrows():
        print(f"strength {row['strength']:.2f}  in {row['count']:3d} baskets  " + "  +  ".join(row["itemset"]))
