"""Try frequent mining on the real data.

    python -m src.demo_frequent                      # defaults
    python -m src.demo_frequent 0.01 apriori 15      # support, algorithm, rows
"""
import sys
from src.mining.data_loader import load_clean, to_baskets
from src.mining.frequent import mine_frequent

if __name__ == "__main__":
    support = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
    algo = sys.argv[2] if len(sys.argv) > 2 else "fpgrowth"
    top = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    r = mine_frequent(to_baskets(load_clean()), min_support=support, algorithm=algo)
    print(f"{algo} | min_support={support} | baskets mined={r.n_baskets} | "
          f"{len(r.itemsets)} itemsets (2+ items) | {r.runtime_s:.2f}s\n")
    for _, row in r.itemsets.head(top).iterrows():
        print(f"{row['support']:.3%}  ({row['count']} baskets)  " + "  +  ".join(row["itemset"]))
