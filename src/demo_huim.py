"""Try high-utility mining (most profitable combinations) on the real data.
Needs Java and tools/spmf.jar.

    python -m src.demo_huim                  # defaults: min utility 10000, pairs, top 10
    python -m src.demo_huim 5000 3 15        # min utility (money), max itemset size (2 or 3), rows
    python -m src.demo_huim 10000 2 5 efim   # optional 4th argument: efim (see README warning)
"""
import sys
from src.core.errors import MiningError
from src.mining.data_loader import load_clean
from src.mining.huim import mine_high_utility

if __name__ == "__main__":
    money = float(sys.argv[1]) if len(sys.argv) > 1 else 10000
    max_len = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    top = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    algo = sys.argv[4] if len(sys.argv) > 4 else "fhm+"

    try:
        r = mine_high_utility(load_clean(), min_utility=money, max_len=max_len, algorithm=algo)
    except MiningError as e:
        sys.exit(f"Could not run: {e}")
    total = r.params["total_utility"]
    print(f"{algo} | min_utility={money:,.0f} ({money/total:.3%} of {total:,.0f} total sales) | "
          f"max_len={max_len} | baskets mined={r.n_baskets} | {r.params['total_found']} itemsets | "
          f"{r.runtime_s:.1f}s\n")
    for _, row in r.itemsets.head(top).iterrows():
        print(f"{row['utility']:>10,.0f}  ({row['utility_share']:.2%} of sales, {row['count']} baskets)  "
              + "  +  ".join(row["itemset"]))
