"""Try the validator: describe what the user wants as a SPEC (words, not exact values).

    python -m src.demo_validate '{"kind":"frequent","strictness":"strict"}'
    python -m src.demo_validate '{"filters":{"include_products":["cakestand"]}}'
    python -m src.demo_validate '{"kind":"high_utility","filters":{"exclude_products":["bunting"],"countries":["uk"]}}' --run

Without --run it only checks the spec. With --run, a valid spec is also mined
(high_utility needs Java and tools/spmf.jar).
"""
import json
import sys

from src.core.errors import MiningError
from src.core.validation import validate_spec
from src.demo_engine import print_result
from src.mining.engine import run_mining

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--run"]
    raw = args[0] if args else '{"kind":"frequent","strictness":"strict"}'
    try:
        r = validate_spec(json.loads(raw))
    except json.JSONDecodeError as e:
        sys.exit(f"That is not valid JSON ({e}). Wrap the whole spec in single quotes.")

    print(f"STATUS: {r.status}" + (f"   ({r.mineable_baskets:,} baskets can be mined)" if r.mineable_baskets else ""))
    for n in r.notes:
        print("  understood:", n)
    for w in r.warnings:
        print("  WARNING:", w)
    for e in r.errors:
        print("  ERROR:", e)
    for q in r.questions:
        print(f"  QUESTION ({q.field}): {q.question}")
        for o in q.options:
            print(f"      - {o}")
    if r.request:
        print("\nREQUEST:", json.dumps(r.request.model_dump(exclude_defaults=True), default=str))
        if "--run" in sys.argv:
            print()
            try:
                print_result(run_mining(r.request))
            except MiningError as e:
                sys.exit(f"Could not run: {e}")
