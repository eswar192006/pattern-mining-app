"""Sample API replies for the interface tests (same shape as src/core/pipeline.py produces)."""
import copy

A, B, C3 = "PARTY BUNTING", "REGENCY CAKESTAND 3 TIER", "JUMBO BAG RED RETROSPOT"


def row(rank, items, kind="frequent", value=0.4, count=30):
    r = {"rank": rank, "itemset": list(items), "length": len(items), "count": count, "support": value}
    if kind == "rare":
        r["strength"] = value
    if kind == "high_utility":
        r["utility"] = value * 1000
        r["utility_share"] = value / 10
    return r


def result(kind="frequent", n=3, size=2, total=None):
    pool = [A, B, C3, "PAPER CHAIN KIT 50'S CHRISTMAS", "LUNCH BAG RED RETROSPOT", "SET OF 3 CAKE TINS PANTRY DESIGN"]
    rows = []
    for i in range(n):
        items = [pool[(i + k) % len(pool)] for k in range(size)]
        rows.append(row(i + 1, items, kind, value=round(0.5 - 0.03 * i, 3), count=30 - i))
    return {"kind": kind, "n_baskets_selected": 70, "n_baskets_mined": 65, "total_found": total or n, "shown": n,
            "itemsets": rows, "hint": None if rows else "Try a lower threshold.", "notes": ["Standing note."],
            "filter_trace": [["all orders", 70], ["weekday = Sunday", 30]], "params": {"min_support": 0.3},
            "timings": {"mining_s": 0.1}}


def reply(status="ok", kind="frequent", n=3, size=2, answer="The pair bought together most often is A and B.",
          source="llm", **over):
    body = {"session_id": "sid-1", "status": status, "turn_type": "new", "message": None,
            "request": {"kind": kind} if status == "ok" else None, "changes": [], "notes": [], "warnings": [],
            "questions": [], "errors": [], "result": None, "answer": None, "error": None,
            "session": {"current": {"kind": kind}, "turns": 1, "can_undo": False, "pending_questions": 0},
            "timings": {"language_s": 1.2, "total_s": 1.4}}
    if status == "ok":
        body["result"] = result(kind, n, size)
        body["timings"].update({"mining_s": 0.1, "explain_s": 0.9})
        if answer is not None:
            body["answer"] = {"text": answer, "source": source, "attempts": 1, "reason": None if source == "llm" else "numbers did not match"}
    body.update(copy.deepcopy(over))
    return body


HEALTH_OK = {"status": "ok", "data": {"ready": True, "baskets": 19960, "error": None},
             "llm": {"ready": True, "model": "gemini-3.6-flash", "error": None},
             "mining": {"java": True, "spmf_jar": True, "high_utility_available": True}, "sessions": 0}
CATALOG = {"baskets": 19960, "products": 3900, "date_range": ["2010-12-01", "2011-12-09"],
           "weekdays_with_orders": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Sunday"],
           "countries": ["United Kingdom"], "example_products": [A], "kinds": {}, "presets": {}, "limits": {}}
