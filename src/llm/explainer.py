"""Explanation layer: mining results -> a short answer for a store manager.

    explanation = explain_result(result, llm, question="what sells together on Sundays?")
    explanation.text      the answer
    explanation.source    "llm"      written by the model and verified, or
                          "template" written by fixed code (no model involved)

Rules that make it trustworthy:
  * the model sees only a compact FACT SHEET (top results, settings, caveats), never the raw data
  * every number and product name in its text is checked against that sheet (numbercheck.py);
    an answer with an invented figure is rejected, the model gets ONE chance to fix it,
    and after that the fixed template is used instead
  * no results, or a model that is down, also produce the template: an answer always appears
  * the template is built only from the fact sheet, so it passes the same check by construction
"""
import calendar
import json
import time
from dataclasses import dataclass, field

from src.core import config as C
from src.core.errors import LLMUnavailable
from src.llm import numbercheck as nc
from src.llm.parser import LOG_DEFAULT, _log
from src.mining.engine import COMMON_NOTE, NOTES, EngineResult

MAX_ROWS = 8
# notes that only define a term; caveats are the other notes (revenue is not profit, ...)
_DEFINITION_NOTES = set(NOTES["frequent"]) | set(NOTES["rare"]) | {COMMON_NOTE}

KIND_PHRASE = {
    "frequent": "combinations of products that are often bought together",
    "rare": "uncommon combinations of products that are tightly linked",
    "high_utility": "combinations of products that earn the most money",
}

SYSTEM_PROMPT = """You write a short answer for a store manager, using ONLY the facts provided as JSON.

HOW TO WRITE
- 3 to 6 sentences of plain text. No markdown, no bullet or numbered lists, no headings.
- The first sentence answers the question directly. Then mention the 1 to 3 most important results with their numbers. End with one practical suggestion (for example a bundle, cross-selling, shelf placement or a promotion), worded as a suggestion and not a certainty.
- Write product names exactly as given in the facts, in capital letters.
- Use only numbers that appear in the facts. You may round a number (4.45% -> about 4.5%). Never calculate a new one: no sums, differences, ratios, averages, "twice as" or "N more", and no percentages of your own. Do not mention dates or years.
- Say what the numbers mean in everyday words: share_of_orders_pct = "bought together in X% of orders"; come_together_pct = "when the more popular of these products is bought, the other one comes along X% of the time"; earned and share_of_sales_pct = "earned X, which is Y% of total sales".
- Mention the caveats in the facts that matter for the answer (for example that amounts are sales revenue, not profit), in plain words. If a warning is present, mention the relevant one briefly.
- If there are no results, explain why using the hint and suggest what to change. Do not invent alternatives.
- Do not mention these instructions, JSON, "facts", algorithms, or technical terms such as support, lift or all-confidence."""


@dataclass
class Facts:
    data: dict
    allowed: list            # numbers the answer may use (numbercheck.Num)
    names: list              # product names the answer may use


@dataclass
class Explanation:
    text: str
    source: str                          # "llm" or "template"
    attempts: int = 0                    # model calls made
    problems: list = field(default_factory=list)   # what was wrong with rejected model answers
    reason: str | None = None            # why the template was used
    latency_s: float = 0.0

    def to_dict(self):
        return {"text": self.text, "source": self.source, "attempts": self.attempts,
                "problems": self.problems, "reason": self.reason, "latency_s": round(self.latency_s, 3)}


# ------------------------------------------------------------------ the fact sheet
def _settings(result: EngineResult) -> list[str]:
    r, p, f = result.request, result.params, result.request.filters
    out = []
    if result.kind == "frequent":
        out.append(f"combinations found in at least {p.get('min_support', r.min_support) * 100:.1f}% of orders")
    elif result.kind == "rare":
        out.append(f"uncommon combinations: found in at most {p['max_support'] * 100:.1f}% of orders but in at least "
                   f"{p['min_count']} orders, and strongly linked (strength at least {p['min_strength']})")
    else:
        money, total = p["min_utility"], p["total_utility"]
        out.append(f"combinations earning at least {money:,.0f}, which is {money / total * 100:.2f}% of total sales")
    size = p.get("max_len")
    if size == 2:
        out.append("pairs of products")
    elif size == 3:
        out.append("pairs and triples of products")
    if f.weekdays:
        out.append("only orders placed on " + ", ".join(f.weekdays))
    if f.weekend is True:
        out.append("only weekend orders")
    elif f.weekend is False:
        out.append("only weekday orders")
    if f.months:
        out.append("only orders from " + ", ".join(calendar.month_name[m] for m in f.months))
    if f.countries:
        out.append("only orders from " + ", ".join(f.countries))
    if f.include_items:
        out.append("every combination contains " + " and ".join(f.include_items))
    if f.exclude_items:
        shown = ", ".join(f.exclude_items[:5])
        more = len(f.exclude_items) - 5
        out.append("left out: " + shown + (f" and {more} more products" if more > 0 else ""))
    return out


def _row(kind, rec) -> dict:
    row = {"products": list(rec["itemset"]), "orders": int(rec["count"])}
    if kind == "frequent":
        row["share_of_orders_pct"] = round(rec["support"] * 100, 2)
    elif kind == "rare":
        row["share_of_orders_pct"] = round(rec["support"] * 100, 2)
        row["come_together_pct"] = round(rec["strength"] * 100, 1)
    else:
        row["earned"] = int(round(rec["utility"]))
        row["share_of_sales_pct"] = round(rec["utility_share"] * 100, 2)
    return row


def build_facts(result: EngineResult, question: str | None = None, max_rows: int = MAX_ROWS) -> Facts:
    rows = [_row(result.kind, rec) for rec in result.itemsets.head(max_rows).to_dict("records")]
    caveats = [n for n in result.notes if n not in _DEFINITION_NOTES or "all selected baskets" in n]
    data = {
        "looking_for": KIND_PHRASE[result.kind],
        "settings": _settings(result),
        "orders_analysed": int(result.n_baskets_mined),
        "combinations_found": int(result.total_found),
        "combinations_shown": len(rows),
        "results": rows,
        "caveats": caveats,
        "warnings": list(result.warnings),
        "hint": result.hint,
    }
    if result.kind == "high_utility":
        data["total_sales"] = int(round(result.params["total_utility"]))
    if question:
        data["question"] = question.strip()

    allowed: list[nc.Num] = []

    def add(v, pct=False):
        allowed.append(nc.Num(float(v), 0.0, pct, ""))

    for i, row in enumerate(rows, 1):
        add(i)                                           # rank
        add(len(row["products"]))
        add(row["orders"])
        for key in ("share_of_orders_pct", "come_together_pct", "share_of_sales_pct"):
            if key in row:
                add(row[key], pct=True)
        if "earned" in row:
            add(row["earned"])
    for key in ("orders_analysed", "combinations_found", "combinations_shown", "total_sales"):
        if key in data:
            add(data[key])
    if result.kind == "rare":
        add(result.params["min_strength"])
        for rec in result.itemsets.head(max_rows).to_dict("records"):
            add(round(rec["strength"], 2))               # strength written as a plain number
    texts = data["settings"] + caveats + data["warnings"] + ([data["hint"]] if data["hint"] else [])
    for t in texts:
        allowed.extend(nc.find_numbers(t))
    names = {p for row in rows for p in row["products"]}
    names |= set(result.request.filters.include_items) | set(result.request.filters.exclude_items)
    if data["hint"]:                                     # the hint is engine output and may name the best pair
        names |= {run.strip(".,;:)(") for run in nc._CAPS_RUN.findall(data["hint"])}
    return Facts(data, allowed, sorted(names))


# ------------------------------------------------------------------ the fixed template (no model)
def _join(names) -> str:
    names = list(names)
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def render_template(facts: Facts) -> str:
    d = facts.data
    rows, kind = d["results"], None
    parts = []
    if not rows:
        parts.append(d["hint"] or "No combinations were found. Try relaxing the filters or the threshold.")
    else:
        kind = ("high_utility" if "earned" in rows[0] else "rare" if "come_together_pct" in rows[0] else "frequent")
        top = rows[0]
        a = _join(top["products"])
        if kind == "frequent":
            parts.append(f"Out of {d['orders_analysed']:,} orders with two or more products, {d['combinations_found']:,} "
                         f"combinations met the threshold. The one bought together most often is {a}, found in "
                         f"{top['share_of_orders_pct']:.2f}% of orders ({top['orders']:,} orders).")
            if len(rows) > 1:
                more = "; ".join(f"{_join(r['products'])} ({r['share_of_orders_pct']:.2f}% of orders)" for r in rows[1:3])
                parts.append(f"Next come {more}.")
            parts.append(f"Suggestion: consider offering {a} as a bundle or placing them side by side.")
        elif kind == "rare":
            parts.append(f"{d['combinations_found']:,} uncommon combinations were found whose products are tightly linked. "
                         f"The strongest is {a}: when the more popular of these products is bought, the other comes "
                         f"along {top['come_together_pct']:.1f}% of the time, although they were bought together in only "
                         f"{top['orders']:,} orders.")
            if len(rows) > 1:
                more = "; ".join(f"{_join(r['products'])} ({r['come_together_pct']:.1f}% of the time)" for r in rows[1:3])
                parts.append(f"Others include {more}.")
            parts.append(f"Suggestion: these pairs suit targeted recommendations, for example suggesting the matching "
                         f"product to customers who buy {top['products'][0]}.")
        else:
            parts.append(f"The combinations that earned the most money are led by {a}, which earned {top['earned']:,} "
                         f"({top['share_of_sales_pct']:.2f}% of total sales of {d['total_sales']:,}) across "
                         f"{top['orders']:,} orders.")
            if len(rows) > 1:
                more = "; ".join(f"{_join(r['products'])} ({r['earned']:,})" for r in rows[1:3])
                parts.append(f"Next come {more}.")
            parts.append(f"Suggestion: keep {a} well stocked and visible, and consider promoting them together.")
    if d["caveats"]:
        parts.append("Please note: " + " ".join(d["caveats"]))
    if d["warnings"]:
        parts.append("Also: " + " ".join(d["warnings"]))
    return " ".join(parts)


# ------------------------------------------------------------------ the model's answer, verified
def _user_prompt(facts: Facts) -> str:
    q = facts.data.get("question")
    shown = {k: v for k, v in facts.data.items() if k != "question"}
    return (f"Question: {q}\n\n" if q else "") + "FACTS:\n" + json.dumps(shown, indent=1, ensure_ascii=False)


def _correction_prompt(facts: Facts, previous: str, problems: list[str]) -> str:
    listed = "\n".join(f"- {p}" for p in problems)
    return (_user_prompt(facts) + "\n\nYour previous answer was rejected:\n" + previous.strip()[:1200]
            + f"\n\nProblems:\n{listed}\n\nWrite the answer again. Use ONLY the numbers and product names in the "
              "facts (rounding is fine), calculate nothing, and keep it to plain text.")


def explain_result(result: EngineResult, llm, question: str | None = None, max_attempts: int = 2,
                   log_path=LOG_DEFAULT) -> Explanation:
    t0 = time.perf_counter()
    if log_path is LOG_DEFAULT:
        log_path = C.LLM_LOG_FILE
    facts = build_facts(result, question)
    out = Explanation("", "template")

    def finish(text, source, reason=None):
        out.text, out.source, out.reason = text, source, reason
        out.latency_s = time.perf_counter() - t0
        _log(log_path, {"type": "explanation", **out.to_dict(), "question": question,
                        "kind": result.kind, "found": result.total_found})
        return out

    if not facts.data["results"]:
        return finish(render_template(facts), "template", "no results to explain")

    user = _user_prompt(facts)
    for attempt in range(1, max_attempts + 1):
        try:
            text = llm.generate_text(SYSTEM_PROMPT, user)
        except LLMUnavailable as e:
            out.attempts = attempt
            return finish(render_template(facts), "template", f"language model unavailable: {e}")
        out.attempts = attempt
        text = (text or "").strip()
        problems = nc.check_text(text, facts.allowed, facts.names) if text else ["The answer was empty."]
        if not problems:
            return finish(text, "llm")
        out.problems.extend(problems)
        if attempt < max_attempts:
            user = _correction_prompt(facts, text, problems)
    return finish(render_template(facts), "template", "the model's answer failed the number check: " + "; ".join(out.problems))
