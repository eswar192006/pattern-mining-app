"""What the page shows, decided in plain Python (no Streamlit), so it is easy to test.

ui/present.py has the words for patterns and chips; this file turns one API reply into the pieces of the
screen: the answer badge, the metric cards, the table rows, the status pills and the buttons.
"""
import csv
import io

from ui.present import REFINEMENTS, format_primary, kind_info, primary_value

# Replies whose message is the whole story: (headline, tone). Tones: info, warn, bad.
MESSAGE_STATUSES = {
    "out_of_scope": ("That is outside what I can look at", "info"),
    "reset": ("Fresh start", "info"),
    "nothing_to_undo": ("Nothing to undo", "info"),
    "nothing_to_run": ("Nothing to run yet", "info"),
    "failed": ("I could not understand that", "warn"),
    "invalid": ("That request cannot be run", "warn"),
    "llm_unavailable": ("The language model is not available", "bad"),
}


def answer_badge(answer: dict | None) -> tuple[str, str]:
    """(label, tone): who wrote the answer. The numbers inside are always checked against the data."""
    if not answer:
        return "Results only (no written answer)", "muted"
    if answer.get("source") == "llm":
        return "Written by the model · every number checked", "ok"
    return "Fixed template · the model's text was not used", "warn"


def answer_reason(answer: dict | None) -> str:
    """Why the template was used instead of the model, when it was."""
    if answer and answer.get("source") != "llm":
        return answer.get("reason") or ""
    return ""


def readiness_pills(health: dict | None) -> list[dict]:
    """The status pills: data, model, Java. state is ok / warn / bad."""
    if not health:
        return [{"label": "Server", "state": "bad", "detail": "not reachable"}]
    data, llm, mining = health.get("data", {}), health.get("llm", {}), health.get("mining", {})
    pills = [{"label": "Data", "state": "ok" if data.get("ready") else "bad",
              "detail": f"{data['baskets']:,} orders" if data.get("ready") and data.get("baskets") else (data.get("error") or "not prepared")}]
    pills.append({"label": "Model", "state": "ok" if llm.get("ready") else "bad",
                  "detail": llm.get("model") or "not configured"})
    java_ok = bool(mining.get("high_utility_available"))
    pills.append({"label": "Java", "state": "ok" if java_ok else "warn",
                  "detail": "money questions ready" if java_ok else "money questions unavailable"})
    return pills


def blocking_problem(health: dict | None) -> tuple[str, str] | None:
    """(headline, how to fix it) when chatting cannot work at all, else None."""
    if not health:
        return None
    data, llm = health.get("data", {}), health.get("llm", {})
    if not data.get("ready"):
        return ("The sales data is not prepared yet",
                (data.get("error") or "The cleaned data file is missing.")
                + "  Put the dataset in data/raw/ and run:  python -m src.prepare_data")
    if not llm.get("ready"):
        return ("The language model is not set up",
                (llm.get("error") or "The configured language model is unavailable.")
                + "  Then restart the server.")
    return None


def metric_cards(reply: dict) -> list[tuple[str, str, str]]:
    """(label, value, small caption) for the cards above the results."""
    result = reply.get("result") or {}
    timings = reply.get("timings") or {}
    cards = [("Orders analysed", f"{result.get('n_baskets_mined', 0):,}",
              f"of {result.get('n_baskets_selected', 0):,} that match (single-product orders skipped)"),
             ("Combinations found", f"{result.get('total_found', 0):,}", f"showing the top {result.get('shown', 0):,}")]
    total = timings.get("total_s")
    if total is not None:
        caption = " · ".join(f"{label} {timings[key]:.1f}s" for key, label in
                             (("language_s", "understanding"), ("mining_s", "counting"), ("explain_s", "writing"))
                             if key in timings)
        cards.append(("Time taken", f"{total:.1f}s", caption))
    return cards


def table_rows(result: dict | None) -> list[dict]:
    """The results as plain rows for a table (columns depend on the kind of pattern)."""
    rows = []
    kind = (result or {}).get("kind", "frequent")
    for r in (result or {}).get("itemsets", []):
        row = {"Rank": r.get("rank"), "Products": " + ".join(r.get("itemset", [])), "Size": len(r.get("itemset", []))}
        if kind == "frequent":
            row["Share of orders"] = format_primary("frequent", primary_value("frequent", r))
        elif kind == "rare":
            row["Linked strength"] = format_primary("rare", primary_value("rare", r))
            row["Share of orders"] = f"{r.get('support', 0) * 100:.2f}%"
        else:
            row["Sales earned"] = format_primary("high_utility", primary_value("high_utility", r))
            row["Share of sales"] = f"{r.get('utility_share', 0) * 100:.2f}%"
        row["Orders"] = r.get("count")
        rows.append(row)
    return rows


def table_csv(rows: list[dict]) -> bytes:
    """The table as CSV text for the download button (UTF-8 with a BOM so Excel shows symbols correctly)."""
    if not rows:
        return b""
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return b"\xef\xbb\xbf" + out.getvalue().encode("utf-8")


def refinement_buttons(kind: str | None) -> list[tuple[str, str]]:
    return REFINEMENTS.get(kind or "", REFINEMENTS["frequent"])


def reply_kind(reply: dict) -> str | None:
    result = reply.get("result") or {}
    return result.get("kind") or (reply.get("request") or {}).get("kind")


def has_results(reply: dict) -> bool:
    return bool((reply.get("result") or {}).get("itemsets"))


def headline(reply: dict) -> str:
    """One line that says what this reply is, for the top of its card."""
    kind = reply_kind(reply)
    if reply.get("status") == "ok" and kind:
        info = kind_info(kind)
        return f"{info['icon']}  {info['title']}"
    if reply.get("status") == "needs_clarification":
        return "A quick question first"
    return MESSAGE_STATUSES.get(reply.get("status"), ("", ""))[0]


def escape_dollars(text: str) -> str:
    """Streamlit's markdown treats $...$ as maths; show money signs as plain text."""
    return (text or "").replace("$", "\\$")
