"""Plain-words presentation of results: no Streamlit here, so it is easy to test.

Turns technical fields (min_support, strength, utility_share...) into the words a store manager uses,
and decides which numbers to show for each kind of pattern.
"""
import calendar

KINDS = {
    "frequent": {"title": "Often bought together", "icon": "🛒", "accent": "#7c5cff",
                 "blurb": "Products that customers keep putting in the same basket.",
                 "metric": "Share of orders", "unit": "%"},
    "rare": {"title": "Hidden gems", "icon": "💎", "accent": "#22d3ee",
             "blurb": "Uncommon pairings whose products are tightly linked: when one is bought, the other follows.",
             "metric": "Linked strength", "unit": "%"},
    "high_utility": {"title": "Money makers", "icon": "💰", "accent": "#ffb454",
                     "blurb": "Combinations that bring in the most sales revenue.",
                     "metric": "Sales earned", "unit": ""},
}

EXAMPLES = {
    "frequent": ["Which products are usually bought together?",
                 "What do customers buy together on Sundays?",
                 "What sells together with the regency cakestand?"],
    "rare": ["Any unusual pairs that rarely sell but always go together?",
             "Show me lots of hidden gems in weekend orders."],
    "high_utility": ["Which combinations earn the most money?",
                     "Biggest earners in December, but leave out anything with bunting.",
                     "Show me the most profitable combinations in Germany, only the very best."],
}

# (button text, message sent). The model understands these as edits of the current request.
REFINEMENTS = {
    "frequent": [("Only Sundays", "only on Sundays"), ("Stricter", "make it stricter"), ("Show more", "show me more results"),
                 ("By money instead", "now show me what earns the most money"), ("Unusual pairs", "now show me the unusual hidden pairs")],
    "rare": [("Only Sundays", "only on Sundays"), ("Stricter", "make it stricter"), ("Show more", "show me more results"),
             ("Often together instead", "now show me what is bought together most often"),
             ("By money instead", "now show me what earns the most money")],
    "high_utility": [("Only Sundays", "only on Sundays"), ("Stricter", "make it stricter"), ("Show more", "show me more results"),
                     ("Often together instead", "now show me what is bought together most often"),
                     ("Unusual pairs", "now show me the unusual hidden pairs")],
}


def kind_info(kind: str) -> dict:
    return KINDS.get(kind, {"title": "Patterns", "icon": "✨", "accent": "#7c5cff", "blurb": "", "metric": "Value", "unit": ""})


def short(name: str, n: int = 24) -> str:
    return name if len(name) <= n else name[: n - 1].rstrip() + "…"


def combo_name(row: dict, sep: str = " + ", n: int | None = None) -> str:
    text = sep.join(row["itemset"])
    return text if n is None else short(text, n)


def combo_lines(row: dict, n: int = 30) -> str:
    """Each product on its own line (for chart labels where both names must stay readable)."""
    return "<br>+ ".join(short(p, n) for p in row["itemset"])


def primary_value(kind: str, row: dict) -> float:
    """The single number that ranks results of this kind (and sets bar heights)."""
    if kind == "frequent":
        return row["support"] * 100
    if kind == "rare":
        return row["strength"] * 100
    return row["utility"]


def format_primary(kind: str, value: float) -> str:
    return f"{value:,.0f}" if kind == "high_utility" else f"{value:.1f}%"


def plain_metrics(kind: str, row: dict) -> str:
    """One sentence about a result, in everyday words (used in hover text and cards)."""
    orders = f"{row['count']:,} orders"
    if kind == "frequent":
        return f"bought together in {row['support'] * 100:.2f}% of orders ({orders})"
    if kind == "rare":
        return f"the other comes along {row['strength'] * 100:.0f}% of the time ({orders})"
    return f"earned {row['utility']:,.0f} ({row['utility_share'] * 100:.2f}% of sales, {orders})"


def _list(values) -> str:
    return ", ".join(str(v) for v in values)


def understood_chips(spec: dict | None) -> list[tuple[str, str]]:
    """What the system understood, as (label, value) chips, from the conversation's current spec
    (the user's own words, before exact names are resolved)."""
    if not spec:
        return []
    chips = []
    if spec.get("kind"):
        chips.append(("Looking for", kind_info(spec["kind"])["title"]))
    chips.append(("Strictness", spec.get("strictness", "normal").capitalize()))
    f = spec.get("filters", {}) or {}
    if f.get("weekdays"):
        chips.append(("Days", _list(f["weekdays"])))
    if f.get("weekend") is True:
        chips.append(("When", "Weekends"))
    elif f.get("weekend") is False:
        chips.append(("When", "Weekdays"))
    if f.get("months"):
        chips.append(("Months", _list(calendar.month_name[m] for m in f["months"])))
    if f.get("countries"):
        chips.append(("Countries", _list(f["countries"])))
    if f.get("include_products"):
        chips.append(("Must include", _list(f["include_products"])))
    if f.get("exclude_products"):
        chips.append(("Leave out", _list(f["exclude_products"])))
    if spec.get("max_len"):
        chips.append(("Size", "Pairs" if spec["max_len"] == 2 else f"Up to {spec['max_len']} products"))
    if spec.get("top_k") and spec["top_k"] != 20:
        chips.append(("Show", f"Top {spec['top_k']}"))
    for key, label, fmt in (("min_support", "Min. share of orders", lambda v: f"{v * 100:.1f}%"),
                            ("min_strength", "Min. strength", lambda v: f"{v:g}"),
                            ("min_utility", "Min. earnings", lambda v: f"{v:,.0f}"),
                            ("min_utility_pct", "Min. share of sales", lambda v: f"{v * 100:.2f}%")):
        if spec.get(key) is not None:
            chips.append((label, fmt(spec[key])))
    return chips


def timing_summary(timings: dict) -> str:
    parts = []
    for key, label in (("language_s", "understanding"), ("mining_s", "exact counting"), ("explain_s", "writing")):
        if key in timings:
            parts.append(f"{label} {timings[key]:.1f}s")
    return " · ".join(parts)


def friendly_error(error: dict | None) -> tuple[str, str] | None:
    """-> (headline, detail) for an error from the API, or None. The server's own message is used
    when it has one: it already names the setting to change."""
    if not error:
        return None
    headlines = {
        "limit_exceeded": "That request is too heavy to run",
        "engine_unavailable": "A needed component is missing on the server",
        "server_busy": "The server is busy right now",
        "rate_limited": "Too many requests",
    }
    return headlines.get(error.get("code"), "Something went wrong"), error.get("message", "")
