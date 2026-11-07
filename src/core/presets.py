"""Threshold presets: 'loose' / 'normal' / 'strict' for each kind of mining.

The language model picks a word; THIS file decides the number. That is more
reliable than asking an LLM to invent a threshold like 0.013.

Values were measured on the Online Retail data (pairs; list sizes shown):
  frequent      min_support      0.7% -> ~6,000   1% -> ~1,600   2% -> ~130
  rare          min_strength     0.2  -> ~11,900  0.3 -> ~2,000  0.5 -> ~140
  high_utility  min_utility_pct  0.05% -> ~7,200  0.1% -> ~1,800 0.2% -> ~340  (of total sales)
Support and utility share are fractions, so they adapt when filters shrink the data.
"""

STRICTNESS = ("loose", "normal", "strict")

PRESETS = {
    "frequent": {"loose": {"min_support": 0.007}, "normal": {"min_support": 0.01}, "strict": {"min_support": 0.02}},
    "rare": {"loose": {"min_strength": 0.2}, "normal": {"min_strength": 0.3}, "strict": {"min_strength": 0.5}},
    "high_utility": {"loose": {"min_utility_pct": 0.0005}, "normal": {"min_utility_pct": 0.001},
                     "strict": {"min_utility_pct": 0.002}},
}

KIND_DESCRIPTIONS = {
    "frequent": "products that are often bought together",
    "rare": "uncommon combinations whose products are tightly linked",
    "high_utility": "combinations that earn the most money",
}


def preset_values(kind: str, strictness: str) -> dict:
    return dict(PRESETS[kind][strictness])


def describe(kind: str, strictness: str) -> str:
    v = PRESETS[kind][strictness]
    if kind == "frequent":
        return f"combinations found in at least {v['min_support']:.1%} of baskets"
    if kind == "rare":
        return f"rare combinations with strength at least {v['min_strength']}"
    return f"combinations earning at least {v['min_utility_pct']:.2%} of total sales"
