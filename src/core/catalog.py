"""Facts about the dataset, and matching of free-text names to real values.

Used by validation (to resolve 'cakestand' to a real product, or ask which one)
and, in Step 8, to tell the language model what the data contains.
"""
import difflib
import functools
import re

import pandas as pd

from src.core.request import DAYS

COUNTRY_ALIASES = {
    "UK": "United Kingdom", "U.K.": "United Kingdom", "GB": "United Kingdom", "BRITAIN": "United Kingdom",
    "GREAT BRITAIN": "United Kingdom", "ENGLAND": "United Kingdom",
    "US": "USA", "U.S.": "USA", "AMERICA": "USA", "UNITED STATES": "USA",
    "IRELAND": "EIRE", "HOLLAND": "Netherlands", "UAE": "United Arab Emirates",
}


def _tokens(text: str) -> list[str]:
    """Upper-case words, punctuation removed, plural 'S' stripped ('BAGS' -> 'BAG')."""
    words = re.sub(r"[^A-Z0-9 ]", " ", text.upper()).split()
    return [w[:-1] if len(w) > 3 and w.endswith("S") else w for w in words]


class Catalog:
    def __init__(self, df: pd.DataFrame):
        stats = df.groupby("item").agg(baskets=("basket_id", "nunique"), revenue=("utility", "sum"))
        self.items = stats.sort_values(["baskets", "revenue"], ascending=False)
        self.item_names = list(self.items.index)
        self._tok = {n: frozenset(_tokens(n)) for n in self.item_names}
        self._norm = {n: " ".join(_tokens(n)) for n in self.item_names}
        self._by_norm = {}
        for n in self.item_names:                       # most popular wins a normalised-name tie
            self._by_norm.setdefault(self._norm[n], n)

        per_basket = df.drop_duplicates("basket_id")
        self.countries = per_basket.groupby("country").size().sort_values(ascending=False)
        self.n_baskets = int(df["basket_id"].nunique())
        self.total_utility = float(df["utility"].sum())
        self.weekdays_present = set(df["weekday"].unique()) if "weekday" in df else set()
        self.months_present = sorted(int(m) for m in df["month"].unique()) if "month" in df else []
        self.date_range = ((str(df["invoice_date"].min())[:10], str(df["invoice_date"].max())[:10])
                           if "invoice_date" in df else None)

    # ---- products
    def exact_product(self, text: str) -> str | None:
        """Same product ignoring case, punctuation and plural 's'."""
        return self._by_norm.get(" ".join(_tokens(text)))

    def match_products(self, text: str) -> list[str]:
        """Products containing every word of `text`, most popular first."""
        q = frozenset(_tokens(text))
        if not q:
            return []
        return [n for n in self.item_names if q <= self._tok[n]]

    def similar_products(self, text: str, n: int = 5) -> list[str]:
        close = difflib.get_close_matches(" ".join(_tokens(text)), list(self._by_norm), n=n, cutoff=0.6)
        return [self._by_norm[c] for c in close]

    # ---- countries
    def match_country(self, text: str):
        """-> (resolved name or None, suggestions). Handles case, aliases, unique partial matches."""
        names = list(self.countries.index)
        lower = {c.lower(): c for c in names}
        t = text.strip()
        if t.lower() in lower:
            return lower[t.lower()], []
        alias = COUNTRY_ALIASES.get(t.upper())
        if alias and alias.lower() in lower:
            return lower[alias.lower()], []
        partial = [c for c in names if t.lower() in c.lower()]
        if len(partial) == 1:
            return partial[0], []
        if partial:
            return None, partial[:8]
        return None, difflib.get_close_matches(t, names, n=3, cutoff=0.6)

    # ---- for the language model prompt (Step 8)
    def prompt_summary(self, n_examples: int = 8, n_countries: int = 8) -> str:
        days = [d for d in DAYS if d in self.weekdays_present]
        missing = [d for d in DAYS if d not in self.weekdays_present]
        lines = [f"Dataset: orders from a UK online gift retailer ({self.n_baskets:,} baskets, "
                 f"{len(self.item_names):,} products)."]
        if self.date_range:
            lines.append(f"Dates: {self.date_range[0]} to {self.date_range[1]}.")
        lines.append(f"Order days present: {', '.join(days)}." + (f" No orders on {', '.join(missing)}." if missing else ""))
        lines.append("Top countries: " + ", ".join(self.countries.index[:n_countries]) + ".")
        lines.append("Most popular products: " + "; ".join(self.item_names[:n_examples]) + ".")
        return "\n".join(lines)


@functools.lru_cache(maxsize=1)
def get_catalog() -> Catalog:
    from src.mining.engine import get_data
    return Catalog(get_data())
