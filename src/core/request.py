"""The mining request: ONE description of what the user wants.

Everything above the mining layer (validation, LLM parser, API, UI) produces or
consumes this object. Strict by design: unknown fields are rejected, so a
mistake by the language model shows up as an error instead of being ignored.

Fields by kind of mining (others are ignored with a warning):
  frequent      min_support, algorithm ("fpgrowth" | "apriori")
  rare          max_support, min_count, min_strength
  high_utility  min_utility (money) OR min_utility_pct (share of total sales),
                algorithm ("fhm+" | "efim")
  all kinds     max_len, top_k, filters
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _dedupe(values):
    seen, out = set(), []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


class Filters(BaseModel):
    """Which baskets and products to use. All filters combine with AND."""
    model_config = ConfigDict(extra="forbid")

    weekdays: list[str] = Field(default_factory=list, description="Only baskets from these weekdays, e.g. ['Sunday'].")
    weekend: bool | None = Field(None, description="True: weekend baskets only. False: weekday baskets only.")
    months: list[int] = Field(default_factory=list, description="Only baskets from these months (1-12).")
    countries: list[str] = Field(default_factory=list, description="Only baskets from these countries.")
    include_items: list[str] = Field(default_factory=list,
                                     description="Every result must contain ALL of these products.")
    exclude_items: list[str] = Field(default_factory=list,
                                     description="Remove these products from every basket before mining.")

    @field_validator("weekdays")
    @classmethod
    def _days(cls, v):
        out = []
        for d in v:
            name = d.strip().title()
            if name not in DAYS:
                raise ValueError(f"'{d}' is not a weekday name")
            out.append(name)
        return _dedupe(out)

    @field_validator("months")
    @classmethod
    def _months(cls, v):
        for m in v:
            if not 1 <= m <= 12:
                raise ValueError(f"month {m} must be between 1 and 12")
        return _dedupe(v)

    @field_validator("countries")
    @classmethod
    def _countries(cls, v):
        return _dedupe([c.strip() for c in v if c.strip()])

    @field_validator("include_items", "exclude_items")
    @classmethod
    def _items(cls, v):
        return _dedupe([i.strip().upper() for i in v if i.strip()])


class MiningRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["frequent", "rare", "high_utility"]

    # frequent
    min_support: float = Field(0.01, gt=0, le=1, description="Share of baskets that must contain the combination.")
    # rare
    max_support: float = Field(0.005, gt=0, le=1, description="Rare means in at most this share of baskets.")
    min_count: int = Field(10, ge=2, description="...but in at least this many baskets (noise floor).")
    min_strength: float = Field(0.3, gt=0, le=1, description="How tightly the products belong together (all-confidence).")
    # high utility
    min_utility: float | None = Field(None, gt=0, description="Minimum money earned by the combination.")
    min_utility_pct: float | None = Field(None, gt=0, le=1, description="Minimum money as a share of total sales.")
    # algorithm choice (frequent: fpgrowth/apriori, high_utility: fhm+/efim)
    algorithm: str | None = None
    # all kinds
    max_len: int | None = Field(None, ge=2, le=5, description="Largest combination size. Rare/high-utility allow 2 or 3.")
    top_k: int = Field(20, ge=1, le=500, description="How many results to return.")
    filters: Filters = Field(default_factory=Filters)
