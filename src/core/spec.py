"""The SPEC: what the language model fills in from the user's sentence.

It is looser than the engine's MiningRequest on purpose:
  * product and country names are FREE TEXT ("cakestand"), not exact data values
  * thresholds are a WORD (strictness), with optional explicit numbers
  * `kind` may be missing, if the sentence did not say what the user wants
validation.validate_spec() turns a Spec into an exact MiningRequest, or says what
is unclear. Same strictness (extra fields rejected) as the request.
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.core.request import DAYS


def _dedupe(values):
    seen, out = set(), []
    for v in values:
        if v.lower() not in seen:
            seen.add(v.lower())
            out.append(v)
    return out


class SpecFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weekdays: list[str] = Field(default_factory=list, description="Weekday names, e.g. ['Sunday'].")
    weekend: bool | None = Field(None, description="True: weekends only. False: weekdays only.")
    months: list[int] = Field(default_factory=list, description="Months 1-12.")
    countries: list[str] = Field(default_factory=list, description="Country names as the user said them.")
    include_products: list[str] = Field(default_factory=list,
                                        description="Products the user wants IN every result, as the user said them.")
    exclude_products: list[str] = Field(default_factory=list,
                                        description="Products or product words the user wants left out.")

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
        return sorted(set(v))

    @field_validator("countries", "include_products", "exclude_products")
    @classmethod
    def _text(cls, v):
        return _dedupe([x.strip() for x in v if x.strip()])


class Spec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["frequent", "rare", "high_utility"] | None = Field(
        None, description="frequent = often bought together; rare = uncommon but tightly linked; "
                          "high_utility = earns the most money. null if the user did not say.")
    strictness: Literal["loose", "normal", "strict"] = Field(
        "normal", description="How demanding the threshold is: loose = more results, strict = only the best.")
    # optional explicit numbers (override the strictness word for that setting)
    min_support: float | None = Field(None, gt=0, le=1)
    max_support: float | None = Field(None, gt=0, le=1)
    min_count: int | None = Field(None, ge=2)
    min_strength: float | None = Field(None, gt=0, le=1)
    min_utility: float | None = Field(None, gt=0, description="Money amount.")
    min_utility_pct: float | None = Field(None, gt=0, le=1, description="Share of total sales.")
    algorithm: str | None = None
    max_len: int | None = Field(None, ge=2, le=5, description="Largest combination size.")
    top_k: int = Field(20, ge=1, description="How many results to show.")
    filters: SpecFilters = Field(default_factory=SpecFilters)


def spec_json_schema() -> dict:
    """JSON schema handed to the language model in Step 8."""
    return Spec.model_json_schema()
