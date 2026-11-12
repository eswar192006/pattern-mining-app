"""Unified mining engine:  run_mining(request) -> EngineResult.

One entry point for all three kinds of mining (frequent, rare, high_utility).
It applies the filters, calls the right miner, applies the 'include' filter,
shortens the list, and explains empty results. Safety limits stay inside the
miners; they surface here as typed errors (InvalidRequest / LimitExceeded /
EngineUnavailable), all subclasses of MiningError.

Filter meanings (all combine with AND):
  weekdays / weekend / months / countries   keep only those baskets
  exclude_items   remove these products from every basket BEFORE mining
  include_items   keep only results that contain ALL these products. Support,
                  strength and utility are still measured over all selected
                  baskets, not only the ones containing the included products.
"""
import difflib
import functools
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pydantic import ValidationError

from src.core import config as C
from src.core.errors import InvalidRequest, LimitExceeded
from src.core.request import MiningRequest
from src.mining import hints
from src.mining.data_loader import load_clean, to_baskets
from src.mining.frequent import mine_frequent
from src.mining.huim import mine_high_utility
from src.mining.rare import mine_rare

COMMON_FIELDS = {"kind", "max_len", "top_k", "filters"}
RELEVANT_FIELDS = {
    "frequent": {"min_support", "algorithm"},
    "rare": {"max_support", "min_count", "min_strength"},
    "high_utility": {"min_utility", "min_utility_pct", "algorithm"},
}
ALGORITHMS = {"frequent": ("fpgrowth", "apriori"), "high_utility": ("fhm+", "efim")}
DEFAULT_ALGORITHM = {"frequent": "fpgrowth", "high_utility": "fhm+"}

NOTES = {
    "frequent": ["Support is the share of baskets that contain every product in the combination."],
    "rare": ["Rare means a combination is uncommon. Strength is the share of baskets containing the most "
             "popular product of the combination that also contain the rest."],
    "high_utility": ["Utility is sales revenue (quantity x price), not profit, because the data has no cost information.",
                     "A combination containing a very high-selling product ranks high mostly because of that product."],
}
COMMON_NOTE = "Orders with a single product are ignored, because they cannot contain a combination."
EMPTY_COLUMNS = ["rank", "itemset", "length", "count", "support"]


@functools.lru_cache(maxsize=1)
def get_data() -> pd.DataFrame:
    """The cleaned dataset, loaded once per process."""
    return load_clean()


@dataclass
class EngineResult:
    kind: str
    request: MiningRequest
    itemsets: pd.DataFrame
    n_baskets_selected: int      # baskets left after filters (any size)
    n_baskets_mined: int         # baskets actually mined (2+ products)
    total_found: int             # results before top_k
    warnings: list = field(default_factory=list)   # about THIS request
    notes: list = field(default_factory=list)      # standing caveats for this kind
    hint: str | None = None                        # what to try when nothing is found
    filter_trace: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON-friendly version (used by the API in Step 11)."""
        rows = [{k: _native(v) for k, v in rec.items()} for rec in self.itemsets.to_dict("records")]
        return {
            "kind": self.kind, "request": self.request.model_dump(),
            "n_baskets_selected": self.n_baskets_selected, "n_baskets_mined": self.n_baskets_mined,
            "total_found": self.total_found, "shown": len(rows), "itemsets": rows,
            "warnings": self.warnings, "notes": self.notes, "hint": self.hint,
            "filter_trace": [[a, int(b)] for a, b in self.filter_trace],
            "params": {k: _native(v) for k, v in self.params.items()},
            "timings": {k: round(float(v), 3) for k, v in self.timings.items()},
        }


def _native(v):
    if isinstance(v, (list, tuple)):
        return [_native(x) for x in v]
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and v != v:
        return None
    return v


def _coerce(request) -> MiningRequest:
    if isinstance(request, MiningRequest):
        return request
    try:
        return MiningRequest.model_validate(request)
    except ValidationError as e:
        msg = "; ".join(f"{'.'.join(str(p) for p in err['loc']) or 'request'}: {err['msg']}" for err in e.errors())
        raise InvalidRequest(f"Invalid request: {msg}") from e


def _suggest(name, universe, what):
    near = [i for i in universe if name in i][:5] or difflib.get_close_matches(name, universe, n=3, cutoff=0.6)
    hint = f" Did you mean: {', '.join(near)}?" if near else ""
    return InvalidRequest(f"Unknown {what} '{name}'.{hint}")


def _check_kind_settings(req):
    """Resolve algorithm and max_len for this kind; warn about settings that do not apply."""
    kind, warnings = req.kind, []
    ignored = sorted(req.model_fields_set - COMMON_FIELDS - RELEVANT_FIELDS[kind])
    if ignored:
        warnings.append(f"Ignored settings that do not apply to {kind} mining: {', '.join(ignored)}.")

    algo = None
    if kind in ALGORITHMS:
        algo = req.algorithm or DEFAULT_ALGORITHM[kind]
        if algo not in ALGORITHMS[kind]:
            raise InvalidRequest(f"Algorithm '{algo}' is not available for {kind} mining; "
                                 f"choose one of {', '.join(ALGORITHMS[kind])}.")
    if kind == "high_utility" and req.min_utility is not None and req.min_utility_pct is not None:
        raise InvalidRequest("Give either min_utility (money) or min_utility_pct (share of sales), not both.")

    max_len = req.max_len
    if kind in ("rare", "high_utility"):
        max_len = 2 if max_len is None else max_len
        if max_len not in (2, 3):
            raise InvalidRequest(f"{kind} mining supports combinations of 2 or 3 products (max_len={max_len}).")
    return algo, max_len, warnings


def _apply_basket_filters(df, f, warnings):
    trace = [("all baskets", int(df["basket_id"].nunique()))]

    def step(label, mask):
        nonlocal df
        df = df[mask]
        trace.append((label, int(df["basket_id"].nunique())))

    present = set(df["weekday"].unique())
    if f.weekdays:
        missing = [d for d in f.weekdays if d not in present]
        if missing:
            warnings.append(f"This dataset has no orders on {', '.join(missing)}.")
        step(f"weekday in {f.weekdays}", df["weekday"].isin(f.weekdays))
    if f.weekend is not None:
        wk = set(df.loc[df["is_weekend"], "weekday"].unique())
        if f.weekend and wk != {"Saturday", "Sunday"}:
            have = ", ".join(sorted(wk)) or "no days"
            warnings.append(f"This dataset has orders on only {have} among the weekend days, "
                            f"so 'weekend' means {have}.")
        step("weekend only" if f.weekend else "weekdays only", df["is_weekend"] == f.weekend)
    if f.months:
        step(f"month in {f.months}", df["month"].isin(f.months))
    if f.countries:
        lookup = {c.lower(): c for c in df["country"].unique()}
        chosen = []
        for c in f.countries:
            if c.lower() not in lookup:
                raise _suggest(c, sorted(lookup.values()), "country")
            chosen.append(lookup[c.lower()])
        step(f"country in {chosen}", df["country"].isin(chosen))
    return df, trace


def selection_size(df: pd.DataFrame, filters) -> dict:
    """How much data a set of filters leaves, WITHOUT mining (used by validation).
    -> {'selected': baskets left, 'mineable': baskets with 2+ products,
        'total_utility': money in the mineable baskets, 'trace': filter steps}"""
    d, trace = _apply_basket_filters(df, filters, [])
    if filters.exclude_items:
        d = d[~d["item"].isin(filters.exclude_items)]
        trace.append((f"exclude products {filters.exclude_items}", int(d["basket_id"].nunique())))
    m = d[d.groupby("basket_id")["item"].transform("size") >= 2]
    return {"selected": int(d["basket_id"].nunique()), "mineable": int(m["basket_id"].nunique()),
            "total_utility": float(m["utility"].sum()), "trace": trace}


def _empty_hint(req, kind, df, baskets, res, found_before_include, include):
    """Say what to try when nothing was found."""
    if kind == "rare" and len(include) == 2:
        inc = ", ".join(include)
        needed = set(include)
        mined = [set(b) for b in baskets if len(set(b)) >= 2]
        pair_count = sum(needed <= basket for basket in mined)
        item_counts = [sum(item in basket for basket in mined) for item in include]
        strength = pair_count / max(item_counts) if pair_count and max(item_counts) else 0.0
        support = pair_count / res.n_baskets if res.n_baskets else 0.0
        if (pair_count >= req.min_count and support <= req.max_support
                and strength < req.min_strength):
            retry_strength = max(0.01, int(strength * 100) / 100)
            return (f"{inc} appear together in {pair_count:,} baskets ({support:.2%} of orders), but their "
                    f"link strength is {strength:.1%}, below your {req.min_strength:.0%} cutoff. "
                    f"Try setting minimum strength to {retry_strength:.0%} to include this pair.")

    if found_before_include and include:
        inc = ", ".join(include)
        have = hints.baskets_containing(df, include)
        return (f"{found_before_include:,} combinations met the threshold, but none contain {inc} "
                f"(which appears together in {have:,} baskets). Lower the threshold or drop the include filter.")
    if include:
        have = hints.baskets_containing(df, include)
        if have == 0:
            return f"No basket in this selection contains all of: {', '.join(include)}."

    if kind == "frequent":
        best = hints.best_pair(baskets)
        if best is None:
            return "No basket in this selection has two or more products."
        a, b = best["pair"]
        top = f"The most common pair ({a} + {b}) appears in {best['support']:.2%} of the {best['n']:,} baskets"
        if best["support"] < C.MIN_SUPPORT_FLOOR:
            return f"{top}, below the lowest allowed support ({C.MIN_SUPPORT_FLOOR:.1%}). Try removing filters."
        return (f"No combination appears in {req.min_support:.1%} of baskets. {top}; "
                f"try min_support {best['support']:.2%} or lower (lowest allowed: {C.MIN_SUPPORT_FLOOR:.1%}).")

    if kind == "rare":
        try:
            probe = mine_rare(baskets, max_support=req.max_support, min_count=req.min_count,
                              min_strength=0.05, max_len=2, top_k=1)
        except LimitExceeded:
            return (f"Nothing reaches strength {req.min_strength}, but many weaker combinations exist; "
                    f"try a lower min_strength (for example 0.2).")
        if len(probe.itemsets) == 0:
            return (f"No pair of products appears together in at least {req.min_count} baskets while staying "
                    f"under {req.max_support:.2%} support. Lower min_count or raise max_support.")
        row = probe.itemsets.iloc[0]
        a, b = row["itemset"]
        return (f"Nothing reaches strength {req.min_strength}. The strongest rare pair ({a} + {b}) has strength "
                f"{row['strength']:.2f} (in {int(row['count'])} baskets); try min_strength {row['strength']:.2f} or lower.")

    best = hints.best_pair_utility(df)
    if best is None:
        return "No basket in this selection has two or more products."
    a, b = best["pair"]
    money, total = res.params["min_utility"], res.params["total_utility"]
    floor_money = C.HUIM_MIN_UTILITY_PCT_FLOOR[res.params["max_len"]] * total
    if best["utility"] < floor_money:
        return (f"The best pair ({a} + {b}) earns {best['utility']:,.0f}, below the lowest allowed "
                f"threshold ({floor_money:,.0f}). Try removing filters.")
    return (f"No combination earns {money:,.0f}. The best pair ({a} + {b}) earns {best['utility']:,.0f}; "
            f"try min_utility {best['utility']:,.0f} or lower (lowest allowed: {floor_money:,.0f}).")


def run_mining(request, df: pd.DataFrame | None = None) -> EngineResult:
    t0 = time.perf_counter()
    req = _coerce(request)
    data = get_data() if df is None else df
    kind = req.kind
    algo, max_len, warnings = _check_kind_settings(req)
    notes = NOTES[kind] + [COMMON_NOTE]
    f = req.filters

    # --- products named in the request must exist
    universe = sorted(data["item"].unique())
    known = set(universe)
    for name in f.include_items + f.exclude_items:
        if name not in known:
            raise _suggest(name, universe, "product")
    both = set(f.include_items) & set(f.exclude_items)
    if both:
        raise InvalidRequest(f"The same product cannot be both included and excluded: {', '.join(sorted(both))}.")
    if max_len is not None and len(f.include_items) > max_len:
        raise InvalidRequest(f"{len(f.include_items)} products must appear together but combinations are limited "
                             f"to {max_len}. Raise max_len or include fewer products.")

    # --- filters
    d, trace = _apply_basket_filters(data, f, warnings)
    if f.exclude_items:
        d = d[~d["item"].isin(f.exclude_items)]
        trace.append((f"exclude products {f.exclude_items}", int(d["basket_id"].nunique())))
    t_filter = time.perf_counter()
    selected = int(d["basket_id"].nunique())

    base_params = {"max_len": max_len, "algorithm": algo}
    if selected == 0:
        empty_at = next((label for label, n in trace if n == 0), "the filters")
        return EngineResult(kind, req, pd.DataFrame(columns=EMPTY_COLUMNS), 0, 0, 0, warnings, notes,
                            f"No baskets remain after: {empty_at}. Relax the filters.", trace, base_params,
                            {"filter_s": t_filter - t0, "total_s": time.perf_counter() - t0})

    # --- mine
    baskets = to_baskets(d) if kind != "high_utility" else None
    if kind == "frequent":
        res = mine_frequent(baskets, min_support=req.min_support, algorithm=algo, min_len=2, max_len=max_len)
    elif kind == "rare":
        res = mine_rare(baskets, max_support=req.max_support, min_count=req.min_count,
                        min_strength=req.min_strength, max_len=max_len)
    else:
        res = mine_high_utility(d, min_utility=req.min_utility,
                                min_utility_pct=None if req.min_utility is not None else (req.min_utility_pct or 0.001),
                                max_len=max_len, algorithm=algo)

    # --- include filter, shorten, rank
    items = res.itemsets
    found_before_include = len(items)
    if f.include_items:
        need = set(f.include_items)
        items = items[items["itemset"].map(lambda s: need <= set(s))]
        notes.append("Support/strength/utility are measured over all selected baskets, "
                     "not only those containing the included products.")
    total_found = len(items)
    items = items.head(req.top_k).reset_index(drop=True)
    items.insert(0, "rank", range(1, len(items) + 1))
    t_mine = time.perf_counter()

    hint = None
    if total_found == 0:
        if baskets is None and kind != "high_utility":
            baskets = to_baskets(d)
        hint = _empty_hint(req, kind, d, baskets, res, found_before_include, f.include_items)
    t_end = time.perf_counter()

    params = {**base_params, **{k: v for k, v in res.params.items() if k != "total_found"}}
    if kind == "frequent":
        params["min_support"] = req.min_support
    return EngineResult(
        kind, req, items, selected, res.n_baskets, total_found, warnings, notes, hint, trace, params,
        {"filter_s": t_filter - t0, "mining_s": res.runtime_s, "hint_s": t_end - t_mine, "total_s": t_end - t0},
    )
