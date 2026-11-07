"""Spec -> checked MiningRequest, or the questions that must be answered first.

    result = validate_spec(spec)         # spec: a Spec or a plain dict
    result.status   "ok" | "needs_clarification" | "invalid"
    result.request  the exact MiningRequest to run (only when status == "ok")
    result.questions  what to ask the user (needs_clarification)
    result.warnings   things changed or worth knowing (safety floors, ignored settings, ...)
    result.notes      how the sentence was interpreted ('cakestand' -> REGENCY CAKESTAND 3 TIER)
    result.errors     contradictions that cannot be fixed by guessing (invalid)

Principles:
  * never guess between several products: ask (but a keyword in 'exclude' may cover a group)
  * never silently change a number: values below a safety floor are raised, with a warning
  * report ALL problems at once, so the user answers everything in one go
"""
from dataclasses import dataclass, field

from pydantic import ValidationError

from src.core import config as C
from src.core import presets
from src.core.catalog import Catalog, get_catalog
from src.core.request import DAYS, Filters, MiningRequest
from src.core.spec import Spec
from src.mining.engine import RELEVANT_FIELDS, ALGORITHMS, get_data, selection_size

MAX_OPTIONS = 8              # most choices shown in a question
MAX_EXCLUDE_GROUP = 60       # a keyword in 'exclude' may cover up to this many products
OVERRIDE_FIELDS = ["min_support", "max_support", "min_count", "min_strength", "min_utility", "min_utility_pct"]

KIND_OPTIONS = {
    "Products often bought together": "frequent",
    "Unusual but tightly linked combinations": "rare",
    "Combinations that earn the most money": "high_utility",
}


@dataclass
class ClarifyingQuestion:
    field: str
    question: str
    options: list = field(default_factory=list)

    def to_dict(self):
        return {"field": self.field, "question": self.question, "options": list(self.options)}


@dataclass
class ValidationResult:
    status: str
    request: MiningRequest | None = None
    questions: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    mineable_baskets: int | None = None

    def to_dict(self):
        return {
            "status": self.status,
            "request": self.request.model_dump() if self.request else None,
            "questions": [q.to_dict() for q in self.questions],
            "warnings": self.warnings, "notes": self.notes, "errors": self.errors,
            "mineable_baskets": self.mineable_baskets,
        }


def _not_found(what, text, similar):
    q = f"I could not find a {what} called '{text}'."
    return q + (" Did you mean one of these?" if similar else "")


def _resolve_include(cat, text, field_name, notes, questions):
    exact = cat.exact_product(text)
    if exact:
        if exact.lower() != text.lower():
            notes.append(f"'{text}' -> {exact}")
        return [exact]
    cands = cat.match_products(text)
    if len(cands) == 1:
        notes.append(f"'{text}' -> {cands[0]}")
        return cands
    if len(cands) > 1:
        questions.append(ClarifyingQuestion(
            field_name, f"'{text}' matches {len(cands)} products. Which one did you mean?", cands[:MAX_OPTIONS]))
    else:
        sim = cat.similar_products(text)
        questions.append(ClarifyingQuestion(field_name, _not_found("product", text, sim), sim))
    return []


def _resolve_exclude(cat, text, field_name, notes, questions):
    exact = cat.exact_product(text)
    if exact:
        if exact.lower() != text.lower():
            notes.append(f"'{text}' -> {exact}")
        return [exact]
    cands = cat.match_products(text)
    if 0 < len(cands) <= MAX_EXCLUDE_GROUP:
        shown = ", ".join(cands[:3]) + (", ..." if len(cands) > 3 else "")
        notes.append(f"'{text}' matched {len(cands)} product(s), all excluded: {shown}")
        return cands
    if len(cands) > MAX_EXCLUDE_GROUP:
        questions.append(ClarifyingQuestion(
            field_name, f"'{text}' matches {len(cands)} products, too many to leave out by one word. "
                        "Please name a more specific product.", cands[:MAX_OPTIONS]))
    else:
        sim = cat.similar_products(text)
        questions.append(ClarifyingQuestion(field_name, _not_found("product", text, sim), sim))
    return []


def validate_spec(spec, df=None, catalog: Catalog | None = None) -> ValidationResult:
    try:
        sp = spec if isinstance(spec, Spec) else Spec.model_validate(spec)
    except ValidationError as e:
        msg = "; ".join(f"{'.'.join(str(p) for p in err['loc']) or 'spec'}: {err['msg']}" for err in e.errors())
        return ValidationResult("invalid", errors=[f"Invalid spec: {msg}"])

    data = get_data() if df is None else df
    cat = catalog or (get_catalog() if df is None else Catalog(data))
    questions, warnings, notes, errors = [], [], [], []
    f = sp.filters

    # ---- 1. kind
    if sp.kind is None:
        questions.append(ClarifyingQuestion("kind", "What kind of patterns do you want to see?", list(KIND_OPTIONS)))

    # ---- 2. weekdays that have no data at all
    absent = [d for d in f.weekdays if d not in cat.weekdays_present]
    if f.weekdays and len(absent) == len(f.weekdays):
        present = [d for d in DAYS if d in cat.weekdays_present]
        questions.append(ClarifyingQuestion(
            "filters.weekdays", f"There are no orders on {', '.join(absent)} in this data. Which day do you want instead?",
            present))

    # ---- 3. products and countries -> exact values
    include, exclude, countries = [], [], []
    for i, name in enumerate(f.include_products):
        for item in _resolve_include(cat, name, f"filters.include_products[{i}]", notes, questions):
            if item not in include:
                include.append(item)
    for i, name in enumerate(f.exclude_products):
        for item in _resolve_exclude(cat, name, f"filters.exclude_products[{i}]", notes, questions):
            if item not in exclude:
                exclude.append(item)
    for i, name in enumerate(f.countries):
        resolved, sim = cat.match_country(name)
        if resolved:
            if resolved.lower() != name.lower():
                notes.append(f"'{name}' -> {resolved}")
            if resolved not in countries:
                countries.append(resolved)
        else:
            questions.append(ClarifyingQuestion(f"filters.countries[{i}]", _not_found("country", name, sim), sim))

    overlap = sorted(set(include) & set(exclude))
    if overlap:
        errors.append(f"The same product cannot be both included and excluded: {', '.join(overlap)}.")

    # ---- 4. how much data is left? (only possible once every name is resolved)
    sel = None
    if not errors and all(q.field == "kind" for q in questions):
        sel = selection_size(data, _filters_for(sp, include, exclude, countries))
        if sel["mineable"] == 0:
            empty_at = next((label for label, n in sel["trace"] if n == 0), "the filters")
            questions.append(ClarifyingQuestion(
                "filters", f"No baskets with two or more products remain after: {empty_at}. "
                           "Which filter should be relaxed?", [label for label, _ in sel["trace"][1:]]))

    if errors:
        return ValidationResult("invalid", None, questions, warnings, notes, errors)
    if questions:
        return ValidationResult("needs_clarification", None, questions, warnings, notes, errors)
    return _finish(sp, include, exclude, countries, sel, warnings, notes)


def _filters_for(sp, include, exclude, countries) -> Filters:
    f = sp.filters
    return Filters(weekdays=f.weekdays, weekend=f.weekend, months=f.months, countries=countries,
                   include_items=include, exclude_items=exclude)


def _finish(sp, include, exclude, countries, sel, warnings, notes) -> ValidationResult:
    kind = sp.kind
    errors, questions = [], []

    # ---- thresholds: preset, then explicit numbers, only those relevant to this kind
    values = presets.preset_values(kind, sp.strictness)
    explicit = {k: getattr(sp, k) for k in OVERRIDE_FIELDS if getattr(sp, k) is not None}
    ignored = sorted(k for k in explicit if k not in RELEVANT_FIELDS[kind])
    if ignored:
        warnings.append(f"Ignored settings that do not apply to {kind} mining: {', '.join(ignored)}.")
    explicit = {k: v for k, v in explicit.items() if k in RELEVANT_FIELDS[kind]}
    if kind == "high_utility":
        if "min_utility" in explicit and "min_utility_pct" in explicit:
            errors.append("Give either min_utility (money) or min_utility_pct (share of sales), not both.")
        elif "min_utility" in explicit:
            values.pop("min_utility_pct", None)
    values.update(explicit)
    shown = {k: v for k, v in values.items() if k not in explicit}
    if shown:
        notes.append(f"Strictness '{sp.strictness}' -> " + ", ".join(f"{k} = {v:g}" for k, v in shown.items())
                     + f" ({presets.describe(kind, sp.strictness)})")

    # ---- algorithm
    algorithm = sp.algorithm
    if algorithm is not None:
        if kind == "rare":
            warnings.append("Ignored settings that do not apply to rare mining: algorithm.")
            algorithm = None
        elif algorithm not in ALGORITHMS[kind]:
            errors.append(f"Algorithm '{algorithm}' is not available for {kind} mining; "
                          f"choose one of {', '.join(ALGORITHMS[kind])}.")
        elif algorithm == "efim":
            warnings.append("EFIM cannot limit combination size, so it may be stopped if it produces too much output.")

    # ---- max_len
    max_len = sp.max_len
    if kind in ("rare", "high_utility"):
        max_len = 2 if max_len is None else max_len
        if max_len > 3:
            warnings.append(f"{kind} mining supports combinations of 2 or 3 products; max_len lowered from {max_len} to 3.")
            max_len = 3
    if max_len is not None and len(include) > max_len:
        errors.append(f"{len(include)} products must appear together but combinations are limited to {max_len}. "
                      "Include fewer products or raise the size limit.")

    # ---- top_k
    top_k = sp.top_k
    if top_k > 500:
        warnings.append(f"top_k lowered from {top_k} to the maximum of 500.")
        top_k = 500

    # ---- how much data is left, and safety floors that depend on it
    filters = _filters_for(sp, include, exclude, countries)
    n, total = sel["mineable"], sel["total_utility"]

    if kind == "frequent":
        floor = C.MIN_SUPPORT_FLOOR
        if values["min_support"] < floor:
            warnings.append(f"min_support {values['min_support']:.2%} is below the lowest allowed value; "
                            f"raised to {floor:.1%} (lower values make mining too slow).")
            values["min_support"] = floor
        if n and values["min_support"] * n < 3:
            warnings.append(f"Only {n:,} baskets are selected, so a combination needs just "
                            f"{max(1, round(values['min_support'] * n))} basket(s) to count as frequent; "
                            "results may be coincidence.")

    elif kind == "rare":
        max_support = values.get("max_support", 0.005)
        min_count = values.get("min_count", 10)
        band_max = max_support * n
        if band_max < min_count:
            questions.append(ClarifyingQuestion(
                "kind",
                f"Only {n:,} baskets match your filters, so 'rare' (at most {max_support:.1%} of baskets, "
                f"about {int(band_max)}) cannot reach the minimum of {min_count} baskets. Nothing could qualify. "
                "What would you like to do?",
                ["Remove or widen the filters", f"Lower the minimum count to {max(2, int(band_max))} or less",
                 "Switch to frequent patterns"]))
        elif band_max < 2 * min_count:
            span = f"exactly {min_count}" if int(band_max) == min_count else f"{min_count} to {int(band_max)}"
            warnings.append(f"With {n:,} baskets selected, only combinations seen in {span} baskets count as rare, "
                            "so few or no results are expected.")

    else:
        floor = C.HUIM_MIN_UTILITY_PCT_FLOOR[max_len]
        current = values.get("min_utility_pct")
        if current is None and total > 0:
            current = values["min_utility"] / total
        if current is not None and current < floor:
            warnings.append(f"The minimum earnings you asked for is below the lowest allowed value "
                            f"({floor:.2%} of the {total:,.0f} total sales for combinations of {max_len}); raised to that value.")
            values.pop("min_utility", None)
            values["min_utility_pct"] = floor

    if errors or questions:
        return ValidationResult("invalid" if errors else "needs_clarification", None, questions, warnings, notes,
                                errors, n)

    request_fields = {"kind": kind, "top_k": top_k, "filters": filters, **values}
    if algorithm is not None:
        request_fields["algorithm"] = algorithm
    if max_len is not None and (sp.max_len is not None or kind in ("rare", "high_utility")):
        request_fields["max_len"] = max_len
    request = MiningRequest(**request_fields)
    return ValidationResult("ok", request, [], warnings, notes, [], n)
