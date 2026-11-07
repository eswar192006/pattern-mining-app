"""Applying a follow-up EDIT to the current spec, in plain code (no model involved).

    new_spec, notes, problems = apply_edit(current_spec, reply)
    lines = diff_specs(current_spec, new_spec)        # what changed, in words

Why code and not the model: the guarantee "everything the user did not mention stays as it was"
must not depend on a language model remembering to copy fields. The model only says what changes.

Rules
  * only the fields the model actually wrote (model_fields_set) are changed
  * list settings (weekdays, months, countries, products) are replaced as a whole: the model sees the
    current lists, so it writes the complete new list
  * `clear` removes a setting (back to its default). Unknown names are a problem, and so is setting and
    clearing the same thing
  * `fresh` starts from an empty spec
  * when the kind of pattern changes, explicit thresholds that do not apply to the new kind are dropped,
    with a note, instead of lingering as ignored settings
"""
from pydantic import ValidationError

from src.core.spec import Spec, SpecFilters
from src.llm.reply import FollowUpReply
from src.mining.engine import RELEVANT_FIELDS

THRESHOLD_FIELDS = ["min_support", "max_support", "min_count", "min_strength", "min_utility",
                    "min_utility_pct", "algorithm"]
TOP_LEVEL = [f for f in Spec.model_fields if f != "filters"]
FILTER_FIELDS = list(SpecFilters.model_fields)
CLEARABLE = TOP_LEVEL + ["filters"] + [f"filters.{f}" for f in FILTER_FIELDS]

LABELS = {
    "kind": "type of pattern", "strictness": "strictness", "top_k": "number of results", "max_len": "combination size",
    "min_support": "minimum support", "max_support": "maximum support", "min_count": "minimum orders",
    "min_strength": "minimum strength", "min_utility": "minimum earnings", "min_utility_pct": "minimum share of sales",
    "algorithm": "algorithm", "filters.weekdays": "weekdays", "filters.weekend": "weekend / weekdays",
    "filters.months": "months", "filters.countries": "countries",
    "filters.include_products": "must include", "filters.exclude_products": "leave out",
}


def _default(model, name):
    return model.model_fields[name].get_default(call_default_factory=True)


def apply_edit(current: Spec | None, reply: FollowUpReply):
    """-> (new Spec or None, notes, problems). `problems` is empty when the edit could be applied."""
    base = Spec() if (reply.fresh or current is None) else current
    data = base.model_dump()
    patch = reply.spec if reply.spec is not None else Spec()
    notes, problems = [], []

    touched = set()
    for name in patch.model_fields_set - {"filters"}:
        data[name] = getattr(patch, name)
        touched.add(name)
    if "filters" in patch.model_fields_set:
        for name in patch.filters.model_fields_set:
            data["filters"][name] = getattr(patch.filters, name)
            touched.add(f"filters.{name}")

    for path in dict.fromkeys(reply.clear):
        if path not in CLEARABLE:
            problems.append(f"'{path}' is not a setting that can be cleared (choose from: {', '.join(CLEARABLE)}).")
        elif path in touched or any(t.startswith(path + ".") for t in touched):
            problems.append(f"'{path}' is both set and cleared in the same edit.")
        elif path == "filters":
            data["filters"] = SpecFilters().model_dump()
        elif path.startswith("filters."):
            name = path.split(".", 1)[1]
            data["filters"][name] = _default(SpecFilters, name)
        else:
            data[path] = _default(Spec, path)
    if problems:
        return None, notes, problems

    new_kind = data["kind"]
    if new_kind and new_kind != base.kind and base.kind is not None:
        for name in THRESHOLD_FIELDS:
            if name not in RELEVANT_FIELDS[new_kind] and data[name] is not None and name not in touched:
                notes.append(f"Dropped {LABELS.get(name, name)}: it does not apply to {new_kind.replace('_', ' ')} patterns.")
                data[name] = None

    try:
        return Spec.model_validate(data), notes, []
    except ValidationError as e:
        parts = [f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()[:4]]
        return None, notes, ["The edited request is not valid: " + "; ".join(parts) + "."]


def _show(value) -> str:
    if value is None or value == [] or value == "":
        return "(not set)"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def diff_specs(old: Spec | None, new: Spec) -> list[str]:
    """What changed between two specs, one plain line per setting."""
    a = (old or Spec()).model_dump()
    b = new.model_dump()
    lines = []
    for name in TOP_LEVEL:
        if a[name] != b[name]:
            lines.append(f"{LABELS.get(name, name)}: {_show(a[name])} -> {_show(b[name])}")
    for name in FILTER_FIELDS:
        if a["filters"][name] != b["filters"][name]:
            lines.append(f"{LABELS[f'filters.{name}']}: {_show(a['filters'][name])} -> {_show(b['filters'][name])}")
    return lines
