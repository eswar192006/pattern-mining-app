"""Check that a text written by a language model contains no invented numbers or products.

The model may ROUND a figure ("4.45%" -> "about 4.5%", "59,435" -> "about 59,000") but must not
introduce one that is not in the facts it was given (no sums, differences, ratios, dates...).

A number written in the text is accepted when some supplied number lies within HALF A UNIT of its
last written digit, which is exactly what honest rounding means:
    "4.5%"      last digit = tenths      -> accepts 4.45 .. 4.55 (e.g. 4.454)
    "59,000"    last digit = thousands   -> accepts 58,500 .. 59,500 (only for numbers >= 1,000)
    "9.8 million", "59k" are scaled the same way.
Rounding may never change a figure by more than 15% ("about 2,000" for 1,601 would be rejected,
even though it is the nearest thousand), because a stretched figure misleads just like an invented one.
Percentages are only compared with percentages, plain numbers with plain numbers.
Figures written in words ("seventeen") are checked the same way, and comparison words ("twice",
"double", "half") are rejected outright, since the data gives no comparisons.
Digits inside product names (REGENCY CAKESTAND 3 TIER) are ignored, and ALL-CAPS product-like
phrases that are not (part of) a supplied product name are reported.
"""
import re
from dataclasses import dataclass

_NUM = re.compile(
    r"(?<![A-Za-z0-9.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?"
    r"(?:\s?(%|percent\b|per cent\b|k\b|K\b|thousand\b|million\b|billion\b))?")
_SCALE = {"k": 1e3, "K": 1e3, "thousand": 1e3, "million": 1e6, "billion": 1e9}
MAX_ROUNDING = 0.15
_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split())}
_WORDS.update({"thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
               "hundred": 100, "thousand": 1000, "million": 1_000_000})
_WORD_RE = re.compile(r"\b(" + "|".join(_WORDS) + r")\b", re.I)
_COMPARE_RE = re.compile(r"\b(twice|thrice|double[ds]?|triple[ds]?|half|quarter|dozens?|times as|\w+-fold|(?:two|three|four|five|six|seven|eight|nine|ten)fold)\b", re.I)
_CAPS_RUN = re.compile(r"\b[A-Z][A-Z0-9&'/.-]*(?: [A-Z][A-Z0-9&'/.-]*)+")


@dataclass(frozen=True)
class Num:
    value: float
    place: float          # size of the last written digit (0.1 for "4.5", 1000 for "59,000")
    pct: bool
    text: str


def find_numbers(text: str) -> list[Num]:
    out = []
    for m in _NUM.finditer(text):
        whole, frac, unit = m.group(1), m.group(2), m.group(3)
        digits = whole.replace(",", "")
        value = float(digits + (frac or ""))
        if frac:
            place = 10.0 ** -(len(frac) - 1)
        elif value >= 1000:
            place = 10.0 ** (len(digits) - len(digits.rstrip("0")))     # 59,000 -> rounded to thousands
        else:
            place = 1.0
        if unit in _SCALE:
            value, place = value * _SCALE[unit], place * _SCALE[unit]
        out.append(Num(value, place, unit in ("%", "percent", "per cent"), m.group(0).strip()))
    return out


def strip_names(text: str, names) -> str:
    for name in sorted({n for n in names if n}, key=len, reverse=True):
        for variant in {name, name.rstrip(".")}:
            text = re.sub(re.escape(variant), " ", text, flags=re.I)
    return text


def unknown_products(text: str, names) -> list[str]:
    """ALL-CAPS phrases of 2+ words that are not (part of) a supplied product name."""
    haystack = " | ".join(names)
    bad = []
    for run in _CAPS_RUN.findall(text):
        run = run.strip(".,;:)(")
        if " " in run and run not in haystack and run not in bad:
            bad.append(run)
    return bad


def check_text(text: str, allowed: list[Num], names=()) -> list[str]:
    """-> list of problems (empty = the text is clean)."""
    problems = []
    unknown = unknown_products(text, names)
    if unknown:
        problems.append("product names that are not in the data: " + ", ".join(f"'{u}'" for u in unknown))
    body = strip_names(text, names)
    unsupported = []
    for n in find_numbers(body):
        ok = any(a.pct == n.pct and abs(a.value - n.value) <= min(n.place / 2, MAX_ROUNDING * abs(a.value)) + 1e-9
                 for a in allowed)
        if not ok and n.text not in unsupported:
            unsupported.append(n.text)
    if unsupported:
        problems.append("numbers that are not in the data: " + ", ".join(unsupported))

    words_only = _NUM.sub(" ", body)                      # "9.8 million" was already checked as a number
    spelled = []
    for m in _WORD_RE.finditer(words_only):
        w = m.group(1).lower()
        if not any(not a.pct and abs(a.value - _WORDS[w]) < 0.5 for a in allowed) and w not in spelled:
            spelled.append(w)
    if spelled:
        problems.append("figures written in words that are not in the data: " + ", ".join(spelled))
    compare = sorted({m.group(1).lower() for m in _COMPARE_RE.finditer(words_only)})
    if compare:
        problems.append("comparisons or calculations that the data does not give: " + ", ".join(compare))
    return problems
