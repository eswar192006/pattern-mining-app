import pytest

from src.llm.numbercheck import Num, check_text, find_numbers, unknown_products

NAMES = ["JUMBO BAG PINK POLKADOT", "REGENCY CAKESTAND 3 TIER", "SET OF 6 SPICE TINS"]
ALLOWED = [Num(4.454, 0, True, ""), Num(59.0, 0, True, ""), Num(808, 0, False, ""), Num(59435, 0, False, ""),
           Num(1601, 0, False, ""), Num(0.86, 0, False, ""), Num(18141, 0, False, ""), Num(9538303, 0, False, ""),
           Num(1, 0, False, ""), Num(2, 0, False, "")]


def clean(text):
    return check_text(text, ALLOWED, NAMES) == []


# ---------------------------------------------------------------- reading numbers
@pytest.mark.parametrize("text,value,place,pct", [
    ("808", 808, 1, False), ("1,601", 1601, 1, False), ("1601", 1601, 1, False),
    ("4.45%", 4.45, 0.01, True), ("4.5 percent", 4.5, 0.1, True), ("59,000", 59000, 1000, False),
    ("60,000", 60000, 10000, False), ("59k", 59000, 1000, False), ("59.4k", 59400, 100, False),
    ("9.8 million", 9.8e6, 1e5, False), ("5 million", 5e6, 1e6, False), ("20", 20, 1, False), ("0.86", 0.86, 0.01, False)])
def test_numbers_are_read_with_their_written_precision(text, value, place, pct):
    (n,) = find_numbers(f"about {text} here")
    assert n.value == pytest.approx(value) and n.place == pytest.approx(place) and n.pct == pct


def test_digits_inside_words_are_not_numbers():
    assert find_numbers("T2 and A1 and 3D-printed") == [] or [n.value for n in find_numbers("T2 and A1")] == []
    assert [n.value for n in find_numbers("2 products, 3 orders")] == [2, 3]


# ---------------------------------------------------------------- honest rounding is fine
@pytest.mark.parametrize("text", [
    "bought together in 4.45% of orders", "in about 4.5% of orders", "roughly 4% of orders",
    "808 orders", "about 59,000 in sales", "around 59.4k in sales", "about 60,000 in sales", "59 percent",
    "1,601 combinations", "1601 combinations", "about 1,600 combinations", "18,141 orders",
    "total sales of about 9.5 million", "about 9.54 million", "strength 0.86", "strength about 0.9",
    "the 1 and 2 results", "top 2 pairs",
])
def test_rounded_and_exact_figures_pass(text):
    assert clean(text), check_text(text, ALLOWED, NAMES)


# ---------------------------------------------------------------- invented figures are caught
@pytest.mark.parametrize("text", [
    "nearly 5% of orders",                 # 4.454 does not round to 5
    "809 orders", "820 orders",            # not 808
    "62,000 in sales", "about 61,000 in sales",
    "1,700 combinations",
    "twice as often (200%)", "17% more than the next",
    "in 2011 it sold", "since December 2010",   # dates and years are not allowed
    "808% of orders",                      # a count is not a percentage
    "4.45 orders",                         # a percentage is not a count
    "strength 0.7", "7 products",
    "about 2,000 combinations",            # nearest thousand, but 25% too high
])
def test_invented_or_mismatched_figures_are_flagged(text):
    problems = check_text(text, ALLOWED, NAMES)
    assert problems and "numbers that are not in the data" in problems[0], text


def test_the_problem_message_lists_every_offender_once():
    (p,) = check_text("52% of orders, then 52% again and 3,333 orders", ALLOWED, NAMES)
    assert p == "numbers that are not in the data: 52%, 3,333"


# ---------------------------------------------------------------- product names
def test_digits_in_real_product_names_are_ignored_any_case():
    assert clean("REGENCY CAKESTAND 3 TIER sells in 808 orders")
    assert clean("the Regency Cakestand 3 Tier sells in 808 orders")
    assert clean("SET OF 6 SPICE TINS and JUMBO BAG PINK POLKADOT")
    assert not clean("the 3 tier cakestand sells in 808 orders")        # paraphrased name: stray '3'


def test_unknown_products_are_found():
    assert unknown_products("Try MEGA MUG SET with JUMBO BAG PINK POLKADOT.", NAMES) == ["MEGA MUG SET"]
    assert unknown_products("JUMBO BAG is popular", NAMES) == []            # part of a real name
    assert unknown_products("POLKADOT and the UK market", NAMES) == []      # single capitals are ignored
    assert unknown_products("a SET OF 6 SPICE TINS.", NAMES) == []          # trailing punctuation
    assert "product names that are not in the data" in check_text("buy PINK ELEPHANT LAMP", ALLOWED, NAMES)[0]


# ---------------------------------------------------------------- figures in words, and comparisons
@pytest.mark.parametrize("text", [
    "one pair and two products", "the top pair", "the first of many",
    "the 9.5 million figure", "total sales of 9.5 million",     # 'million' after a digit is part of that number
])
def test_harmless_words_pass(text):
    assert clean(text), check_text(text, ALLOWED, NAMES)


@pytest.mark.parametrize("text,fragment", [
    ("seventeen orders", "figures written in words"),
    ("four products together", "figures written in words"),
    ("about two hundred orders", "hundred"),
    ("a million sales", "million"),
    ("That is twice as often as the next pair.", "twice"),
    ("It doubled sales", "doubled"),
    ("half of all orders", "half"),
    ("a threefold increase", "threefold"),
    ("3 times as many", "numbers that are not in the data"),
])
def test_number_words_and_comparisons_are_flagged(text, fragment):
    problems = check_text(text, ALLOWED, NAMES)
    assert problems and fragment in " ".join(problems), (text, problems)


def test_a_spelled_number_is_fine_when_it_is_in_the_data():
    allowed = ALLOWED + [Num(3, 0, False, "")]
    assert check_text("the top three pairs", allowed, NAMES) == []
    assert check_text("the top three pairs", ALLOWED, NAMES) != []
