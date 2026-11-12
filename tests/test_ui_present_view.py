"""The plain-words helpers (ui/present.py) and the screen logic (ui/view.py). No Streamlit, no network."""
from tests.ui_samples import A, B, CATALOG, HEALTH_OK, reply, result
from ui import present, view


# ------------------------------------------------------------------ present.py
def test_every_kind_has_words_examples_and_refinements():
    for kind in ("frequent", "rare", "high_utility"):
        info = present.kind_info(kind)
        assert info["title"] and info["icon"] and info["blurb"]
        assert present.EXAMPLES[kind] and present.REFINEMENTS[kind]
        assert all(label and message for label, message in present.REFINEMENTS[kind])
    assert present.kind_info("nonsense")["title"] == "Patterns"


def test_primary_value_and_formatting_per_kind():
    assert present.primary_value("frequent", {"support": 0.4}) == 40
    assert present.primary_value("rare", {"strength": 0.8}) == 80
    assert present.primary_value("high_utility", {"utility": 1234.5}) == 1234.5
    assert present.format_primary("frequent", 40) == "40.0%"
    assert present.format_primary("high_utility", 1234.7) == "1,235"


def test_plain_metrics_use_everyday_words():
    assert "46.15% of orders (30 orders)" in present.plain_metrics("frequent", {"support": 0.4615, "count": 30})
    assert "80% of the time" in present.plain_metrics("rare", {"strength": 0.8, "count": 12})
    assert "earned 1,200" in present.plain_metrics("high_utility", {"utility": 1200, "utility_share": 0.05, "count": 9})


def test_chips_describe_what_was_understood():
    spec = {"kind": "high_utility", "strictness": "strict", "max_len": 2, "top_k": 5,
            "filters": {"weekdays": ["Sunday"], "months": [12], "exclude_products": ["BUNTING"], "weekend": True}}
    chips = dict(present.understood_chips(spec))
    assert chips["Looking for"] == "Money makers" and chips["Strictness"] == "Strict"
    assert chips["Days"] == "Sunday" and chips["Months"] == "December" and chips["Leave out"] == "BUNTING"
    assert chips["Size"] == "Pairs" and chips["Show"] == "Top 5" and chips["When"] == "Weekends"
    assert present.understood_chips(None) == [] and present.understood_chips({}) == []


def test_chip_for_thresholds_and_triples():
    chips = dict(present.understood_chips({"kind": "frequent", "max_len": 3, "min_support": 0.025}))
    assert chips["Size"] == "Up to 3 products" and chips["Min. share of orders"] == "2.5%"


def test_friendly_error_uses_the_servers_message():
    head, detail = present.friendly_error({"code": "limit_exceeded", "message": "Too heavy: raise the threshold."})
    assert head == "That request is too heavy to run" and "raise the threshold" in detail
    assert present.friendly_error(None) is None
    assert present.friendly_error({"code": "weird", "message": "x"})[0] == "Something went wrong"


def test_short_names_and_chart_labels():
    assert present.short("A" * 40, 10) == "A" * 9 + "…" and present.short("ok") == "ok"
    row = {"itemset": [A, B]}
    assert present.combo_name(row) == f"{A} + {B}" and present.combo_lines(row).count("<br>") == 1


# ------------------------------------------------------------------ view.py
def test_answer_badge_tells_who_wrote_the_answer():
    assert view.answer_badge({"source": "llm"})[1] == "ok" and "model" in view.answer_badge({"source": "llm"})[0]
    label, tone = view.answer_badge({"source": "template", "reason": "numbers did not match"})
    assert tone == "warn" and "template" in label.lower()
    assert view.answer_badge(None)[1] == "muted"
    assert view.answer_reason({"source": "template", "reason": "numbers did not match"}) == "numbers did not match"
    assert view.answer_reason({"source": "llm", "reason": None}) == "" and view.answer_reason(None) == ""


def test_status_pills_for_ready_and_broken_systems():
    pills = {p["label"]: p for p in view.readiness_pills(HEALTH_OK)}
    assert [p["state"] for p in pills.values()] == ["ok", "ok", "ok"] and "19,960" in pills["Data"]["detail"]
    assert pills["Model"]["detail"] == "gemini-3.6-flash"
    no_java = dict(HEALTH_OK, mining={"java": False, "spmf_jar": True, "high_utility_available": False})
    assert {p["label"]: p["state"] for p in view.readiness_pills(no_java)}["Java"] == "warn"
    assert view.readiness_pills(None)[0] == {"label": "Server", "state": "bad", "detail": "not reachable"}


def test_blocking_problems_name_the_fix():
    assert view.blocking_problem(HEALTH_OK) is None and view.blocking_problem(None) is None
    no_key = dict(HEALTH_OK, llm={"ready": False, "model": None, "error": "Ollama is unavailable."})
    head, fix = view.blocking_problem(no_key)
    assert "language model" in head and "Ollama is unavailable." in fix and "restart" in fix.lower()
    no_data = dict(HEALTH_OK, data={"ready": False, "baskets": None, "error": "clean file missing"})
    head, fix = view.blocking_problem(no_data)
    assert "data" in head and "prepare_data" in fix


def test_metric_cards_show_orders_combinations_and_time():
    cards = view.metric_cards(reply())
    assert [c[0] for c in cards] == ["Orders analysed", "Combinations found", "Time taken"]
    assert cards[0][1] == "65" and cards[1][1] == "3" and cards[2][1] == "1.4s"
    assert "counting 0.1s" in cards[2][2] and "writing 0.9s" in cards[2][2]


def test_table_rows_have_kind_specific_columns():
    freq = view.table_rows(result("frequent", 2))
    assert list(freq[0]) == ["Rank", "Products", "Size", "Share of orders", "Orders"] and freq[0]["Rank"] == 1
    rare = view.table_rows(result("rare", 2))
    assert "Linked strength" in rare[0] and "Share of orders" in rare[0]
    money = view.table_rows(result("high_utility", 2))
    assert "Sales earned" in money[0] and "Share of sales" in money[0]
    assert view.table_rows(None) == [] and view.table_rows(result(n=0)) == []
    triple = view.table_rows(result("frequent", 2, size=3))
    assert triple[0]["Size"] == 3 and triple[0]["Products"].count(" + ") == 2


def test_csv_download_has_header_rows_and_a_bom():
    data = view.table_csv(view.table_rows(result("frequent", 2)))
    text = data.decode("utf-8-sig").splitlines()
    assert data.startswith(b"\xef\xbb\xbf") and text[0] == "Rank,Products,Size,Share of orders,Orders" and len(text) == 3
    assert view.table_csv([]) == b""


def test_kind_headline_and_empty_detection():
    assert "Often bought together" in view.headline(reply()) and view.headline(reply(status="needs_clarification")) == "A quick question first"
    assert view.headline(reply(status="out_of_scope")) == "That is outside what I can look at"
    assert view.has_results(reply()) and not view.has_results(reply(n=0)) and not view.has_results({"result": None})
    assert view.reply_kind(reply(kind="rare")) == "rare" and view.reply_kind({"request": {"kind": "frequent"}}) == "frequent"


def test_refinements_depend_on_the_kind():
    labels = [label for label, _ in view.refinement_buttons("rare")]
    assert "Often together instead" in labels and "Unusual pairs" not in labels
    assert view.refinement_buttons(None) == present.REFINEMENTS["frequent"]


def test_dollar_signs_are_escaped_for_markdown():
    assert view.escape_dollars("$5 and $10") == "\\$5 and \\$10" and view.escape_dollars(None) == ""


def test_catalog_sample_matches_what_the_sidebar_reads():
    assert CATALOG["date_range"][0] and "Saturday" not in CATALOG["weekdays_with_orders"]
