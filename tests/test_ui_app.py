"""The Streamlit page (ui/app.py), driven headlessly with Streamlit's own test mode.

A scripted stand-in replaces the API client, so these tests need no server, no model and no network.
They skip automatically if streamlit or plotly is not installed.
"""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest  # noqa: E402

from tests.ui_samples import CATALOG, HEALTH_OK, reply  # noqa: E402
from ui.api_client import ApiProblem  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "ui" / "app.py")


class FakeApi:
    """Plays back scripted replies (or raises scripted problems) and remembers what the page asked."""

    base_url = "http://fake.test"

    def __init__(self, replies=(), health=None, problem=None):
        self.replies, self.calls = list(replies), []
        self._health, self._problem = health or HEALTH_OK, problem

    def probe(self):
        return (None, self._problem) if self._problem else (self._health, None)

    def catalog(self):
        return CATALOG

    def _next(self, *call):
        self.calls.append(call)
        item = self.replies.pop(0) if self.replies else reply()
        if isinstance(item, Exception):
            raise item
        return item

    def chat(self, message, session_id=None, explain=True):
        return self._next("chat", message, session_id, explain)

    def undo(self, session_id, explain=True):
        return self._next("undo", session_id, explain)

    def run(self, session_id, explain=True):
        return self._next("run", session_id, explain)

    def reset(self, session_id):
        return self._next("reset", session_id)


def start(api):
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["api"] = api
    return at.run()


def texts(at):
    return " ".join(str(m.value) for m in at.markdown)


def button_keys(at):
    return {b.key for b in at.button}


# ------------------------------------------------------------------ first screens
def test_welcome_screen_has_three_kinds_with_clickable_examples():
    at = start(FakeApi())
    assert not at.exception
    page = texts(at)
    assert "Often bought together" in page and "Hidden gems" in page and "Money makers" in page
    assert {"ex_frequent_0", "ex_rare_0", "ex_high_utility_0"} <= button_keys(at)
    assert len(at.chat_input) == 1 and len(at.chat_message) == 0


def test_status_pills_and_sidebar_summary_show_when_ready():
    at = start(FakeApi())
    page = texts(at)
    assert "gemini-3.6-flash" in page and "19,960" in page
    assert any("No orders on: Saturday" in str(c.value) for c in at.sidebar.caption)


def test_server_down_screen_explains_and_offers_a_retry():
    problem = ApiProblem("server_down", "The pattern-mining server is not running", "Start it: uvicorn api.main:create_app")
    at = start(FakeApi(problem=problem))
    assert not at.exception
    assert any("not running" in str(e.value) for e in at.error)
    assert any("uvicorn" in str(c.value) for c in at.code)
    assert "recheck" in button_keys(at) and len(at.chat_input) == 0


def test_missing_key_screen_names_the_fix():
    health = dict(HEALTH_OK, llm={"ready": False, "model": None, "error": "Ollama is unavailable."})
    at = start(FakeApi(health=health))
    assert any("language model is not set up" in str(e.value) and "Ollama is unavailable." in str(e.value) for e in at.error)
    assert len(at.chat_input) == 0


# ------------------------------------------------------------------ a whole conversation
def test_typing_a_question_shows_the_answer_chips_charts_and_next_steps():
    api = FakeApi([reply(answer="Bunting and cakestands sell together.")])
    at = start(api).chat_input[0].set_value("what sells together?").run()
    assert not at.exception and api.calls == [("chat", "what sells together?", None, True)]
    assert len(at.chat_message) == 2                                    # the question and the answer
    page = texts(at)
    assert "Bunting and cakestands sell together." in page and "every number checked" in page
    assert "What I understood" in " ".join(str(e.label) for e in at.expander)
    assert len(at.get("plotly_chart")) >= 3                              # ranking, skyline, star map
    assert {"ref_0", "undo", "rerun", "reset"} <= button_keys(at)
    assert at.button(key="undo").disabled and not at.button(key="rerun").disabled


def test_clicking_an_example_asks_that_question():
    api = FakeApi()
    at = start(api).button(key="ex_frequent_0").click().run()
    assert not at.exception and api.calls[0][:2] == ("chat", "Which products are usually bought together?")


def test_refinement_buttons_send_an_edit_in_the_same_conversation():
    api = FakeApi([reply(), reply(changes=["Only Sundays"], session={"current": {"kind": "frequent"}, "turns": 2,
                                                                      "can_undo": True, "pending_questions": 0})])
    at = start(api).chat_input[0].set_value("what sells together?").run()
    at = at.button(key="ref_0").click().run()
    assert api.calls[1] == ("chat", "only on Sundays", "sid-1", True)
    assert "Only Sundays" in texts(at) and not at.button(key="undo").disabled


def test_undo_run_again_and_start_over_use_the_session():
    api = FakeApi([reply(), reply(), reply(), reply(status="reset", message="Starting over.")])
    at = start(api).chat_input[0].set_value("hello").run()
    at = at.button(key="rerun").click().run()
    assert api.calls[1] == ("run", "sid-1", True)
    at = at.button(key="reset").click().run()
    assert api.calls[-1] == ("reset", "sid-1") and len(at.chat_message) == 0 and "ex_frequent_0" in button_keys(at)


def test_clarifying_questions_become_buttons():
    ask = reply(status="needs_clarification", message="Which one?", result=None, answer=None,
                questions=[{"field": "include_products", "question": "Which cakestand do you mean?", "options": ["REGENCY CAKESTAND 3 TIER", "CAKESTAND 2 TIER"]}])
    api = FakeApi([ask, reply()])
    at = start(api).chat_input[0].set_value("cakestand").run()
    assert "Which cakestand do you mean?" in texts(at) and {"q_0_0", "q_0_1"} <= button_keys(at)
    at = at.button(key="q_0_0").click().run()
    assert api.calls[1][:2] == ("chat", "REGENCY CAKESTAND 3 TIER")


def test_the_template_badge_and_reason_show_when_the_model_text_was_rejected():
    at = start(FakeApi([reply(source="template")])).chat_input[0].set_value("x").run()
    page = texts(at)
    assert "Fixed template" in page and any("numbers did not match" in str(c.value) for c in at.caption)


def test_empty_results_suggest_loosening_instead_of_charts():
    empty = reply(n=0)
    at = start(FakeApi([empty])).chat_input[0].set_value("something strict").run()
    assert not at.exception and len(at.get("plotly_chart")) == 0
    assert any("lower threshold" in str(i.value) for i in at.info) and "ref_0" in button_keys(at)


def test_off_topic_and_failed_replies_show_their_message():
    off = reply(status="out_of_scope", message="I only look at sales patterns.", result=None, answer=None)
    at = start(FakeApi([off])).chat_input[0].set_value("what is the weather").run()
    assert any("only look at sales patterns" in str(i.value) for i in at.info)


def test_model_unavailable_offers_to_try_again():
    down = reply(status="llm_unavailable", message="Gemini is busy.", result=None, answer=None)
    api = FakeApi([down, reply()])
    at = start(api).chat_input[0].set_value("hello").run()
    assert any("Gemini is busy." in str(e.value) for e in at.error) and "retry_msg" in button_keys(at)
    at = at.button(key="retry_msg").click().run()
    assert [c[1] for c in api.calls] == ["hello", "hello"]


def test_connection_problems_during_a_chat_show_a_retry():
    problem = ApiProblem("rate_limited", "Too many requests", "Please wait 12 seconds.", 12)
    api = FakeApi([problem, reply()])
    at = start(api).chat_input[0].set_value("hello").run()
    assert any("Too many requests" in str(e.value) for e in at.error) and "retry_problem" in button_keys(at)
    at = at.button(key="retry_problem").click().run()
    assert len(api.calls) == 2 and not at.exception


def test_expired_conversation_starts_fresh_on_the_next_question():
    gone = ApiProblem("session_expired", "This conversation has expired", "Ask again.")
    api = FakeApi([reply(), gone, reply()])
    at = start(api).chat_input[0].set_value("one").run()
    at = at.chat_input[0].set_value("two").run()
    at = at.chat_input[0].set_value("three").run()
    assert [c[2] for c in api.calls] == [None, "sid-1", None]            # after the expiry no old id is sent


def test_turning_the_written_answer_off_saves_the_model_call():
    api = FakeApi([reply(answer=None)])
    at = start(api)
    at.toggle(key="explain").set_value(False).run()
    at.chat_input[0].set_value("hello").run()
    assert api.calls[0][3] is False
