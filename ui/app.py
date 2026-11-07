"""The chat page.   Start the server first, then the page (two terminals, from the project folder):

    uvicorn api.main:create_app --factory --port 8000
    streamlit run ui/app.py

The page only talks to the API (ui/api_client.py); it never imports the mining or the model code.
All wording and numbers come from ui/present.py and ui/view.py, which are tested without Streamlit.
"""
import html
import inspect
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:                     # `streamlit run ui/app.py` only puts ui/ on the path
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from ui import charts, view  # noqa: E402
from ui.api_client import START_COMMAND, ApiClient, ApiProblem  # noqa: E402
from ui.present import EXAMPLES, KINDS, friendly_error, kind_info, timing_summary, understood_chips  # noqa: E402

HEALTH_REFRESH_S = 30

CSS = """
<style>
.hero{background:linear-gradient(120deg,#7c5cff 0%,#c04bff 45%,#22d3ee 100%);border-radius:16px;padding:20px 26px;margin-bottom:12px}
.hero h1{margin:0;font-size:1.65rem;line-height:1.25;color:#fff}
.hero p{margin:6px 0 0;color:rgba(255,255,255,.92);font-size:.98rem}
.pill{display:inline-block;padding:3px 11px;border-radius:999px;font-size:.78rem;margin:0 6px 6px 0;border:1px solid rgba(255,255,255,.18)}
.pill.ok{background:rgba(74,222,128,.14);color:#86efac}
.pill.warn{background:rgba(255,180,84,.14);color:#ffb454}
.pill.bad{background:rgba(255,92,157,.16);color:#ff8fb8}
.pill.muted,.pill.info{background:rgba(201,209,245,.10);color:#c9d1f5}
.chip{display:inline-block;padding:4px 12px;border-radius:10px;font-size:.85rem;margin:0 8px 8px 0;background:rgba(124,92,255,.18);border:1px solid rgba(124,92,255,.5)}
.chip b{color:#c9b8ff;font-weight:600;margin-right:6px}
.metric{border:1px solid rgba(255,255,255,.12);border-radius:14px;padding:12px 16px;background:rgba(255,255,255,.04)}
.metric .v{font-size:1.7rem;font-weight:700;line-height:1.2}
.metric .l{font-size:.8rem;opacity:.75;text-transform:uppercase;letter-spacing:.04em}
.metric .c{font-size:.78rem;opacity:.65;margin-top:2px}
.kind-title{font-size:1.15rem;font-weight:700;margin-bottom:4px}
.small{font-size:.85rem;opacity:.75}
</style>
"""


# ------------------------------------------------------------------------------------------ small helpers
def esc(text) -> str:
    return html.escape(str(text), quote=True)


def fit(fn) -> dict:
    """Keyword arguments that make a widget fill its column, whatever the Streamlit version calls it."""
    return {"width": "stretch"} if "width" in inspect.signature(fn).parameters else {"use_container_width": True}


def pill(label: str, state: str, detail: str = "") -> str:
    text = f"{esc(label)}: {esc(detail)}" if detail else esc(label)
    return f'<span class="pill {esc(state)}">{text}</span>'


def queue_job(action: str, text: str = "") -> None:
    """Button handlers only note what to do; the page does it on the next run, below the history."""
    st.session_state.queue = (action, text)
    if action == "chat":
        st.session_state.last_message = text


def init_state() -> None:
    ss = st.session_state
    if "api" not in ss:
        ss.api = ApiClient()
    for key, value in (("sid", None), ("history", []), ("queue", None), ("explain", True), ("health", None),
                       ("health_problem", None), ("health_at", 0.0), ("catalog", None), ("last_message", ""),
                       ("figs", {})):
        ss.setdefault(key, value)


def refresh_health(force: bool = False) -> None:
    ss = st.session_state
    stale = time.monotonic() - ss.health_at > HEALTH_REFRESH_S
    if force or stale or (ss.health is None and ss.health_problem is None):
        ss.health, ss.health_problem = ss.api.probe()
        ss.health_at = time.monotonic()
        ss.catalog = None
    if ss.health and ss.catalog is None and not view.blocking_problem(ss.health):
        try:
            ss.catalog = ss.api.catalog()
        except ApiProblem:
            ss.catalog = None


def recheck() -> None:
    st.session_state.health = st.session_state.health_problem = None
    st.session_state.health_at = 0.0


# ------------------------------------------------------------------------------------------ header and screens
def render_header() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="hero"><h1>🛍️ Sales pattern explorer</h1>'
                "<p>Ask in plain English. A language model understands and explains; exact algorithms do all the counting.</p></div>",
                unsafe_allow_html=True)
    if (st.session_state.health or {}).get("demo_mode") == "replay":
        st.info("Replay demo is active. Saved responses interpret the six walkthrough questions; all results are mined from the local data. No Gemini requests are sent.")
    pills = view.readiness_pills(st.session_state.health)
    st.markdown("".join(pill(p["label"], p["state"], p["detail"]) for p in pills), unsafe_allow_html=True)


def render_blocked(headline: str, detail: str, command: str | None = None) -> None:
    st.error(f"**{headline}**\n\n{detail}")
    if command:
        st.code(command, language="bash")
    st.button("Check again", key="recheck", on_click=recheck)


def render_welcome(health: dict | None) -> None:
    st.markdown("#### What would you like to know about your sales?")
    st.caption("Pick one of these to start, or type your own question below.")
    java_ok = bool((health or {}).get("mining", {}).get("high_utility_available"))
    for col, kind in zip(st.columns(3), KINDS):
        info = kind_info(kind)
        with col, st.container(border=True):
            st.markdown(f'<div class="kind-title">{esc(info["icon"])}  {esc(info["title"])}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="small">{esc(info["blurb"])}</div>', unsafe_allow_html=True)
            if kind == "high_utility" and not java_ok:
                st.caption("⚠️ Needs Java, which was not found on the server.")
            for i, example in enumerate(EXAMPLES[kind]):
                st.button(example, key=f"ex_{kind}_{i}", on_click=queue_job, args=("chat", example))


def render_sidebar() -> None:
    ss = st.session_state
    with st.sidebar:
        st.markdown("### Your data")
        cat = ss.catalog
        if cat:
            st.metric("Orders", f"{cat['baskets']:,}")
            st.metric("Products", f"{cat['products']:,}")
            if cat.get("date_range"):
                st.caption(f"From {cat['date_range'][0]} to {cat['date_range'][1]}")
            days = cat.get("weekdays_with_orders") or []
            missing = [d for d in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday") if d not in days]
            if missing:
                st.caption("No orders on: " + ", ".join(missing))
        else:
            st.caption("The summary appears when the server is ready.")
        st.divider()
        st.toggle("Write answers with the model", key="explain",
                  help="Off: you see the results only. It saves one model call per question, which helps on a free quota.")
        st.divider()
        with st.expander("Try asking"):
            for kind in KINDS:
                st.markdown(f"**{kind_info(kind)['title']}**")
                for i, example in enumerate(EXAMPLES[kind]):
                    st.button(example, key=f"side_ex_{kind}_{i}", on_click=queue_job, args=("chat", example))
        st.button("Check connection", key="side_recheck", on_click=recheck)
        st.caption(f"Server: {ss.api.base_url}")


# ------------------------------------------------------------------------------------------ one reply
def render_chips(reply: dict) -> None:
    spec = dict((reply.get("session") or {}).get("current") or {})
    if not spec:
        return
    spec.setdefault("kind", view.reply_kind(reply))
    chips = understood_chips(spec)
    with st.expander("What I understood", expanded=True):
        if chips:
            st.markdown("".join(f'<span class="chip"><b>{esc(a)}</b>{esc(b)}</span>' for a, b in chips),
                        unsafe_allow_html=True)
        changes = reply.get("changes") or []
        if changes:
            st.markdown("**What changed**")
            st.markdown("\n".join(f"- {view.escape_dollars(c)}" for c in changes))


def render_answer(reply: dict) -> None:
    answer = reply.get("answer")
    label, tone = view.answer_badge(answer)
    with st.container(border=True):
        st.markdown(f'<div class="kind-title">{esc(view.headline(reply))}</div>', unsafe_allow_html=True)
        if answer and answer.get("text"):
            st.markdown(view.escape_dollars(answer["text"]))
        st.markdown(pill(label, tone), unsafe_allow_html=True)
        reason = view.answer_reason(answer)
        if reason:
            st.caption(f"Why the template: {reason}")


def render_metrics(reply: dict) -> None:
    for col, (label, value, caption) in zip(st.columns(3), view.metric_cards(reply)):
        col.markdown(f'<div class="metric"><div class="l">{esc(label)}</div><div class="v">{esc(value)}</div>'
                     f'<div class="c">{esc(caption)}</div></div>', unsafe_allow_html=True)


def figure_for(idx: int, name: str, build, result: dict):
    """Build each chart once per reply; later page runs reuse it."""
    cache = st.session_state.figs
    key = (idx, name)
    if key not in cache:
        cache[key] = build(result)
    return cache[key]


def render_results(reply: dict, idx: int) -> None:
    result = reply.get("result") or {}
    if not view.has_results(reply):
        st.info(result.get("hint") or "No combinations matched. Try loosening the request.")
        return
    render_metrics(reply)
    tabs = st.tabs(["Ranking", "3D Skyline", "3D Star map", "Table", "Details"])
    plot_args = fit(st.plotly_chart)
    with tabs[0]:
        st.plotly_chart(figure_for(idx, "ranking", charts.ranking_figure, result), key=f"plot_rank_{idx}", **plot_args)
    with tabs[1]:
        st.caption("One tower per combination: taller means stronger. Drag to rotate, scroll to zoom, hover for details.")
        st.plotly_chart(figure_for(idx, "skyline", charts.skyline_figure, result), key=f"plot_sky_{idx}", **plot_args)
    with tabs[2]:
        st.caption("Products are stars; a link means they were bought together. Families of products form clusters. "
                   "Press Spin to rotate it.")
        st.plotly_chart(figure_for(idx, "stars", charts.constellation_figure, result), key=f"plot_star_{idx}", **plot_args)
    rows = view.table_rows(result)
    with tabs[3]:
        st.dataframe(rows, hide_index=True, **fit(st.dataframe))
        st.download_button("Download as CSV", view.table_csv(rows), file_name="patterns.csv", mime="text/csv",
                           key=f"csv_{idx}")
    with tabs[4]:
        render_details(reply)


def render_details(reply: dict) -> None:
    result = reply.get("result") or {}
    timing = timing_summary(reply.get("timings") or {})
    if timing:
        st.markdown(f"**Time:** {timing}")
    for note in list(reply.get("notes") or []) + list(result.get("notes") or []):
        st.markdown(f"- {view.escape_dollars(note)}")
    if result.get("filter_trace"):
        st.markdown("**How the orders were narrowed down**")
        st.dataframe([{"Step": a, "Orders left": b} for a, b in result["filter_trace"]], hide_index=True, **fit(st.dataframe))
    if reply.get("request"):
        st.markdown("**The exact request that was run**")
        st.json(reply["request"])


def render_questions(reply: dict) -> None:
    for qi, q in enumerate(reply.get("questions") or []):
        st.markdown(f"**{view.escape_dollars(q.get('question', ''))}**")
        options = q.get("options") or []
        if options:
            for col, (oi, option) in zip(st.columns(min(len(options), 4)), list(enumerate(options))[:4]):
                col.button(str(option), key=f"q_{qi}_{oi}", on_click=queue_job, args=("chat", str(option)))
            for oi, option in list(enumerate(options))[4:]:
                st.button(str(option), key=f"q_{qi}_{oi}", on_click=queue_job, args=("chat", str(option)))
        else:
            st.caption("Type your answer below.")


def render_next_steps(reply: dict) -> None:
    """Refinement chips, undo, run again and start over, under the newest reply."""
    ss = st.session_state
    st.markdown('<div class="small">Try next</div>', unsafe_allow_html=True)
    buttons = list(view.refinement_buttons(view.reply_kind(reply)))
    if reply.get("status") == "ok" and not view.has_results(reply):
        buttons.insert(0, ("Loosen it", "make it looser"))
    if reply.get("status") == "ok":
        for col, (i, (label, message)) in zip(st.columns(len(buttons)), enumerate(buttons)):
            col.button(label, key=f"ref_{i}", on_click=queue_job, args=("chat", message))
    session = reply.get("session") or {}
    c1, c2, c3, _ = st.columns([1, 1, 1, 3])
    c1.button("↩ Undo", key="undo", disabled=not session.get("can_undo"), on_click=queue_job, args=("undo",))
    c2.button("↻ Run again", key="rerun", disabled=not session.get("current"), on_click=queue_job, args=("run",))
    c3.button("✕ Start over", key="reset", on_click=queue_job, args=("reset",))
    if reply.get("status") == "llm_unavailable" and ss.last_message:
        st.button("Try the same message again", key="retry_msg", on_click=queue_job, args=("chat", ss.last_message))


def render_reply(reply: dict, idx: int, latest: bool) -> None:
    status = reply.get("status")
    if status == "ok":
        render_answer(reply)
        problem = friendly_error(reply.get("error"))
        if problem:
            st.warning(f"**{problem[0]}**\n\n{problem[1]}")
        for w in (reply.get("warnings") or [])[:3]:
            st.warning(view.escape_dollars(w))
        if latest:
            render_chips(reply)
            if reply.get("result"):
                render_results(reply, idx)
    elif status == "needs_clarification":
        with st.container(border=True):
            st.markdown(f'<div class="kind-title">{esc(view.headline(reply))}</div>', unsafe_allow_html=True)
            if reply.get("message"):
                st.markdown(view.escape_dollars(reply["message"]))
            if latest:
                render_questions(reply)
    else:
        headline, tone = view.MESSAGE_STATUSES.get(status, ("Something went wrong", "warn"))
        lines = [view.escape_dollars(reply.get("message") or "")] + [view.escape_dollars(e) for e in reply.get("errors") or []]
        text = f"**{headline}**\n\n" + "\n\n".join(x for x in lines if x)
        (st.error if tone == "bad" else st.info if tone == "info" else st.warning)(text)
    if latest:
        render_next_steps(reply)


def render_problem(entry: dict) -> None:
    problem: ApiProblem = entry["problem"]
    st.error(f"**{problem.headline}**\n\n{problem.detail}")
    if entry.get("retry") and problem.kind in ("timeout", "server_down", "rate_limited"):
        st.button("Try again", key="retry_problem", on_click=queue_job, args=entry["retry"])


# ------------------------------------------------------------------------------------------ running a job
def run_job(job: tuple) -> None:
    """Call the API for one queued action, store the outcome in the history, and redraw the page."""
    ss = st.session_state
    action, text = job
    api, explain = ss.api, bool(ss.explain)
    try:
        with st.chat_message("assistant"):
            with st.spinner("Understanding your question, counting exactly, then writing the answer…"):
                if action == "chat":
                    reply = api.chat(text, ss.sid, explain)
                elif action == "undo":
                    reply = api.undo(ss.sid, explain)
                elif action == "run":
                    reply = api.run(ss.sid, explain)
                else:
                    reply = api.reset(ss.sid) if ss.sid else {}
    except ApiProblem as problem:
        if problem.kind == "session_expired":
            ss.sid = None
        if problem.kind in ("server_down", "not_ready"):
            recheck()                                       # look at the server again on the next run
        ss.history.append({"role": "problem", "problem": problem, "retry": job if action == "chat" else None})
        st.rerun()
        return
    ss.sid = reply.get("session_id", ss.sid)
    ss.figs = {}
    if action == "reset":
        ss.history, ss.last_message = [], ""
    else:
        ss.history.append({"role": "assistant", "reply": reply})
    st.rerun()


USER_LABELS = {"undo": "↩ Undo", "run": "↻ Run again"}


def render_history(busy: bool) -> None:
    history = st.session_state.history
    last_reply = max((i for i, e in enumerate(history) if e["role"] == "assistant"), default=-1)
    for i, entry in enumerate(history):
        role = entry["role"]
        if role == "user":
            with st.chat_message("user"):
                st.markdown(view.escape_dollars(entry["text"]))
        elif role == "assistant":
            with st.chat_message("assistant"):
                render_reply(entry["reply"], i, latest=(i == last_reply and i == len(history) - 1 and not busy))
        else:
            with st.chat_message("assistant"):
                render_problem(entry)


def main() -> None:
    st.set_page_config(page_title="Sales pattern explorer", page_icon="🛍️", layout="wide")
    init_state()
    refresh_health()
    ss = st.session_state
    render_header()
    render_sidebar()

    if ss.health_problem is not None:
        render_blocked(ss.health_problem.headline, ss.health_problem.detail, START_COMMAND)
        st.stop()
    blocking = view.blocking_problem(ss.health)
    if blocking:
        render_blocked(blocking[0], blocking[1])
        st.stop()

    job, ss.queue = ss.queue, None
    if job and job[0] in ("chat", "undo", "run"):
        ss.history.append({"role": "user", "text": job[1] if job[0] == "chat" else USER_LABELS[job[0]]})
    if not ss.history and not job:
        render_welcome(ss.health)
    render_history(busy=bool(job))
    if job:
        run_job(job)

    text = st.chat_input("Ask about your sales patterns, for example: what sells together on Sundays?")
    if text and text.strip():
        queue_job("chat", text.strip())
        st.rerun()


main()
