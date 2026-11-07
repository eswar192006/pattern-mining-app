"""Talk to the whole system. Ask a question, then keep refining it.

    python -m src.demo_chat                 # chat: the request is shown after every message
    python -m src.demo_chat --explain       # also mine and write the answer in plain English

Try:  which products sell together?  ->  only on Sundays  ->  now by money  ->  stricter  ->  undo
Type  /spec  to see the current request,  /undo  and  /reset  to go back or start over without asking
the model,  and an empty line to quit.

Needs the configured live model (Ollama by default) and the cleaned data (python -m src.prepare_data).
Money-based questions also need Java and tools/spmf.jar when mining.
"""
import json
import sys
import textwrap

from src.core.errors import LLMUnavailable, MiningError
from src.demo_engine import print_result
from src.llm.client import create_llm_client
from src.llm.conversation import converse, reset_session, undo_session
from src.llm.explainer import explain_result
from src.llm.session import Session
from src.mining.engine import run_mining


def run_and_show(request, llm, question, explain):
    try:
        result = run_mining(request)
    except MiningError as e:
        print("  Could not run:", e)
        return
    print_result(result)
    if explain:
        e = explain_result(result, llm, question=question)
        print(f"\nANSWER  (written by: {e.source}; model calls: {e.attempts}; {e.latency_s:.1f}s)")
        print(textwrap.fill(e.text, 100, initial_indent="  ", subsequent_indent="  "))
        if e.reason:
            print("  [fixed template used because:", e.reason + "]")


def show_turn(t, llm, explain, question):
    print(f"\n[{t.turn_type or t.status}]  status: {t.status}   ({t.attempts} model call(s), {t.latency_s:.1f}s)")
    if t.message:
        print("  ", t.message)
    for line in t.changes:
        print("  changed:", line)
    for n in t.notes:
        print("  note:", n)
    v = t.validation
    if v:
        for w in v.warnings:
            print("  WARNING:", w)
        for q in v.questions:
            print(f"  QUESTION: {q.question}")
            for o in q.options:
                print(f"      - {o}")
        for e in v.errors:
            print("  ERROR:", e)
        if v.request:
            run_and_show(v.request, llm, question, explain)


if __name__ == "__main__":
    explain = "--explain" in sys.argv
    try:
        llm = create_llm_client()
    except LLMUnavailable as e:
        sys.exit(str(e))
    session = Session()
    print("Ask about your sales patterns, then refine (empty line to quit). /spec  /undo  /reset")
    while True:
        try:
            line = input("\n> ").strip()
        except EOFError:
            break
        if not line:
            break
        if line == "/spec":
            print("  current request:", json.dumps(session.current()) if session.spec else "none")
            continue
        if line in ("/undo", "/reset"):
            show_turn(undo_session(session) if line == "/undo" else reset_session(session), llm, explain, line)
            continue
        show_turn(converse(session, line, llm), llm, explain, line)
