"""Ask questions in plain English. The configured model translates them; everything else is exact.

    python -m src.demo_parse "Which products sell together on Sundays?"
    python -m src.demo_parse "Biggest earners in Germany" --run     # also mine and show results
    python -m src.demo_parse "What sells together on Sundays?" --explain   # ... and write the answer in plain English
    python -m src.demo_parse                                         # interactive: type questions, empty line to quit

Needs: Ollama running with the configured model (or Gemini configured in .env), and cleaned data.
--run on a high-utility question also needs Java and tools/spmf.jar. --explain implies --run.
"""
import sys
import textwrap

from src.core.errors import LLMUnavailable, MiningError
from src.demo_engine import print_result
from src.llm.client import create_llm_client
from src.llm.explainer import explain_result
from src.llm.parser import parse_question
from src.mining.engine import run_mining


def show(question, llm, run, explain=False):
    r = parse_question(question, llm)
    print(f"\nSTATUS: {r.status}   (attempts: {r.attempts}, {r.latency_s:.1f}s, model: {r.model})")
    if r.message:
        print("  ", r.message)
    v = r.validation
    if v:
        for n in v.notes:
            print("  understood:", n)
        for w in v.warnings:
            print("  WARNING:", w)
        for q in v.questions:
            print(f"  QUESTION: {q.question}")
            for o in q.options:
                print(f"      - {o}")
        for e in v.errors:
            print("  ERROR:", e)
        if v.request:
            print("  REQUEST:", v.request.model_dump_json(exclude_defaults=True))
            if run:
                print()
                try:
                    result = run_mining(v.request)
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
                    for p in e.problems:
                        print("  [rejected model answer:", p + "]")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a not in ("--run", "--explain")]
    explain = "--explain" in sys.argv
    run = "--run" in sys.argv or explain
    try:
        llm = create_llm_client()
    except LLMUnavailable as e:
        sys.exit(str(e))
    if args:
        show(" ".join(args), llm, run, explain)
    else:
        print("Type a question (empty line to quit).")
        while True:
            try:
                q = input("\n> ").strip()
            except EOFError:
                break
            if not q:
                break
            show(q, llm, run, explain)
