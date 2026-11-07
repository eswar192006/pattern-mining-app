"""Run Step 13 evaluation phases and build report-ready CSVs, Markdown and HTML charts.

Examples (from the project root):
  python -m evaluation.run --phase templates
  python -m evaluation.run --phase understanding --limit 10
  python -m evaluation.run --phase hybrid
  python -m evaluation.run --phase baseline
  python -m evaluation.run --phase benchmarks
  python -m evaluation.run --phase errors
  python -m evaluation.run --phase report
  python -m evaluation.run --phase all

Gemini calls are cached by full prompt/schema in .evaluation_cache/ and paced. The all phase can
take a while and use quota on its first run; later runs replay cached model calls.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import plotly.express as px
import pandas as pd

from src.core import config as C
from src.core.errors import LLMUnavailable
from src.core.request import MiningRequest
from src.mining.engine import EngineResult
from src.llm.client import GeminiClient
from src.llm.conversation import Session, converse
from src.llm.explainer import build_facts
from src.llm.numbercheck import check_text
from src.llm.prompts import PROMPT_VERSION
from src.mining.data_loader import load_clean
from src.mining.engine import run_mining
from src.core.pipeline import Pipeline, RESULT_KEYS

EVAL = ROOT / "evaluation"
QUESTIONS = EVAL / "questions.jsonl"
RESULTS = EVAL / "results"
CACHE_DIR = ROOT / ".evaluation_cache"
LLM_CACHE = CACHE_DIR / "model_calls.json"
ORACLE_CACHE = CACHE_DIR / "mining_oracles.json"
STOPWATCH = RESULTS / "stopwatch_study.csv"
BASELINE_REVIEW = RESULTS / "baseline_review.csv"
PROMPT_REVIEW = RESULTS / "prompt_changes.csv"
BASELINE_SAMPLE = RESULTS / "baseline_sample.json"


def read_cases(limit: int | None = None) -> list[dict]:
    cases = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Case ids in questions.jsonl must be unique")
    if limit is None and not 40 <= len(cases) <= 60:
        raise ValueError(f"The evaluation set must contain 40 to 60 cases, found {len(cases)}")
    question_count = sum(len(case.get("steps", [])) for case in cases)
    if limit is None and question_count > 60:
        raise ValueError(f"The full evaluation set must contain at most 60 question turns, found {question_count}")
    for case in cases:
        if not case.get("steps"):
            raise ValueError(f"{case['id']} has no steps")
        for step in case["steps"]:
            if step["status"] == "ok" and "request" not in step:
                raise ValueError(f"{case['id']} has an ok step without a gold request")
    return cases[:limit] if limit else cases


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def canonical_request(request: dict | None) -> dict | None:
    if request is None:
        return None
    return MiningRequest.model_validate(request).model_dump(exclude_defaults=True)


def _flatten(value, prefix="") -> dict[str, object]:
    out = {}
    if isinstance(value, dict):
        for key, item in value.items():
            out.update(_flatten(item, f"{prefix}.{key}" if prefix else key))
    else:
        out[prefix] = value
    return out


def _stable(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


class CachedPacedLLM:
    """Drop-in Gemini wrapper: cache identical model calls and pace only uncached calls."""

    def __init__(self, client, cache: dict, pace_s: float):
        self.client, self.cache, self.pace_s = client, cache, max(0.0, pace_s)
        self.model = client.model
        self.last_call = 0.0
        self.cache_hits = 0
        self.new_calls = 0

    def _call(self, mode, system, user, schema=None):
        signature = {"model": self.model, "mode": mode, "system": system, "user": user, "schema": schema}
        key = hashlib.sha256(_stable(signature).encode()).hexdigest()
        if key in self.cache:
            self.cache_hits += 1
            entry = self.cache[key]
            if "error" in entry:
                raise RuntimeError(entry["error"])
            return entry["value"]
        delay = self.pace_s - (time.monotonic() - self.last_call)
        if self.last_call and delay > 0:
            time.sleep(delay)
        self.last_call = time.monotonic()
        self.new_calls += 1
        value = self.client.generate_json(system, user, schema) if mode == "json" else self.client.generate_text(system, user)
        self.cache[key] = {"value": value}
        write_json(LLM_CACHE, self.cache)
        return value

    def generate_json(self, system, user, schema=None):
        return self._call("json", system, user, schema)

    def generate_text(self, system, user):
        return self._call("text", system, user)


def _gold(step):
    return canonical_request(step.get("request")) if step.get("request") is not None else None


def _field_stats(expected, actual):
    e, a = _flatten(expected or {}), _flatten(actual or {})
    keys = sorted(set(e) | set(a))
    return {key: (e.get(key) == a.get(key)) for key in keys}


def _result_without_time(result):
    if result is None:
        return None
    return {k: result.get(k) for k in RESULT_KEYS if k != "timings"}


def _oracle(request: dict, df, cache: dict):
    key = hashlib.sha256((_stable(request) + str(C.CLEAN_FILE.stat().st_mtime_ns)).encode()).hexdigest()
    if key not in cache:
        cache[key] = run_mining(request, df=df).to_dict()
        write_json(ORACLE_CACHE, cache)
    return {k: cache[key].get(k) for k in RESULT_KEYS if k != "timings"}


def _row(case, index, step, reply, oracle_cache, df, mode):
    expected_status = step["status"]
    expected_req = _gold(step)
    actual_status = reply.get("status")
    actual_req = canonical_request(reply.get("request"))
    validation = reply.get("questions") or []
    actual_clarification = validation[0].get("field") if validation else None
    expected_clarification = step.get("clarification_field")
    fields = _field_stats(expected_req, actual_req) if expected_req is not None else {}
    req_correct = (actual_req == expected_req) if expected_req is not None else None

    result_correct = None
    number_ok = None
    if mode == "hybrid" and expected_req is not None:
        actual_result = _result_without_time(reply.get("result"))
        if actual_result is None or actual_req is None or actual_req != expected_req:
            result_correct = False
        else:
            expected_result = _oracle(expected_req, df, oracle_cache)
            result_correct = actual_result == expected_result
        if reply.get("answer") and reply.get("result") and actual_req:
            data = reply["result"]
            facts_result = EngineResult(
                kind=data["kind"], request=MiningRequest.model_validate(actual_req),
                itemsets=pd.DataFrame(data.get("itemsets") or []),
                n_baskets_selected=data.get("n_baskets_selected", 0), n_baskets_mined=data.get("n_baskets_mined", 0),
                total_found=data.get("total_found", 0), warnings=data.get("warnings") or [],
                notes=data.get("notes") or [], hint=data.get("hint"), params=data.get("params") or {},
            )
            facts = build_facts(facts_result, question=step["message"])
            number_ok = not check_text(reply["answer"].get("text", ""), facts.allowed, facts.names)

    timings = reply.get("timings") or {}
    out = {
        "case_id": case["id"], "group": case["group"], "step": index + 1, "mode": mode,
        "prompt_version": PROMPT_VERSION,
        "question": step["message"], "expected_status": expected_status, "actual_status": actual_status,
        "status_correct": expected_status == actual_status,
        "expected_clarification_field": expected_clarification, "actual_clarification_field": actual_clarification,
        "clarification_correct": (expected_clarification == actual_clarification) if expected_clarification else None,
        "request_correct": req_correct, "expected_request": _stable(expected_req) if expected_req is not None else "",
        "actual_request": _stable(actual_req) if actual_req is not None else "",
        "result_correct": result_correct, "answer_source": (reply.get("answer") or {}).get("source", ""),
        "numbers_traceable": number_ok, "language_s": timings.get("language_s"),
        "mining_s": timings.get("mining_s"), "explain_s": timings.get("explain_s"),
        "total_s": timings.get("total_s"), "field_matches": _stable(fields),
        "answer": (reply.get("answer") or {}).get("text", ""),
        "errors": _stable(reply.get("message") or reply.get("errors") or reply.get("error") or []),
    }
    return out


def _save_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.unlink(missing_ok=True)
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _append_review_template(path: Path, rows: list[dict], cols: list[str], key_fields=None):
    if path.exists():
        if rows:
            with path.open(encoding="utf-8-sig", newline="") as f:
                existing = list(csv.DictReader(f))
            key_fields = tuple(key_fields or (("case_id",) if "case_id" in cols else (cols[0],)))
            row_key = lambda row: tuple(row.get(field, "") for field in key_fields)
            by_key = {row_key(row): row for row in existing}
            for incoming in rows:
                current = by_key.get(row_key(incoming), {col: "" for col in cols})
                for col in cols:
                    value = incoming.get(col, "")
                    if value and (not current.get(col) or col in ("question", "answer")):
                        current[col] = value
                by_key[row_key(incoming)] = current
            _save_csv(path, list(by_key.values()))
        return
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=cols).writeheader()
        return
    _save_csv(path, [{**{k: row.get(k, "") for k in cols}, **{c: "" for c in cols if c not in row}}
                     for row in rows])


def _data_hash():
    return hashlib.sha256(C.CLEAN_FILE.read_bytes()).hexdigest() if C.CLEAN_FILE.exists() else "missing"


def run_model_phase(phase: str, cases, pace_s: float, log_name: str):
    if not C.CLEAN_FILE.exists():
        raise SystemExit("Clean data is missing. Run `python -m src.prepare_data` first.")
    df = load_clean()
    cache = read_json(LLM_CACHE, {})
    llm = CachedPacedLLM(GeminiClient(), cache, pace_s)
    oracle_cache = read_json(ORACLE_CACHE, {})
    rows = []
    old_log = C.LLM_LOG_FILE
    C.LLM_LOG_FILE = RESULTS / log_name
    stopped = ""
    stem = "request_accuracy" if phase == "understanding" else "hybrid_results"
    try:
        pipeline = Pipeline(llm, df)
        for case_no, case in enumerate(cases, 1):
            print(f"[{case_no}/{len(cases)}] {case['id']} ({case['group']})", flush=True)
            session = Session()
            for step_no, step in enumerate(case["steps"]):
                if phase == "understanding":
                    turn = converse(session, step["message"], llm, df=df, log_path=C.LLM_LOG_FILE)
                    validation = turn.validation
                    reply = {"status": turn.status,
                             "request": validation.request.model_dump(exclude_defaults=True) if validation and validation.request else None,
                             "questions": [question.to_dict() for question in validation.questions] if validation else [],
                             "timings": {"language_s": turn.latency_s, "total_s": turn.latency_s},
                             "errors": validation.errors if validation else [], "message": turn.message}
                else:
                    reply = pipeline.chat(session, step["message"], explain=True)
                rows.append(_row(case, step_no, step, reply, oracle_cache, df, phase))
                _save_csv(RESULTS / f"{stem}.csv", rows)
                if reply.get("status") == "llm_unavailable":
                    stopped = reply.get("message") or "Gemini was unavailable; remaining cases were not run."
                    break
            if stopped:
                break
    finally:
        C.LLM_LOG_FILE = old_log
    _save_csv(RESULTS / f"{stem}.csv", rows)
    expected_steps = sum(len(case["steps"]) for case in cases)
    write_json(RESULTS / f"{stem}_status.json", {"cases": len(cases), "expected_steps": expected_steps,
                                                   "evaluated_steps": len(rows), "complete": not stopped and len(rows) == expected_steps,
                                                   "stopped_reason": stopped})
    print(f"Saved {len(rows)} step results to evaluation/results/{stem}.csv")
    return rows, llm, bool(stopped)


def _make_baseline_sample(df):
    basket_ids = sorted(df["basket_id"].unique(), key=lambda x: str(x))
    chosen = set(random.Random(2027).sample(basket_ids, min(30, len(basket_ids))))
    sample = []
    for basket_id, group in df[df["basket_id"].isin(chosen)].groupby("basket_id", sort=False):
        sample.append({"order": str(basket_id), "products": group["item"].tolist(),
                       "sales_revenue": round(float(group["utility"].sum()), 2)})
    return sample


def _baseline_prompt(sample, case):
    user_messages = "\n".join(f"Turn {i + 1}: {s['message']}" for i, s in enumerate(case["steps"]))
    return ("You are a retail sales analyst. Answer the user's question using ONLY this small transaction sample. "
            "Do not imply it represents all orders. Clearly say when the sample is too small to support a conclusion. "
            "Do not invent figures or product names.\n\nSAMPLE TRANSACTIONS (30 orders):\n"
            + json.dumps(sample, ensure_ascii=False) + "\n\nUSER QUESTION / CONVERSATION:\n" + user_messages)


def run_baseline(cases, pace_s):
    if not C.CLEAN_FILE.exists():
        raise SystemExit("Clean data is missing. Run `python -m src.prepare_data` first.")
    df = load_clean()
    sample = _make_baseline_sample(df)
    write_json(BASELINE_SAMPLE, sample)
    cache = read_json(LLM_CACHE, {})
    llm = CachedPacedLLM(GeminiClient(), cache, pace_s)
    rows = []
    stopped = ""
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] baseline {case['id']}", flush=True)
        prompt = _baseline_prompt(sample, case)
        try:
            answer = llm.generate_text("Answer from the provided sample, with an explicit sample-size caveat.", prompt)
            status = "ok"
        except LLMUnavailable as exc:
            answer, status = str(exc), "llm_unavailable"
            stopped = str(exc)
        rows.append({"case_id": case["id"], "group": case["group"], "status": status,
                     "question": " / ".join(s["message"] for s in case["steps"]),
                     "sample_orders": len(sample), "answer": answer, "model": llm.model})
        _save_csv(RESULTS / "baseline_answers.csv", rows)
        if stopped:
            break
    _save_csv(RESULTS / "baseline_answers.csv", rows)
    write_json(RESULTS / "baseline_status.json", {"cases": len(cases), "evaluated_cases": len(rows),
                                                    "complete": not stopped and len(rows) == len(cases),
                                                    "stopped_reason": stopped})
    review = [{"case_id": r["case_id"], "question": r["question"], "answer": r["answer"],
               "reviewer": "", "answer_correct": "", "all_numbers_supported": "", "notes": ""}
              for r in rows if r["status"] == "ok"]
    _append_review_template(BASELINE_REVIEW, review, ["case_id", "question", "answer", "reviewer",
                                                       "answer_correct", "all_numbers_supported", "notes"])
    print(f"Saved baseline answers and review sheet ({len(rows)} cases)")
    return rows, llm, bool(stopped)


def run_benchmarks(fractions, repeats):
    if not C.CLEAN_FILE.exists():
        raise SystemExit("Clean data is missing. Run `python -m src.prepare_data` first.")
    df = load_clean()
    unique = sorted(df["basket_id"].unique(), key=lambda x: str(x))
    rng = random.Random(2027)
    shuffled = unique[:]
    rng.shuffle(shuffled)
    experiments = [
        ("frequent", {"kind": "frequent", "min_support": 0.02, "algorithm": "fpgrowth"}),
        ("frequent", {"kind": "frequent", "min_support": 0.02, "algorithm": "apriori"}),
        ("rare", {"kind": "rare", "max_support": 0.005, "min_count": 10, "min_strength": 0.3, "max_len": 2}),
        ("high_utility", {"kind": "high_utility", "min_utility_pct": 0.002, "max_len": 2, "algorithm": "fhm+"}),
    ]
    rows = []
    for fraction in fractions:
        n = max(2, min(len(unique), int(len(unique) * fraction)))
        picked = set(shuffled[:n])
        sample = df[df["basket_id"].isin(picked)].copy()
        for kind, request in experiments:
            for repeat in range(1, repeats + 1):
                start = time.perf_counter()
                result = run_mining(request, df=sample)
                elapsed = time.perf_counter() - start
                rows.append({"kind": kind, "algorithm": request.get("algorithm") or "default",
                             "data_fraction": fraction, "baskets": n, "rows": len(sample),
                             "repeat": repeat, "elapsed_s": round(elapsed, 4),
                             "results_found": result.total_found, "results_shown": len(result.itemsets),
                             "machine": platform.platform(), "python": platform.python_version(),
                             "processor": platform.processor()})
                print(f"{kind}/{request.get('algorithm') or 'default'} at {n:,} baskets: {elapsed:.2f}s", flush=True)
    _save_csv(RESULTS / "algorithm_benchmarks.csv", rows)
    return rows


def _percent(n, d):
    return 100 * n / d if d else None


def update_prompt_comparisons():
    """Fill the after column only from successful evaluation rows using the current prompt version."""
    if not PROMPT_REVIEW.exists() or not (RESULTS / "request_accuracy.csv").exists():
        return
    with PROMPT_REVIEW.open(encoding="utf-8-sig", newline="") as f:
        changes = list(csv.DictReader(f))
    with (RESULTS / "request_accuracy.csv").open(encoding="utf-8-sig", newline="") as f:
        evaluated = list(csv.DictReader(f))
    touched = False
    for change in changes:
        if change.get("after_cases"):
            continue
        ids = {value.strip() for value in change.get("matched_case_ids", "").split(",") if value.strip()}
        matches = [row for row in evaluated if row.get("case_id") in ids
                   and row.get("prompt_version") == PROMPT_VERSION
                   and row.get("actual_status") != "llm_unavailable"
                   and row.get("expected_status") == "ok"]
        if matches:
            change["after_cases"] = str(len(matches))
            change["after_correct"] = str(sum(row.get("request_correct") == "True" for row in matches))
            touched = True
    if touched:
        _save_csv(PROMPT_REVIEW, changes)


def summarize_results():
    RESULTS.mkdir(parents=True, exist_ok=True)
    update_prompt_comparisons()
    summaries = []
    by_group = defaultdict(list)
    field_acc = defaultdict(lambda: [0, 0])
    stage_values = defaultdict(list)
    for filename in ("request_accuracy.csv", "hybrid_results.csv"):
        path = RESULTS / filename
        if not path.exists():
            continue
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        evaluated = [r for r in rows if r.get("actual_status") != "llm_unavailable"]
        valid = [r for r in evaluated if r.get("expected_status") == "ok"]
        exact = sum(r.get("request_correct") == "True" for r in valid)
        status_ok = sum(r.get("status_correct") == "True" for r in evaluated)
        clarification_tp = sum(r.get("expected_status") == "needs_clarification" and r.get("actual_status") == "needs_clarification" for r in evaluated)
        clarification_fp = sum(r.get("expected_status") != "needs_clarification" and r.get("actual_status") == "needs_clarification" for r in evaluated)
        clarification_fn = sum(r.get("expected_status") == "needs_clarification" and r.get("actual_status") != "needs_clarification" for r in evaluated)
        status_path = RESULTS / filename.replace(".csv", "_status.json")
        run_state = read_json(status_path, {})
        if not run_state and rows:
            case_ids = {r["case_id"] for r in rows}
            case_lookup = {case["id"]: case for case in read_cases()}
            expected_steps = sum(len(case_lookup[cid]["steps"]) for cid in case_ids if cid in case_lookup)
            run_state = {"cases": len(case_ids), "expected_steps": expected_steps,
                         "evaluated_steps": len(rows), "complete": False,
                         "stopped_reason": "This partial output predates run-status tracking; unavailable turns are not accuracy results."}
        summary = {"source": filename.removesuffix(".csv"), "steps": len(rows),
                   "expected_steps": run_state.get("expected_steps", len(rows)),
                   "evaluated_steps": len(evaluated), "unavailable_steps": len(rows) - len(evaluated),
                   "complete": run_state.get("complete", False),
                   "stopped_reason": run_state.get("stopped_reason", ""),
                   "status_accuracy_pct": _percent(status_ok, len(evaluated)),
                   "request_exact_accuracy_pct": _percent(exact, len(valid)),
                   "clarification_cases": sum(bool(r.get("expected_clarification_field")) for r in evaluated),
                   "clarification_accuracy_pct": _percent(sum(r.get("clarification_correct") == "True" for r in evaluated),
                                                            sum(bool(r.get("expected_clarification_field")) for r in evaluated)),
                   "clarification_decision_accuracy_pct": _percent(
                       sum((r.get("actual_status") == "needs_clarification") == (r.get("expected_status") == "needs_clarification") for r in evaluated), len(evaluated)),
                   "clarification_precision_pct": _percent(clarification_tp, clarification_tp + clarification_fp),
                   "clarification_recall_pct": _percent(clarification_tp, clarification_tp + clarification_fn),
                   "result_checked": 0, "result_correct_pct": None,
                   "answers_checked": 0, "numbers_traceable_pct": None}
        if filename == "hybrid_results.csv":
            result_rows = [r for r in valid if r.get("result_correct") in ("True", "False")]
            number_rows = [r for r in evaluated if r.get("numbers_traceable") in ("True", "False")]
            summary["result_checked"] = len(result_rows)
            summary["answers_checked"] = len(number_rows)
            summary["result_correct_pct"] = _percent(sum(r["result_correct"] == "True" for r in result_rows), len(result_rows))
            summary["numbers_traceable_pct"] = _percent(sum(r["numbers_traceable"] == "True" for r in number_rows), len(number_rows))
        summaries.append(summary)
        for row in rows:
            by_group[(filename.removesuffix(".csv"), row["group"])].append(row)
            if row.get("actual_status") == "llm_unavailable":
                continue
            for key in ("language_s", "mining_s", "explain_s", "total_s"):
                try:
                    if row.get(key):
                        stage_values[(filename.removesuffix(".csv"), key)].append(float(row[key]))
                except ValueError:
                    pass
            try:
                fields = json.loads(row.get("field_matches") or "{}")
                for key, ok in fields.items():
                    field_acc[(filename.removesuffix(".csv"), key)][0] += bool(ok)
                    field_acc[(filename.removesuffix(".csv"), key)][1] += 1
            except json.JSONDecodeError:
                pass
    _save_csv(RESULTS / "summary.csv", summaries)
    group_rows = []
    for (source, group), rows in sorted(by_group.items()):
        evaluated = [r for r in rows if r.get("actual_status") != "llm_unavailable"]
        valid = [r for r in evaluated if r.get("expected_status") == "ok"]
        group_rows.append({"source": source, "group": group, "steps": len(rows), "evaluated_steps": len(evaluated),
                           "status_accuracy_pct": _percent(sum(r.get("status_correct") == "True" for r in evaluated), len(evaluated)),
                           "request_exact_accuracy_pct": _percent(sum(r.get("request_correct") == "True" for r in valid), len(valid))})
    _save_csv(RESULTS / "accuracy_by_group.csv", group_rows)
    field_rows = [{"source": source, "field": key, "correct": val[0], "total": val[1], "accuracy_pct": _percent(*val)}
                  for (source, key), val in sorted(field_acc.items())]
    _save_csv(RESULTS / "accuracy_by_field.csv", field_rows)
    time_rows = [{"source": source, "stage": stage, "n": len(vals), "mean_s": statistics.mean(vals),
                  "median_s": statistics.median(vals), "p95_s": sorted(vals)[min(len(vals)-1, math.ceil(.95*len(vals))-1)]}
                 for (source, stage), vals in sorted(stage_values.items()) if vals]
    _save_csv(RESULTS / "timing_summary.csv", time_rows)

    charts = []
    if any(r["request_exact_accuracy_pct"] is not None for r in group_rows):
        fig = px.bar(group_rows, x="group", y="request_exact_accuracy_pct", color="source", barmode="group",
                     title="Request accuracy by question type", labels={"request_exact_accuracy_pct": "Exact request accuracy (%)"})
        fig.write_html(RESULTS / "accuracy_by_group.html", include_plotlyjs="cdn")
        charts.append("accuracy_by_group.html")
    else:
        (RESULTS / "accuracy_by_group.html").unlink(missing_ok=True)
    if field_rows:
        fig = px.bar(field_rows, x="field", y="accuracy_pct", color="source", barmode="group", title="Request accuracy by field",
                     labels={"accuracy_pct": "Field accuracy (%)"})
        fig.update_layout(xaxis_tickangle=-35)
        fig.write_html(RESULTS / "accuracy_by_field.html", include_plotlyjs="cdn")
        charts.append("accuracy_by_field.html")
    else:
        (RESULTS / "accuracy_by_field.html").unlink(missing_ok=True)
    if time_rows:
        fig = px.bar(time_rows, x="stage", y="mean_s", color="source", barmode="group",
                     title="Mean response time by stage", labels={"mean_s": "Mean seconds"})
        fig.write_html(RESULTS / "timings.html", include_plotlyjs="cdn")
        charts.append("timings.html")
    else:
        (RESULTS / "timings.html").unlink(missing_ok=True)
    bench_path = RESULTS / "algorithm_benchmarks.csv"
    bench_rows = []
    if bench_path.exists():
        bench = pd.read_csv(bench_path)
        bench_rows = [{"kind": kind, "algorithm": algorithm, "baskets": int(baskets),
                       "data_fraction": float(fraction), "runs": len(group),
                       "mean_s": float(group["elapsed_s"].mean()),
                       "std_s": float(group["elapsed_s"].std(ddof=0))}
                      for (kind, algorithm, baskets, fraction), group in
                      bench.groupby(["kind", "algorithm", "baskets", "data_fraction"], sort=True)]
        _save_csv(RESULTS / "algorithm_benchmark_summary.csv", bench_rows)
        fig = px.line(bench, x="baskets", y="elapsed_s", color="algorithm", symbol="kind", markers=True,
                      title="Algorithm runtime as data size grows", labels={"elapsed_s": "Seconds"})
        fig.write_html(RESULTS / "algorithm_scaling.html", include_plotlyjs="cdn")
        charts.append("algorithm_scaling.html")
    else:
        (RESULTS / "algorithm_scaling.html").unlink(missing_ok=True)
        (RESULTS / "algorithm_benchmark_summary.csv").unlink(missing_ok=True)

    lines = ["# Evaluation report", "", "Generated by `python -m evaluation.run --phase report`.",
             "Gold questions are hand-labeled in `evaluation/questions.jsonl`; the runner does not generate them.", ""]
    for summary in summaries:
        lines += [f"## {summary['source'].replace('_', ' ').title()}", "",
                  f"- Steps evaluated: {summary['evaluated_steps']} / {summary.get('expected_steps', summary['steps'])}",
                  f"- Run complete: {'yes' if summary['complete'] else 'no'}",
                  f"- Unavailable calls: {summary['unavailable_steps']}",
                  f"- Status accuracy: {summary['status_accuracy_pct']:.1f}%" if summary["status_accuracy_pct"] is not None else "- Status accuracy: n/a",
                  f"- Exact request accuracy: {summary['request_exact_accuracy_pct']:.1f}%" if summary["request_exact_accuracy_pct"] is not None else "- Exact request accuracy: n/a",
                  f"- Clarification field accuracy on positive cases: {summary['clarification_accuracy_pct']:.1f}%" if summary["clarification_accuracy_pct"] is not None else "- Clarification field accuracy on positive cases: n/a",
                  f"- Ask/don't-ask decision accuracy across all questions: {summary['clarification_decision_accuracy_pct']:.1f}%" if summary["clarification_decision_accuracy_pct"] is not None else "- Ask/don't-ask decision accuracy: n/a",
                  f"- Clarification precision / recall: {summary['clarification_precision_pct']:.1f}% / {summary['clarification_recall_pct']:.1f}%"
                  if summary["clarification_precision_pct"] is not None and summary["clarification_recall_pct"] is not None
                  else "- Clarification precision / recall: n/a"]
        if summary["stopped_reason"]:
            lines.append(f"- Stopped reason: {summary['stopped_reason']}")
        elif summary["unavailable_steps"]:
            lines.append("- Some model calls were unavailable; these turns are excluded from accuracy rates.")
        if summary["result_correct_pct"] is not None:
            lines += [f"- Result correctness against the gold-request mining oracle: {summary['result_correct_pct']:.1f}% ({summary['result_checked']} checked)",
                      f"- Answers passing the data-backed number check: {summary['numbers_traceable_pct']:.1f}% ({summary['answers_checked']} checked)" if summary["numbers_traceable_pct"] is not None else "- Number traceability: n/a"]
        lines.append("")
    if field_rows:
        lines += ["## Request accuracy by field", "", "| Source | Field | Correct | Cases | Accuracy |", "|---|---|---:|---:|---:|"]
        lines += [f"| {r['source']} | {r['field']} | {r['correct']} | {r['total']} | {r['accuracy_pct']:.1f}% |" for r in field_rows]
        lines.append("")
    if time_rows:
        lines += ["## Stage timing", "", "| Source | Stage | Mean s | Median s | 95th percentile s |", "|---|---|---:|---:|---:|"]
        lines += [f"| {r['source']} | {r['stage']} | {r['mean_s']:.2f} | {r['median_s']:.2f} | {r['p95_s']:.2f} |" for r in time_rows]
        lines.append("")
    if bench_rows:
        lines += ["## Algorithm benchmarks", "", "Mean runtime over repeated runs on this machine (seconds):", "",
                  "| Kind | Algorithm | Baskets | Mean s | Std. dev. s | Runs |", "|---|---|---:|---:|---:|---:|"]
        lines += [f"| {r['kind']} | {r['algorithm']} | {r['baskets']:,} | {r['mean_s']:.3f} | {r['std_s']:.3f} | {r['runs']} |" for r in bench_rows]
        lines += ["", "Thresholds were held constant across data sizes. Results describe this dataset and machine; they are not cross-machine speed claims.", ""]
    if charts:
        lines += ["## Charts", ""] + [f"- [{name}]({name})" for name in charts] + [""]
    lines += ["## Human studies and direct-answer baseline", "",
              "The single-participant stopwatch study has five manual/app pairs in `stopwatch_study.csv`. Blank review fields are intentionally excluded from accuracy claims.",
              "Review `baseline_review.csv` against the full dataset before reporting direct-Gemini baseline accuracy.", ""]
    if BASELINE_REVIEW.exists():
        with BASELINE_REVIEW.open(encoding="utf-8-sig", newline="") as f:
            reviews = list(csv.DictReader(f))
        reviewed = [r for r in reviews if r.get("answer_correct", "").strip().lower() in ("true", "false", "yes", "no")]
        supported = [r for r in reviews if r.get("all_numbers_supported", "").strip().lower() in ("true", "false", "yes", "no")]
        if reviewed or supported:
            yn = lambda value: value.strip().lower() in ("true", "yes")
            lines += [f"Baseline human review: {sum(yn(r['answer_correct']) for r in reviewed)}/{len(reviewed)} answers marked correct; "
                      f"{sum(yn(r['all_numbers_supported']) for r in supported)}/{len(supported)} marked fully number-supported.", ""]
    if (RESULTS / "baseline_answers.csv").exists():
        baseline_status = read_json(RESULTS / "baseline_status.json", {})
        lines += [f"Direct-answer baseline responses collected: {baseline_status.get('evaluated_cases', 0)} / {baseline_status.get('cases', 0)} cases.",
                  "Baseline accuracy is intentionally unreported until the human review sheet is completed.", ""]
    if STOPWATCH.exists():
        with STOPWATCH.open(encoding="utf-8-sig", newline="") as f:
            study = list(csv.DictReader(f))
        observed = defaultdict(list)
        unanswered = []
        for row in study:
            outcome_note = (row.get("notes", "") + " " + row.get("time_minutes", "")).lower()
            if row.get("method") == "app" and any(token in outcome_note
                                                    for token in ("no answer", "no response", "timed out", "did not show")):
                unanswered.append(row)
            try:
                if row.get("time_minutes", "").strip():
                    observed[row["method"]].append(float(row["time_minutes"]))
            except (ValueError, KeyError):
                continue
        observed = {method: values for method, values in observed.items() if values}
        if observed:
            lines += ["Stopwatch study averages (minutes):"]
            lines += [f"- {method}: {statistics.mean(values):.2f} across {len(values)} completed observations"
                      for method, values in sorted(observed.items())]
            lines.append("")
        if unanswered:
            lines += [f"App stopwatch attempts with no answer: {len(unanswered)}. These are recorded as incomplete, not as successful response times.", ""]
    if PROMPT_REVIEW.exists():
        with PROMPT_REVIEW.open(encoding="utf-8-sig", newline="") as f:
            changes = [r for r in csv.DictReader(f) if r.get("change_id", "").strip()]
        if changes:
            lines += ["## Prompt changes: before and after", "", "| Change | Matched case(s) | Before correct / cases | After correct / cases |", "|---|---|---:|---:|"]
            for row in changes:
                before = f"{row.get('before_correct','?')} / {row.get('before_cases','?')}"
                after = (f"{row.get('after_correct','?')} / {row.get('after_cases','?')}"
                         if row.get("after_cases") else "pending")
                lines.append(f"| {row['change_id']} | {row.get('matched_case_ids','')} | {before} | {after} |")
                lines.append(f"\n**{row['change_id']} change:** {row.get('prompt_change','')}\n\n"
                             f"**Logged issue:** {row.get('error_pattern_from_log','')}\n\n"
                             f"**Notes:** {row.get('notes','')}")
            lines.append("")
        else:
            lines += ["## Prompt changes: before and after", "",
                      "No prompt changes or before/after claims are recorded yet. Use error_cases.csv to identify a recurring failure, change the prompt, then add matched before/after cases to prompt_changes.csv.", ""]
    (RESULTS / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote summary tables and report to {RESULTS}")


def make_templates(cases):
    RESULTS.mkdir(parents=True, exist_ok=True)
    cases = read_cases()
    qids = ["F02", "R03", "H04", "F09", "H02"]
    rows = []
    for qid in qids:
        case = next(c for c in cases if c["id"] == qid)
        for method in ("manual_tuning", "app"):
            rows.append({"participant": "self", "case_id": qid, "question": case["steps"][-1]["message"],
                         "method": method, "time_minutes": "", "request_correct": "", "results_correct": "", "notes": ""})
    stopwatch_columns = ["participant", "case_id", "question", "method", "time_minutes", "request_correct", "results_correct", "notes"]
    if STOPWATCH.exists():
        with STOPWATCH.open(encoding="utf-8-sig", newline="") as f:
            old_rows = list(csv.DictReader(f))
        old_values = {}
        for old in old_rows:
            key = (old.get("case_id"), old.get("method"))
            values = old_values.setdefault(key, {})
            for column in ("time_minutes", "request_correct", "results_correct", "notes"):
                if old.get(column, "").strip() and not values.get(column):
                    values[column] = old[column]
        for row in rows:
            row.update(old_values.get((row["case_id"], row["method"]), {}))
    _save_csv(STOPWATCH, rows)
    baseline = [{"case_id": case["id"], "question": " / ".join(step["message"] for step in case["steps"]),
                 "answer": "", "reviewer": "", "answer_correct": "", "all_numbers_supported": "", "notes": ""}
                for case in cases]
    _append_review_template(BASELINE_REVIEW, baseline, ["case_id", "question", "answer", "reviewer",
                                                         "answer_correct", "all_numbers_supported", "notes"])
    prompt_changes = [{"change_id": "P01", "date": "2026-11-07",
                       "error_pattern_from_log": "The model returned top_k=0 for a question with no requested result count; self-correction returned top_k=1, still an unrequested field.",
                       "prompt_change": "Clarify in the prompt that top_k is omitted unless requested and must be a positive integer (never zero).",
                       "matched_case_ids": "F01", "before_cases": "1", "before_correct": "0",
                       "after_cases": "", "after_correct": "",
                       "notes": "Before value is from the historical log. After measurement awaits a successful Gemini rerun."}]
    _append_review_template(PROMPT_REVIEW, prompt_changes, ["change_id", "date", "error_pattern_from_log", "prompt_change",
                                                             "matched_case_ids", "before_cases", "before_correct",
                                                             "after_cases", "after_correct", "notes"])
    print(f"Created/kept study templates in {RESULTS}")


def analyze_logs():
    paths = [ROOT / "logs" / "llm_requests.jsonl"]
    paths += sorted(RESULTS.glob("evaluation_llm_*.jsonl"))
    rows, counts = [], Counter()
    for path in paths:
        if not path.exists():
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = record.get("type", "parse")
            status = record.get("status", record.get("validation_status", "unknown"))
            problems = record.get("problems") or []
            category = "ok"
            if kind == "explanation":
                category = "number_check_or_answer" if problems else ("template_fallback" if record.get("source") == "template" else "ok")
            elif status in ("failed", "invalid", "llm_unavailable"):
                category = str(status)
            elif problems:
                category = "self_correction"
            counts[(path.name, kind, category)] += 1
            rows.append({"log_file": path.name, "line": line_no, "type": kind, "status": status,
                         "category": category, "question": record.get("question") or record.get("user_message") or "",
                         "problems": _stable(problems), "model": record.get("model", ""),
                         "prompt_version": record.get("prompt_version", "legacy/unversioned")})
    _save_csv(RESULTS / "error_cases.csv", [r for r in rows if r["category"] != "ok"])
    _save_csv(RESULTS / "error_summary.csv", [{"log_file": k[0], "type": k[1], "category": k[2], "count": v}
                                               for k, v in sorted(counts.items())])
    print(f"Analyzed {len(rows)} log entries; review error_cases.csv and prompt_changes.csv for fixes and before/after evidence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("templates", "understanding", "hybrid", "baseline", "benchmarks", "errors", "report", "all"), default="templates")
    parser.add_argument("--limit", type=int, help="Run only the first N gold cases (useful for a small pilot)")
    parser.add_argument("--case-ids", help="Comma-separated case ids for a focused pilot, for example F01,R01,H01,V01,C01,O01,U01,Q01")
    parser.add_argument("--pace-seconds", type=float, default=4.0, help="Minimum gap between uncached Gemini calls")
    parser.add_argument("--benchmark-fractions", default="0.25,0.5,1.0")
    parser.add_argument("--benchmark-repeats", type=int, default=2)
    args = parser.parse_args()
    if args.limit and args.case_ids:
        parser.error("Use either --limit or --case-ids, not both")
    RESULTS.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cases = read_cases(args.limit)
    if args.case_ids:
        wanted = {value.strip() for value in args.case_ids.split(",") if value.strip()}
        all_cases = read_cases()
        unknown = wanted - {case["id"] for case in all_cases}
        if unknown:
            parser.error("Unknown case ids: " + ", ".join(sorted(unknown)))
        cases = [case for case in all_cases if case["id"] in wanted]
    print(f"Loaded {len(cases)} hand-labeled cases; dataset sha256={_data_hash()[:12]}")

    if args.phase in ("templates", "all"):
        make_templates(cases)
    llm_stopped = False
    if args.phase in ("understanding", "hybrid", "all"):
        for phase in (("understanding", "hybrid") if args.phase == "all" else (args.phase,)):
            _, _, phase_stopped = run_model_phase(phase, cases, args.pace_seconds, f"evaluation_llm_{phase}.jsonl")
            llm_stopped = llm_stopped or phase_stopped
            if phase_stopped:
                break
    if args.phase == "baseline" or (args.phase == "all" and not llm_stopped):
        _, _, baseline_stopped = run_baseline(cases, args.pace_seconds)
        llm_stopped = llm_stopped or baseline_stopped
    elif args.phase == "all" and llm_stopped:
        print("Skipping remaining Gemini phases after an unavailable/rate-limited call; cached results are preserved.")
    if args.phase in ("benchmarks", "all"):
        fractions = [float(x) for x in args.benchmark_fractions.split(",")]
        run_benchmarks(fractions, args.benchmark_repeats)
    if args.phase in ("errors", "all"):
        analyze_logs()
    if args.phase in ("report", "all", "understanding", "hybrid", "baseline", "benchmarks", "errors"):
        summarize_results()


if __name__ == "__main__":
    main()
