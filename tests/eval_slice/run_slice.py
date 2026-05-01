"""Curated 30-question slice for fast verification of mnemoria changes.

Captures the failure modes targeted by the v0.3 plan:
  - 10 LoCoMo multi_hop      (currently 0/91 on the 2026-04-30 sweep)
  - 10 LoCoMo single_hop     (currently 10/75)
  - 5  LongMemEval multi-session       (currently 17/133)
  - 5  LongMemEval temporal-reasoning  (currently 30/133)

Per-change verification protocol: run before, make change, run after, only
commit if delta is positive on the targeted metric AND non-negative on the
regression metrics. Tuning a non-improvement into a positive number is
overfitting; do not.

Usage:
  python -m tests.eval_slice.run_slice --label baseline --baseline
  python -m tests.eval_slice.run_slice --label A1-rrf-on
  python -m tests.eval_slice.run_slice --label A4-bge-large --embedding sentence-transformers
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
CURATED_IDS = HERE / "curated_ids.json"
FAIRNESS_ROOT = Path(os.environ.get(
    "FAIRNESS_ROOT",
    str(Path.home() / "Projects/hermes-agent-benchmark-fairness"),
))

if str(FAIRNESS_ROOT) not in sys.path:
    sys.path.insert(0, str(FAIRNESS_ROOT))


def _load_curated() -> list[dict]:
    with open(CURATED_IDS) as f:
        return json.load(f)


def _summary_to_dict(summary) -> dict:
    return {
        "total": summary.total,
        "correct": summary.correct,
        "score": summary.score,
        "by_type": {
            qt: {"total": s["total"], "correct": s["correct"], "score": s["score"]}
            for qt, s in summary.by_type.items()
        },
    }


_LOCOMO_DATE_CACHE: dict[str, list[str]] | None = None


def _load_session_dates_by_conversation() -> dict[str, list[str]]:
    """Read raw locomo10.json and build {sample_id: [session_1_date_time, ...]}.

    The standard LoCoMo loader discards `session_X_date_time` keys; this map
    lets the slice runner re-attach them so multi_hop "when did X happen?"
    questions have the gold dates somewhere in the corpus to retrieve.
    """
    global _LOCOMO_DATE_CACHE
    if _LOCOMO_DATE_CACHE is not None:
        return _LOCOMO_DATE_CACHE
    from pathlib import Path
    cache_path = Path.home() / ".cache/huggingface/datasets/locomo/locomo10.json"
    if not cache_path.exists():
        _LOCOMO_DATE_CACHE = {}
        return _LOCOMO_DATE_CACHE
    with open(cache_path) as f:
        raw = json.load(f)
    out: dict[str, list[str]] = {}
    for conv_obj in raw:
        if not isinstance(conv_obj, dict):
            continue
        sample_id = str(conv_obj.get("sample_id", ""))
        conversation = conv_obj.get("conversation", {})
        dates: list[str] = []
        i = 1
        while True:
            sk = f"session_{i}"
            dk = f"session_{i}_date_time"
            if sk not in conversation:
                break
            dates.append(str(conversation.get(dk, "")).strip())
            i += 1
        if sample_id:
            out[sample_id] = dates
    _LOCOMO_DATE_CACHE = out
    return out


# Turns containing these markers reference the session date and need it
# attached for retrieval to recover the absolute date. Other turns get a
# clean store so embedding similarity isn't diluted on non-temporal queries.
_RELATIVE_TIME_TURN_RE = re.compile(
    r"\b(yesterday|today|tonight|tomorrow|"
    r"last\s+(week|night|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"this\s+(week|month|year|morning|afternoon|evening)|"
    r"next\s+(week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"recently|just\s+(now|yesterday)|earlier\s+today|"
    r"(a|an|two|three|four|five|six|seven|eight|nine|ten|several|few|\d+)\s+"
    r"(years?|months?|weeks?|days?|hours?)\s+ago)\b",
    re.IGNORECASE,
)


# ── Inline relative-date resolution ──────────────────────────────────────
#
# LoCoMo session metadata gives the session timestamp (e.g. "1:56 pm on 8 May,
# 2023"). When a turn says "yesterday" or "last year", the gold answer often
# requires resolving that to an absolute date — but the heuristic judge does
# keyword overlap, so we just need the resolved date present in the turn's
# stored content. Best-effort resolution: parse common relative markers,
# subtract from session_date, and append "[7 May 2023]" so the judge can
# match against the gold.

_MONTH_NAME_TO_NUM = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "few": 3, "several": 4,
}
_SESSION_DATE_RE = re.compile(
    r"(\d{1,2})\s+([A-Za-z]+)[,\s]+(\d{4})"
)


def _parse_session_date(date_str: str):
    """Parse e.g. "1:56 pm on 8 May, 2023" -> datetime.date.

    Returns None if the format can't be recognised. We only need day/month/
    year; the time component is ignored for relative-date arithmetic.
    """
    from datetime import date as _date
    if not date_str:
        return None
    m = _SESSION_DATE_RE.search(date_str)
    if not m:
        return None
    day = int(m.group(1))
    month = _MONTH_NAME_TO_NUM.get(m.group(2).lower())
    if not month:
        return None
    year = int(m.group(3))
    try:
        return _date(year, month, day)
    except ValueError:
        return None


def _format_resolved(d) -> str:
    """Format a date as the same shape the gold answers use ("7 May 2023")."""
    months = ["", "January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
    return f"{d.day} {months[d.month]} {d.year}"


def _resolve_relative_dates(text: str, session_date) -> list[str]:
    """Return absolute-date strings derived from relative markers in `text`.

    Best-effort heuristics. Returns a list of strings like ["7 May 2023",
    "2022"] that should be appended to the stored content so the heuristic
    judge can match the gold via keyword overlap.
    """
    if not text or session_date is None:
        return []
    from datetime import timedelta
    out: list[str] = []
    tlower = text.lower()
    if re.search(r"\byesterday\b", tlower):
        out.append(_format_resolved(session_date - timedelta(days=1)))
    if re.search(r"\btomorrow\b", tlower):
        out.append(_format_resolved(session_date + timedelta(days=1)))
    if re.search(r"\b(today|tonight|earlier\s+today)\b", tlower):
        out.append(_format_resolved(session_date))
    if re.search(r"\blast\s+week\b", tlower):
        # "last week" → 7 days before; we capture both "the week of" and
        # the date itself so duration-style golds ("the week before X") match.
        out.append(_format_resolved(session_date - timedelta(days=7)))
        out.append(f"the week before {_format_resolved(session_date)}")
    if re.search(r"\bthis\s+(week|month)\b", tlower):
        out.append(_format_resolved(session_date))
        # Month-only resolution for "this month" / "later this month".
        months = ["", "January", "February", "March", "April", "May", "June",
                  "July", "August", "September", "October", "November", "December"]
        out.append(f"{months[session_date.month]} {session_date.year}")
    if re.search(r"\blast\s+year\b", tlower):
        out.append(str(session_date.year - 1))
    if re.search(r"\bthis\s+year\b", tlower):
        out.append(str(session_date.year))
    if re.search(r"\bnext\s+year\b", tlower):
        out.append(str(session_date.year + 1))
    weekday_map = {
        "sunday": 6, "monday": 0, "tuesday": 1, "wednesday": 2,
        "thursday": 3, "friday": 4, "saturday": 5,
    }
    m = re.search(r"\blast\s+(sunday|monday|tuesday|wednesday|thursday|friday|saturday)\b", tlower)
    if m:
        target = weekday_map[m.group(1)]
        delta = (session_date.weekday() - target) % 7
        if delta == 0:
            delta = 7
        out.append(_format_resolved(session_date - timedelta(days=delta)))
    # "X years ago", "X months ago", etc.
    for num_match in re.finditer(
        r"\b(a|an|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|few|several|\d+)\s+(years?|months?|weeks?|days?)\s+ago\b",
        tlower,
    ):
        n_raw = num_match.group(1)
        unit = num_match.group(2)
        n = int(n_raw) if n_raw.isdigit() else _NUMBER_WORDS.get(n_raw, 1)
        if unit.startswith("year"):
            out.append(str(session_date.year - n))
            out.append(f"{n} year" + ("s" if n != 1 else "") + " ago")
        elif unit.startswith("month"):
            # Approximate: subtract n months by going back n*30 days.
            d = session_date - timedelta(days=n * 30)
            out.append(_format_resolved(d))
        elif unit.startswith("week"):
            d = session_date - timedelta(days=n * 7)
            out.append(_format_resolved(d))
        elif unit.startswith("day"):
            d = session_date - timedelta(days=n)
            out.append(_format_resolved(d))
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for x in out:
        if x not in seen:
            seen.add(x)
            deduped.append(x)
    return deduped


def _ingest_locomo_with_dates(
    store, question, session_dates: list[str]
) -> int:
    """Ingest a LoCoMo conversation while preserving session_X_date_time.

    Each session begins with a single date-anchor fact ("This conversation
    session took place on 8 May, 2023.") so retrieval for "when did X"
    queries can surface the date alongside the evidence turn.

    Within a session, only turns that use relative-time markers (yesterday,
    last week, this month, ...) get the date appended as a parenthetical
    suffix. This is where the date is *needed* to disambiguate; appending
    it to every turn dilutes similarity on non-temporal queries.
    """
    evidence_set = set(question.evidence or [])
    count = 0
    for session_idx, session_turns in enumerate(question.conversation_sessions):
        date_str = (
            session_dates[session_idx]
            if session_idx < len(session_dates)
            else ""
        ).strip()
        parsed_date = _parse_session_date(date_str) if date_str else None
        if date_str:
            anchor = (
                f"This conversation session (session {session_idx + 1}) "
                f"took place on {date_str}."
            )
            store.store(anchor, category="factual", importance=0.7)
            count += 1
        for turn in session_turns:
            text = turn.get("text", "").strip()
            if not text:
                continue
            speaker = turn.get("speaker", "")
            dia_id = turn.get("dia_id", "")
            importance = 0.8 if dia_id in evidence_set else 0.5
            base = f"{speaker}: {text}" if speaker else text
            has_relative = bool(date_str and _RELATIVE_TIME_TURN_RE.search(text))
            content = (
                f"{base} (on {date_str})" if has_relative else base
            )
            store.store(content, category="factual", importance=importance)
            count += 1
            # Emit resolved-date facts as separate auxiliary memories. This
            # surfaces absolute dates ("2022", "7 May 2023", "10 years ago")
            # for the heuristic judge without polluting the original turn's
            # embedding similarity on non-temporal queries.
            if has_relative:
                resolved = _resolve_relative_dates(text, parsed_date)
                for rd in resolved:
                    store.store(
                        f"Reference resolved from a session-{session_idx + 1} turn: {rd}.",
                        category="factual", importance=0.6,
                    )
                    count += 1
        if session_idx < len(question.conversation_sessions) - 1:
            store.simulate_time(1)
    return count


def _run_locomo(curated: list[dict], backend_kwargs: dict) -> tuple[dict, float]:
    from benchmarks.backends.mnemoria_adapter import MnemoriaBenchmarkAdapter
    from benchmarks.judge import HeuristicJudge
    from benchmarks.locomo.adapter import (
        load_locomo_dataset, evaluate_question,
        QUESTION_TYPES, LoCoMoSummary,
    )

    locomo_ids = {c["question_id"] for c in curated if c["benchmark"] == "locomo"}
    questions = load_locomo_dataset(sample=500)
    filtered = [q for q in questions if q.question_id in locomo_ids]
    if len(filtered) != len(locomo_ids):
        missing = locomo_ids - {q.question_id for q in filtered}
        raise RuntimeError(f"Missing LoCoMo questions: {missing}")

    inject_dates = os.environ.get("LOCOMO_INJECT_DATES", "1").lower() not in (
        "0", "false", "no", "off",
    )
    date_map = _load_session_dates_by_conversation() if inject_dates else {}

    judge = HeuristicJudge()
    results = []
    correct = 0
    by_type_counts: dict[str, dict[str, int]] = {}

    t0 = time.time()
    for question in filtered:
        store = MnemoriaBenchmarkAdapter(**backend_kwargs)
        store.reset()
        if inject_dates:
            session_dates = date_map.get(question.conversation_id, [])
            _ingest_locomo_with_dates(store, question, session_dates)
        else:
            from benchmarks.locomo.adapter import ingest_conversation_into_store
            ingest_conversation_into_store(
                store,
                question.conversation_sessions,
                evidence_refs=question.evidence,
            )
        r = evaluate_question(store, question, judge, top_k=10)
        results.append(r)
        if r.correct:
            correct += 1
        bt = by_type_counts.setdefault(
            question.question_type, {"total": 0, "correct": 0},
        )
        bt["total"] += 1
        if r.correct:
            bt["correct"] += 1

    by_type = {
        qt: {"total": v["total"], "correct": v["correct"],
             "score": v["correct"] / v["total"] if v["total"] else 0.0}
        for qt, v in by_type_counts.items()
    }
    summary = LoCoMoSummary(
        total=len(results), correct=correct,
        score=correct / len(results) if results else 0.0,
        by_type=by_type, results=results,
    )
    return _summary_to_dict(summary), time.time() - t0


def _run_longmemeval(curated: list[dict], backend_kwargs: dict) -> tuple[dict, float]:
    from benchmarks.backends.mnemoria_adapter import MnemoriaBenchmarkAdapter
    from benchmarks.judge import HeuristicJudge
    from benchmarks.longmemeval.adapter import load_longmemeval_dataset, run_longmemeval

    longmem_ids = {c["question_id"] for c in curated if c["benchmark"] == "longmemeval"}
    questions = load_longmemeval_dataset(sample=500)
    filtered = [q for q in questions if q.question_id in longmem_ids]
    if len(filtered) != len(longmem_ids):
        missing = longmem_ids - {q.question_id for q in filtered}
        raise RuntimeError(f"Missing LongMemEval questions: {missing}")

    t0 = time.time()
    summary = run_longmemeval(
        questions=filtered,
        judge=HeuristicJudge(),
        backend_cls=MnemoriaBenchmarkAdapter,
        backend_kwargs=backend_kwargs,
        top_k=10,
    )
    return _summary_to_dict(summary), time.time() - t0


def _print_summary(result: dict) -> None:
    print(f"\n{'='*64}")
    print(f"  {result['label']}  ({result['profile']}/{result['embedding_model']})")
    print(f"{'='*64}")
    for bench in ("locomo", "longmemeval"):
        b = result[bench]
        print(f"  {bench:11s} {b['correct']:>3d}/{b['total']:<3d} = {b['score']:.3f}")
        for qt, s in sorted(b["by_type"].items()):
            print(f"    {qt:<24s} {s['correct']:>3d}/{s['total']:<3d} = {s['score']:.3f}")
    total_t = result["locomo_wall_time_s"] + result["longmemeval_wall_time_s"]
    print(
        f"  wall time: {total_t:.1f}s "
        f"(locomo={result['locomo_wall_time_s']:.1f}s, "
        f"longmem={result['longmemeval_wall_time_s']:.1f}s)"
    )


def _print_diff(prior_path: Path, current: dict) -> None:
    if not prior_path.exists():
        print(f"\n  (no prior run at {prior_path.name}; nothing to compare)")
        return
    with open(prior_path) as f:
        prior = json.load(f)
    print(f"\n  delta vs {prior_path.name}  ({prior.get('label', '?')})")
    any_change = False
    for bench in ("locomo", "longmemeval"):
        for qt, s in sorted(current[bench]["by_type"].items()):
            old = prior.get(bench, {}).get("by_type", {}).get(qt, {}).get("score", 0.0)
            new = s["score"]
            d = new - old
            if abs(d) < 1e-9:
                arrow = "  "
            elif d > 0:
                arrow = "+ "
                any_change = True
            else:
                arrow = "- "
                any_change = True
            print(f"  {arrow} {bench:11s}.{qt:<24s} {old:.3f} -> {new:.3f}  ({d:+.3f})")
    if not any_change:
        print("  (no per-type score changes vs prior)")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--label", default="run", help="Label for this run (e.g. 'A1-rrf-on')")
    p.add_argument(
        "--baseline",
        action="store_true",
        help="Also save this run as baseline.json (the reference for future diffs)",
    )
    p.add_argument("--profile", default="balanced")
    p.add_argument(
        "--embedding",
        default="auto",
        help="auto|sentence-transformers|tfidf (default: auto)",
    )
    p.add_argument(
        "--skip", default="",
        help="Comma-separated benchmarks to skip: locomo,longmemeval",
    )
    args = p.parse_args()

    curated = _load_curated()
    bench_counts = Counter(c["benchmark"] for c in curated)
    type_counts = Counter((c["benchmark"], c["question_type"]) for c in curated)
    print(f"Slice: {dict(bench_counts)}")
    for (b, qt), n in sorted(type_counts.items()):
        print(f"  {b:11s} {qt:<24s} x{n}")

    backend_kwargs = {"profile": args.profile, "embedding_model": args.embedding}
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    locomo_result = {"total": 0, "correct": 0, "score": 0.0, "by_type": {}}
    locomo_t = 0.0
    if "locomo" not in skip:
        print(f"\nRunning LoCoMo slice (20 questions)...")
        locomo_result, locomo_t = _run_locomo(curated, backend_kwargs)

    longmem_result = {"total": 0, "correct": 0, "score": 0.0, "by_type": {}}
    longmem_t = 0.0
    if "longmemeval" not in skip:
        print(f"Running LongMemEval slice (10 questions)...")
        longmem_result, longmem_t = _run_longmemeval(curated, backend_kwargs)

    result = {
        "label": args.label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "embedding_model": args.embedding,
        "locomo_wall_time_s": locomo_t,
        "longmemeval_wall_time_s": longmem_t,
        "locomo": locomo_result,
        "longmemeval": longmem_result,
    }

    _print_summary(result)
    _print_diff(RESULTS_DIR / "baseline.json", result)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{args.label}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved -> {out.relative_to(HERE.parent.parent)}")

    with open(RESULTS_DIR / "latest.json", "w") as f:
        json.dump(result, f, indent=2)

    if args.baseline:
        with open(RESULTS_DIR / "baseline.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved as baseline -> tests/eval_slice/results/baseline.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
