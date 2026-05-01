"""Run the full sample=500 LoCoMo benchmark using the slice's date-injecting
ingestion. Saves per-question detail to benchmarks/results/locomo.v0.3.X.json
so we can quantify the date-injection win and inspect remaining failures.

Usage:
    FAIRNESS_ROOT=~/Projects/hermes-agent-benchmark-fairness \\
        python scripts/run_full_locomo_with_dates.py \\
            --sample 500 --label v0.3.dates --output benchmarks/results/locomo.v0.3.dates.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

FAIRNESS_ROOT = Path(os.environ.get(
    "FAIRNESS_ROOT", str(Path.home() / "Projects/hermes-agent-benchmark-fairness")
))
if str(FAIRNESS_ROOT) not in sys.path:
    sys.path.insert(0, str(FAIRNESS_ROOT))

# Reuse the slice's ingestion so the experiment matches. Loaded by file path
# because tests/eval_slice isn't a package (no __init__.py exposed at install).
import importlib.util  # noqa: E402

_slice_path = REPO / "tests" / "eval_slice" / "run_slice.py"
_spec = importlib.util.spec_from_file_location("eval_slice_run", _slice_path)
_slice_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_slice_mod)
_ingest_locomo_with_dates = _slice_mod._ingest_locomo_with_dates
_load_session_dates_by_conversation = _slice_mod._load_session_dates_by_conversation


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample", type=int, default=500)
    p.add_argument("--label", default="v0.3.dates")
    p.add_argument("--output", required=True, help="Path to write JSON results")
    p.add_argument("--profile", default="balanced")
    p.add_argument("--embedding", default="auto")
    args = p.parse_args()

    from benchmarks.backends.mnemoria_adapter import MnemoriaBenchmarkAdapter
    from benchmarks.judge import HeuristicJudge
    from benchmarks.locomo.adapter import load_locomo_dataset, evaluate_question

    questions = load_locomo_dataset(sample=args.sample)
    print(f"Loaded {len(questions)} questions")

    date_map = _load_session_dates_by_conversation()
    judge = HeuristicJudge()

    backend_kwargs = {"profile": args.profile, "embedding_model": args.embedding}

    by_type: dict[str, dict] = {}
    per_question = []
    correct = 0

    t0 = time.time()
    for i, q in enumerate(questions):
        store = MnemoriaBenchmarkAdapter(**backend_kwargs)
        store.reset()
        _ingest_locomo_with_dates(store, q, date_map.get(q.conversation_id, []))
        r = evaluate_question(store, q, judge, top_k=10)

        if r.correct:
            correct += 1

        bt = by_type.setdefault(q.question_type, {"total": 0, "correct": 0})
        bt["total"] += 1
        if r.correct:
            bt["correct"] += 1

        per_question.append({
            "question_id": q.question_id,
            "question_type": q.question_type,
            "question": q.question,
            "gold": q.answer,
            "recalled_top1": r.recalled,
            "context_top5": r.context,
            "correct": r.correct,
        })

        if (i + 1) % 25 == 0 or i == len(questions) - 1:
            elapsed = time.time() - t0
            print(
                f"  [{i+1}/{len(questions)}] correct={correct} "
                f"({correct/(i+1):.3f}) elapsed={elapsed:.1f}s"
            )

    wall = time.time() - t0
    for qt, v in by_type.items():
        v["score"] = v["correct"] / v["total"] if v["total"] else 0.0

    out = {
        "label": args.label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sample": args.sample,
        "profile": args.profile,
        "embedding_model": args.embedding,
        "wall_time_s": wall,
        "total": len(questions),
        "correct": correct,
        "score": correct / len(questions) if questions else 0.0,
        "by_type": by_type,
        "results": per_question,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print()
    print(f"=== {args.label} ({args.sample} questions) ===")
    print(f"  total       {correct}/{len(questions)} = {out['score']:.3f}")
    for qt, v in sorted(by_type.items()):
        print(f"  {qt:<20s} {v['correct']}/{v['total']} = {v['score']:.3f}")
    print(f"  wall: {wall:.1f}s -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
