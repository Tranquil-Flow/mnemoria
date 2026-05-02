"""Run the full LongMemEval benchmark using date-aware ingestion.

The standard adapter at hermes-agent-benchmark-fairness drops
``haystack_dates`` per session, blocking temporal-reasoning questions.
This script re-attaches them as session-anchor facts plus date suffixes
on turns containing relative-time markers.

Usage:
    FAIRNESS_ROOT=~/Projects/hermes-agent-benchmark-fairness \\
        python scripts/run_full_longmemeval_with_dates.py \\
            --sample 500 --output benchmarks/results/longmemeval.v0.3.dates.json
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

from tests.eval_slice.dated_ingestion import ingest_longmemeval_with_dates  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample", type=int, default=500)
    p.add_argument("--label", default="v0.3.dates")
    p.add_argument("--output", required=True)
    p.add_argument("--profile", default="balanced")
    p.add_argument("--embedding", default="auto")
    args = p.parse_args()

    from benchmarks.backends.mnemoria_adapter import MnemoriaBenchmarkAdapter
    from benchmarks.judge import HeuristicJudge
    from benchmarks.longmemeval.adapter import (
        load_longmemeval_dataset, evaluate_question,
    )

    questions = load_longmemeval_dataset(sample=args.sample)
    print(f"Loaded {len(questions)} questions")

    judge = HeuristicJudge()
    backend_kwargs = {"profile": args.profile, "embedding_model": args.embedding}

    by_type: dict = {}
    per_question = []
    correct = 0

    t0 = time.time()
    for i, q in enumerate(questions):
        store = MnemoriaBenchmarkAdapter(**backend_kwargs)
        store.reset()
        ingest_longmemeval_with_dates(store, q)
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
                f"({correct/(i+1):.3f}) elapsed={elapsed:.1f}s",
                flush=True,
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

    print(f"\n=== {args.label} ({args.sample} questions) ===")
    print(f"  total       {correct}/{len(questions)} = {out['score']:.3f}")
    for qt, v in sorted(by_type.items()):
        print(f"  {qt:<24s} {v['correct']}/{v['total']} = {v['score']:.3f}")
    print(f"  wall: {wall:.1f}s -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
