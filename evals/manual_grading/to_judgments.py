"""Convert a grades file into evals/runs/<run>/judgments.jsonl, in the judge's schema.

    .venv/bin/python -m evals.manual_grading.to_judgments --grades grades/suite_v3.jsonl --run suite_v3_sonnet46

The output has the same fields as `evals/judge.py`, so `evals.report`, the notebooks and
`evals.query_faithfulness` read hand-graded runs unchanged. By default every response in
the run must be graded; --partial writes only the graded ones (e.g. one eval set of a
multi-set run). Grounding and citation relevance weren't checked section by section when
grading by hand: they are "not_assessed" unless a citation failure mode was recorded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..dataset import load_items
from ..run import RUNS_DIR
from .record import check


def to_judgment(g: dict, eval_name: str) -> dict:
    why, sec = g["rationale"], g["secondary_failure_modes"]
    followup = "not_applicable"
    if g["behavior"] == "asked_clarification":
        followup = "fail" if "FOLLOWUP-FAIL" in why else "partial" if "FOLLOWUP-PARTIAL" in why else "pass"
    return {
        "id": g["id"], "eval": eval_name, "judge_model": "manual (Claude Code session)",
        "behavior": g["behavior"], "verdict": g["verdict"], "primary_failure_mode": g["primary_failure_mode"],
        "secondary_failure_modes": sec, "followup_verdict": followup,
        "grounding": "some_unsupported" if "unsupported_citation" in sec else "not_assessed",
        "citation_relevance": "some_irrelevant" if "irrelevant_citation" in sec else "not_assessed",
        "gold_disagreement": "GOLD-DISPUTE" in why, "rationale": why,
        "judge_input_tokens": 0, "judge_output_tokens": 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grades", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--partial", action="store_true", help="allow ungraded responses (they are left out)")
    args = ap.parse_args()

    items = {i["id"]: i for i in load_items()}
    grades = {}
    for line in Path(args.grades).read_text().splitlines():
        if line.strip():
            g = json.loads(line)
            check(g)
            grades[g["id"]] = g  # the last grade for an id wins, so regrades can be appended
    run_dir = RUNS_DIR / args.run
    order = [json.loads(l)["id"] for l in (run_dir / "responses.jsonl").read_text().splitlines() if l.strip()]
    missing = [rid for rid in order if rid not in grades]
    if missing and not args.partial:
        raise SystemExit(f"{len(missing)} responses ungraded (e.g. {missing[:5]}); grade them or pass --partial")

    out = [to_judgment(grades[rid], items[rid]["eval"]) for rid in order if rid in grades]
    with (run_dir / "judgments.jsonl").open("w") as f:
        for j in out:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} judgments to {run_dir / 'judgments.jsonl'}" + (f"; {len(missing)} ungraded left out" if missing else ""))


if __name__ == "__main__":
    main()
