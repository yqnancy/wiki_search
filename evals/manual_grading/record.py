"""Append hand grades to a grades file, one pipe-separated line per item on stdin.

    .venv/bin/python -m evals.manual_grading.record --out grades/suite_v3.jsonl <<'EOF'
    scp-01t|answered|pass|none||Headline separates urban area from city proper.
    scp-10t|answered|partial|scope_unstated||Elbrus alone; no Caucasus caveat.
    EOF

Line format: id|behavior|verdict|primary_failure_mode|secondary,modes|rationale
(blank lines and lines starting with # are skipped). Values are checked against the
taxonomy here, so a typo fails at once rather than at conversion. Grading an item again
appends a new line; the last grade for an id wins when converting.

Rationale markers read by `to_judgments`: GOLD-DISPUTE (the gold looks wrong),
FOLLOWUP-FAIL / FOLLOWUP-PARTIAL (the answer after a clarification; pass if absent).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..taxonomy import BEHAVIORS, FAILURE_MODES


def parse(line: str) -> dict:
    item_id, behavior, verdict, primary, secondary, rationale = [p.strip() for p in line.split("|", 5)]
    grade = {"id": item_id, "behavior": behavior, "verdict": verdict, "primary_failure_mode": primary,
             "secondary_failure_modes": [m for m in secondary.split(",") if m], "rationale": rationale}
    check(grade)
    return grade


def check(g: dict) -> None:
    rid = g["id"]
    assert g["behavior"] in BEHAVIORS, (rid, g["behavior"])
    assert g["verdict"] in ("pass", "partial", "fail"), (rid, g["verdict"])
    assert g["primary_failure_mode"] in FAILURE_MODES, (rid, g["primary_failure_mode"])
    assert all(m in FAILURE_MODES for m in g["secondary_failure_modes"]), (rid, g["secondary_failure_modes"])
    assert (g["verdict"] == "pass") == (g["primary_failure_mode"] == "none"), f"{rid}: pass iff primary mode is none"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="grades JSONL file to append to")
    args = ap.parse_args()
    grades = [parse(l) for l in sys.stdin if l.strip() and not l.lstrip().startswith("#")]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        for g in grades:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    print(f"{len(grades)} grades appended to {out}")


if __name__ == "__main__":
    main()
