"""Print a run's responses next to each item's gold and rubric, for grading by hand.

    .venv/bin/python -m evals.manual_grading.dump --run suite_v3_sonnet46 --eval scope [--start 0 --end 26]
    .venv/bin/python -m evals.manual_grading.dump --run suite_v6_final --compare suite_v4_fix13 --eval ambiguity

With --compare, each item shows both runs' answers under one gold, for grading two
versions of the system side by side. Long fields are cut to keep the dump readable.
"""

from __future__ import annotations

import argparse
import json

from ..dataset import load_items
from ..run import RUNS_DIR


def load_run(run: str) -> dict[str, dict]:
    path = RUNS_DIR / run / "responses.jsonl"
    return {json.loads(l)["id"]: json.loads(l) for l in path.read_text().splitlines() if l.strip()}


def show_turn(t: dict) -> str:
    """One turn's submission: headline, reasoning, assumptions, sources and warnings."""
    if t.get("clarification"):
        c = t["clarification"]
        return f"[ASKED] {c['question'][:300]} | options={c['options']}"
    s = t.get("structured") or {}
    if "answer" not in s:  # research brief, or prose after a failed submit
        return "[REPORT/TEXT] " + (t.get("text") or "")[:900]
    verdict = f"{s['verdict']} | " if s.get("verdict") else ""
    out = [f"{s.get('question_type')}: {verdict}{s.get('answer')}"]
    out += [f"  - {r['statement']} {r.get('citations')}" for r in s.get("reasoning") or []]
    out += [f"  * {a}" for a in s.get("assumptions") or []]
    out.append("  src: " + "; ".join(f"{x['title']} § {x['section']}" for x in s.get("sources") or []))
    if any(e["type"] == "promote" for e in t.get("trace", [])):
        out.append("  (caveat retry fired)")
    if t.get("warnings"):
        out.append(f"  !! warnings: {t['warnings']}")
    return "\n".join(out)


def show_response(r: dict | None) -> str:
    if not r:
        return "!! NO RESPONSE"
    if r.get("error"):
        return f"!! ERROR {r['error'][:200]}"
    out = [show_turn(r["turn1"])]
    if r.get("turn2"):
        out += [f"REPLY: {r['clarification_reply']}", show_turn(r["turn2"])]
    steps = [s for s in r["turn1"]["trace"] if s["type"] in ("search", "disambiguation")]
    trace = " / ".join(f"q={s['query']!r}" if s["type"] == "search"
                       else f"disamb {s['page']}:{'AMBIG' if s.get('ambiguous') else 'resolved'}" for s in steps)
    out.append("  trace: " + trace[:300])
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--eval", required=True, help="one eval set, e.g. ambiguity")
    ap.add_argument("--compare", help="a second run to show under each item")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    args = ap.parse_args()

    runs = [args.run] + ([args.compare] if args.compare else [])
    recs = {run: load_run(run) for run in runs}
    for it in load_items([args.eval])[args.start:args.end]:
        print(f"### {it['id']} [{it['expected_behavior']}] {it.get('category') or it.get('subtype')} {it.get('role') or ''}")
        print(f"Q: {it['question']}")
        print(f"GOLD: {it['gold_answer']}")
        if it.get("gold_answer_after_clarification"):
            print(f"GOLD-AFTER: {it['gold_answer_after_clarification']}")
        if it.get("rubric"):
            print(f"RUBRIC: {it['rubric']}")
        for run in runs:
            if len(runs) > 1:
                print(f"<{run}>")
            print(show_response(recs[run].get(it["id"])))
        print()


if __name__ == "__main__":
    main()
