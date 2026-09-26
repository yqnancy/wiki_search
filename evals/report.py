"""Summarize a judged run: pass rates with 95% CIs, pair metrics, slices, failure modes.

Scoring: pass = 1, partial = 0.5, fail = 0 for the "score" column; "pass rate" counts
only full passes. Paired evals also report:
    trap pass       how often the system handled the trap correctly (recall)
    control pass    how often it answered the matched control correctly
                    (1 - control pass is the false-trigger rate)
    pair accuracy   both halves of a pair passed

    .venv/bin/python -m evals.report --name baseline      # prints and writes report.md
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict

from .dataset import EVAL_NAMES, load_items
from .run import RUNS_DIR
from .taxonomy import FAILURE_MODES

SCORE = {"pass": 1.0, "partial": 0.5, "fail": 0.0}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate(rows: list[dict]) -> str:
    n = len(rows)
    if not n:
        return "–"
    k = sum(r["verdict"] == "pass" for r in rows)
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.0%} [{lo:.0%}–{hi:.0%}]"


def score(rows: list[dict]) -> str:
    return f"{sum(SCORE[r['verdict']] for r in rows) / len(rows):.2f}" if rows else "–"


def table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def slice_table(rows: list[dict], key: str, title: str) -> str:
    groups = defaultdict(list)
    for r in rows:
        groups[r.get(key)].append(r)
    body = [[k, len(v), rate(v), score(v)] for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))]
    return f"**By {title}**\n\n" + table([title, "n", "pass rate [95% CI]", "score"], body)


def failure_table(rows: list[dict]) -> str:
    counts = Counter(r["primary_failure_mode"] for r in rows if r["verdict"] != "pass")
    secondary = Counter(m for r in rows for m in r.get("secondary_failure_modes", []))
    if not counts and not secondary:
        return "_No failures._"
    body = [[FAILURE_MODES[m][0], m, counts.get(m, 0), secondary.get(m, 0)]
            for m in sorted(set(counts) | set(secondary), key=lambda m: (-counts.get(m, 0), m))]
    return table(["group", "failure mode", "primary", "secondary"], body)


def pair_metrics(rows: list[dict]) -> str:
    traps = [r for r in rows if r.get("role") == "trap"]
    controls = [r for r in rows if r.get("role") == "control"]
    pairs = defaultdict(dict)
    for r in rows:
        if r.get("pair_id"):
            pairs[r["pair_id"]][r["role"]] = r
    complete = [p for p in pairs.values() if "trap" in p and "control" in p]
    both = sum(p["trap"]["verdict"] == "pass" and p["control"]["verdict"] == "pass" for p in complete)
    lo, hi = wilson(both, len(complete))
    body = [["trap pass (recall)", rate(traps)], ["control pass", rate(controls)],
            ["pair accuracy", f"{both}/{len(complete)} = {both / max(1, len(complete)):.0%} [{lo:.0%}–{hi:.0%}]"]]
    return table(["metric", "value"], body)


def behavior_matrix(rows: list[dict]) -> str:
    expected = sorted({r["expected_behavior"] for r in rows})
    actual = sorted({r["behavior"] for r in rows})
    body = [[e] + [sum(r["expected_behavior"] == e and r["behavior"] == a for r in rows) for a in actual]
            for e in expected]
    return "**Expected vs actual behavior**\n\n" + table(["expected \\ actual"] + actual, body)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    run_dir = RUNS_DIR / args.name
    judgments = {j["id"]: j for j in (json.loads(l) for l in (run_dir / "judgments.jsonl").read_text().splitlines() if l.strip())}
    responses = {r["id"]: r for r in (json.loads(l) for l in (run_dir / "responses.jsonl").read_text().splitlines() if l.strip())}
    rows = []
    for item in load_items():
        if item["id"] not in judgments:
            continue
        v = item.get("verified", {})
        rec = responses.get(item["id"], {})
        usage = [rec.get(t, {}).get("usage", {}) for t in ("turn1", "turn2")]
        rows.append({
            **{k: item.get(k) for k in ("id", "eval", "role", "pair_id", "subtype", "expected_behavior",
                                        "category", "language", "subtle")},
            "answer_location": v.get("answer_location"), "popularity": v.get("popularity"),
            "alias_redirect": (v.get("alias_redirect") or "").split(":")[0] or None,
            **judgments[item["id"]],
            "llm_calls": sum(u.get("llm_calls", 0) for u in usage),
            "wall_s": rec.get("wall_s", 0),
        })

    out = [f"# Eval report: {args.name}\n"]
    config = run_dir / "config.json"
    if config.exists():
        out.append("```\n" + config.read_text() + "\n```\n")

    overview = []
    for name in EVAL_NAMES:
        r = [x for x in rows if x["eval"] == name]
        if r:
            overview.append([name, len(r), rate(r), score(r), sum(x["gold_disagreement"] for x in r),
                             f"{sum(x['wall_s'] for x in r) / len(r):.0f}s"])
    out.append("## Overview\n\n" + table(["eval", "n", "pass rate [95% CI]", "score", "gold disputes", "mean wall"], overview))

    for name in EVAL_NAMES:
        r = [x for x in rows if x["eval"] == name]
        if not r:
            continue
        out.append(f"\n## {name}\n")
        if name == "general":
            for key, title in [("category", "category"), ("language", "language"), ("popularity", "popularity")]:
                out.append(slice_table(r, key, title) + "\n")
        else:
            out.append(pair_metrics(r) + "\n")
            out.append(slice_table([x for x in r if x["role"] == "trap"], "subtype", "trap subtype") + "\n")
            out.append(slice_table([x for x in r if x["role"] == "control"], "subtype", "control subtype") + "\n")
            out.append(behavior_matrix(r) + "\n")
        if name == "ambiguity":
            passed = [x for x in r if x["role"] == "trap" and x["verdict"] == "pass"]
            mix = Counter(x["behavior"] for x in passed)
            follow = [x for x in r if x["followup_verdict"] != "not_applicable"]
            out.append(f"Trap passes by strategy: {dict(mix)}. "
                       f"Post-clarification answers: {Counter(x['followup_verdict'] for x in follow)}.\n")
        if name == "aliases":
            out.append(slice_table([x for x in r if x["role"] == "trap"], "alias_redirect", "alias redirect") + "\n")
        if name == "false_premise":
            out.append(slice_table([x for x in r if x["role"] == "trap"], "subtle", "subtle trap") + "\n")
        out.append("**Failure modes**\n\n" + failure_table(r) + "\n")
        out.append("**Grounding:** " + ", ".join(f"{k} {v}" for k, v in Counter(x["grounding"] for x in r).most_common()) + "\n")
        relevance = Counter(x["citation_relevance"] for x in r if "citation_relevance" in x)  # absent in older judgments
        if relevance:
            out.append("**Citation relevance:** " + ", ".join(f"{k} {v}" for k, v in relevance.most_common()) + "\n")

    disputes = [x for x in rows if x["gold_disagreement"]]
    if disputes:
        out.append("\n## Gold answers to review\n\n" + "\n".join(f"- `{x['id']}`: {x['rationale']}" for x in disputes))
    failures = [x for x in rows if x["verdict"] != "pass"]
    if failures:
        out.append("\n## All non-passing items\n\n" + table(
            ["id", "verdict", "behavior", "failure mode", "rationale"],
            [[x["id"], x["verdict"], x["behavior"], x["primary_failure_mode"], x["rationale"].replace("|", "/").replace("\n", " ")]
             for x in failures]))

    text = "\n".join(out)
    (run_dir / "report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
