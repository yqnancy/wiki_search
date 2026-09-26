"""Generate evals/latest_results.ipynb: the latest runs of both evals (2026-09-26).

    .venv/bin/python evals/build_latest_results.py
    .venv/bin/jupyter nbconvert --to notebook --execute --inplace evals/latest_results.ipynb

Reads saved run folders only; makes no API calls.
"""

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md("""# Latest eval results (2026-09-26)

The two use cases have separate evals. This notebook shows the latest runs of each; nothing here calls the API.

1. **Research briefs** (`evals/run_eval.py`, `evals/research_set.json`): the goal and reader-level items, re-run after the prompt and tooling fixes. Code graders only (brief format, freshness): the judged graders' packets are written but **not yet graded**. The two original research items were fully graded in the pilot, shown at the end of this section.
2. **Fact verification, ambiguity set** (`evals/run.py`, graded by hand in the Claude Code session): v7 (shorter answer line) against the v4 baseline, plus targeted re-runs of the fixes that followed.

For the full 250-item fact-verification suite (v3), see `evals/results.ipynb`; for the history and reasoning behind every change, `evals/SUMMARY.md`."""),
    code("""import json
from pathlib import Path

import pandas as pd
from IPython.display import Markdown, display

RUNS = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "evals" / "runs").is_dir()) / "evals" / "runs"
pd.set_option("display.max_colwidth", 90)


def jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]"""),
    md("""## 1. Research briefs

| Run | Items | Code state |
|---|---|---|
| `20260926-150450` | 6 goal items | Less strict claim check, lighter goal-section prompt, first table rule (which over-triggered) |
| `20260926-152603` | 8 reader-level items (4 beginner/expert pairs) | Plus the table fix: default no table, `table.purpose` required, send-back check |

Per item: prose words (overview + sections; recommended 300-500, fails above 625), claim-check warnings left after the one revision round, whether a table was used, the two code grades, cost and time."""),
    code("""def research_table(run):
    rows = []
    for r in jsonl(RUNS / run / "results.jsonl"):
        s, g = r.get("structured") or {}, r.get("grades", {})
        prose = sum(len(str(p.get("statement", "")).split()) for p in s.get("overview", []) + [q for sec in s.get("sections", []) for q in sec.get("points", [])])
        rows.append({
            "item": r["id"], "question": r["question"][:70],
            "prose words": prose, "claim warnings": len(r["warnings"]),
            "table": "yes" if (s.get("table") or {}).get("rows") else "",
            "brief_format": g.get("brief_format", {}).get("score"),
            "format failures": "; ".join(g.get("brief_format", {}).get("failures", [])),
            "freshness": g.get("freshness", {}).get("score"),
            "$": r["cost_usd"], "s": round(r["elapsed_s"]),
        })
    return pd.DataFrame(rows)

goal = research_table("20260926-150450_claude-sonnet-4-6")
level = research_table("20260926-152603_claude-sonnet-4-6")
display(Markdown("### Goal items")); display(goal)
display(Markdown("### Reader-level pairs")); display(level)"""),
    code("""both = pd.concat([goal.assign(set="goal"), level.assign(set="reader-level")])
summary = both.groupby("set").agg(items=("item", "count"), median_words=("prose words", "median"),
                                  over_625=("prose words", lambda w: int((w > 625).sum())),
                                  warnings_per_brief=("claim warnings", "mean"), tables=("table", lambda t: int((t == "yes").sum())),
                                  brief_format=("brief_format", "mean"), cost=("$", "sum"), mean_s=("s", "mean")).round(2)
summary"""),
    md("""**Reading the research results**

- **Depth follows the reader's level.** In every pair the expert brief is longer (557-681 vs 359-592 words), reads more sections and costs 2-4x more. Level fit itself is a judged grade, still pending.
- **The table fix works on these items.** Tables appear only in three expert briefs that compare methods or platforms. In the goal run, before the fix, all six goal briefs had one.
- **Expert briefs keep more unsupported claims** (4-6 claim-check warnings, against 0-2 for beginner briefs). Technical detail is harder to keep strictly within the cited text.
- **Goal sections are still advice-heavy** (e.g. "Focus first on pandas…"). The lighter-goal prompt didn't hold; a structural fix was proposed and deferred."""),
    md("### Pilot: the two original research items, fully graded (run `20260925-231835`, before later fixes)"),
    code("""pilot = [r for r in jsonl(RUNS / "20260925-231835_claude-sonnet-4-6" / "results.jsonl") if r["id"].startswith("research-")]
pd.DataFrame([{"item": r["id"], **{k: (v.get("score") if isinstance(v, dict) else None) for k, v in r["grades"].items()}} for r in pilot])"""),
    md("""## 2. Fact verification: ambiguity set

**v7** changed one thing from the shipped system: the `answer` field's description now asks for the bare answer (about 10 words at most), with the reading or scope, and extra facts, moved to the first reasoning item. Graded by hand under the strict footnote rule, like v4."""),
    code("""def amb(run):
    resp = {r["id"]: r for r in jsonl(RUNS / run / "responses.jsonl")}
    judg = {j["id"]: j for j in jsonl(RUNS / run / "judgments.jsonl")}
    return resp, judg

(r4, j4), (r7, j7) = amb("suite_v4_fix13"), amb("suite_v7_short_answer")

def answer_words(r):
    s = (r.get("turn1") or {}).get("structured") or {}
    return len(s["answer"].split()) if isinstance(s.get("answer"), str) else None

rows = []
for run, resp, judg in (("v4 (baseline)", r4, j4), ("v7 (short answer)", r7, j7)):
    words = pd.Series([answer_words(r) for r in resp.values()]).dropna()
    verdicts = pd.Series([j["verdict"] for j in judg.values()])
    rows.append({"run": run, "pass": int((verdicts == "pass").sum()), "partial": int((verdicts == "partial").sum()),
                 "fail": int((verdicts == "fail").sum()), "asked": sum(1 for r in resp.values() if (r.get("turn1") or {}).get("clarification")),
                 "median answer words": words.median(), "answers over 12 words": f"{(words > 12).mean():.0%}"})
pd.DataFrame(rows)"""),
    code("""changed = [{"item": i, "v4": j4[i]["verdict"], "v7": j7[i]["verdict"], "v7 failure mode": j7[i]["primary_failure_mode"],
            "v7 rationale": j7[i].get("rationale", "")[:160]}
           for i in sorted(j7) if j4.get(i, {}).get("verdict") != j7[i]["verdict"] or j7[i]["verdict"] != "pass"]
pd.DataFrame(changed)"""),
    md("""**Reading the ambiguity results**

- **Answer lines roughly halved** (median 17 to 9 words); pass rate 45/50 + 1 partial vs v4's 48/50.
- **Only amb-01t (Inter) plausibly traces to the shorter answer**: it gave one reading and never named the other. amb-11t (Go) and amb-18c (Panthers) trace to the question-only disambiguation check and to search-path variance; amb-22t (Masters) improved.
- **Run-to-run variance is large and unmeasured.** Items flipped between runs with identical code (amb-03t, amb-11t). No configuration has been run with repeats, so a 3-item swing on 50 cannot be told apart from noise.

### Follow-up fixes after v7 (targeted re-runs, not fully graded)

| Run | What changed | Items | Outcome |
|---|---|---|---|
| `suite_v7b_disamb_fix` | The disambiguation check gets the whole page as a compact list (the old 6,000-character cut hid "Go (programming language)", and hid entries on Mercury, Springfield, Panther and Master) | 6 long-page items | 6/6 pass, including amb-18c (was fail). Run directly, the check still resolves amb-11t's "Go" to the language: a calibration issue, not truncation |
| `suite_v7c_readings` | Readings rule: short answer line, one reasoning item per reading | 14 multi-reading items | Answer lines shorter on 7 of 12 answered items but still 13-28 words; exposed a truncation bug (amb-03t) |
| `suite_v7d_complete` | A submission cut off by the output limit, or missing reasoning or sources, is sent back instead of accepted | amb-03t | Complete answer, with reasoning and sources; this time it covered 2 of the 3 Rangers teams |"""),
]

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"name": "wiki-search", "display_name": "wiki_search (.venv)", "language": "python"}
nbf.write(nb, HERE / "latest_results.ipynb")
print("wrote", HERE / "latest_results.ipynb")
