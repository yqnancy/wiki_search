"""Generate evals/results.ipynb: latest results up front, earlier runs in appendices.

    .venv/bin/python evals/build_results_notebook.py
    cd evals && ../.venv/bin/jupyter nbconvert --to notebook --execute results.ipynb --output results.ipynb
"""

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
MAIN_RUN = "suite_v3_sonnet46"      # improved system on the current eval sets
BASE_RUN = "suite_v2_sonnet46"      # original system on the previous eval sets
BASE_FILES = {"general": "archive/general_v2.jsonl", "ambiguity": "archive/ambiguity_v1.jsonl",
              "scope": "archive/scope_v1.jsonl", "aliases": "archive/aliases_v1.jsonl",
              "false_premise": "archive/false_premise_v1.jsonl"}
MAIN_FINDINGS = HERE / "results_findings.md"
BASE_FINDINGS = HERE / "results_suite_v2_findings.md"

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(f"""
# WikiQA eval results

**Main results:** the improved system (`wiki_search/` after the fixes in `evals/SUMMARY.md`) on the current eval sets (general v3, ambiguity/scope/aliases/false-premise v2): one run of `claude-sonnet-4-6`, date pinned to 2026-09-25, 250 items. Run: `evals/runs/{MAIN_RUN}/`.

**Appendix A:** the original system on the previous eval sets (general v2, paired sets v1), the baseline that motivated the fixes. **Appendix B:** composition of the current eval sets.

All runs were graded by hand in a Claude Code session with the taxonomy in `evals/taxonomy.py` and the strict footnote rule (critical corrections/caveats must be in the headline or reasoning, not only in "Assumptions & gaps"). Pass rates count full passes only; CIs are Wilson 95% over items (single run, so they exclude run-to-run variance).
""")

code(f"""
import sys, json, warnings
from pathlib import Path
ROOT = Path.cwd() if (Path.cwd() / "evals").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT)); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import Patch
from IPython.display import display, Markdown
import evals
from evals import analysis as A
from evals.dataset import EVAL_NAMES
from evals.report import wilson
from evals.taxonomy import FAILURE_MODES
pd.set_option("display.max_colwidth", 160)
plt.rcParams.update({{"figure.facecolor": A.SURFACE, "font.size": 9}})
EVAL_LABELS = {{"general": "General", "ambiguity": "Ambiguity", "scope": "Scope", "aliases": "Aliases", "false_premise": "False premise"}}

main = A.load_runs(["{MAIN_RUN}"])
base_items = A.load_archived_items({BASE_FILES!r})
base = A.load_runs(["{BASE_RUN}"], items=base_items)
def responses(run):
    return {{json.loads(l)["id"]: json.loads(l) for l in (ROOT / "evals/runs" / run / "responses.jsonl").read_text().splitlines() if l.strip()}}
main_resp, base_resp = responses("{MAIN_RUN}"), responses("{BASE_RUN}")
print(f"main: {{len(main)}} graded items; baseline: {{len(base)}} graded items")
""")

code("""
def headline(df):
    rows = []
    for ev in EVAL_NAMES + ["all"]:
        d = df if ev == "all" else df[df["eval"] == ev]
        k, n = int(d["passed"].sum()), len(d); lo, hi = wilson(k, n)
        rows.append({"eval": EVAL_LABELS.get(ev, "All items"), "n": n, "pass": k, "partial": int((d.verdict == "partial").sum()),
                     "fail": int((d.verdict == "fail").sum()), "pass rate": k / n, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows).set_index("eval")

def answer_of(resp, rid):
    r = resp.get(rid, {}); t = r.get("turn2") or r.get("turn1") or {}
    if t.get("clarification"): return "[asked] " + t["clarification"]["question"][:160]
    s = t.get("structured") or {}
    txt = (s.get("verdict") or "") + " " + (s.get("answer") or t.get("text", "")[:200])
    return txt.strip()[:220]

def failure_bars(df, title):
    nf = df[~df["passed"]]
    if nf.empty: print("No failures."); return
    fm = nf["failure_mode"].value_counts()
    groups = sorted(nf["failure_group"].unique(), key=lambda g: -(nf["failure_group"] == g).sum())
    gcolor = dict(zip(groups, A.SERIES))
    fig, ax = plt.subplots(figsize=(9, 0.34 * len(fm) + 1.1))
    y = np.arange(len(fm))
    ax.barh(y, fm.values, color=[gcolor[FAILURE_MODES[m][0]] for m in fm.index], height=0.65)
    ax.set_yticks(y, fm.index); ax.invert_yaxis()
    for yi, v in zip(y, fm.values): ax.text(v + 0.05, yi, str(v), va="center", fontsize=8, color=A.TEXT_2)
    ax.legend(handles=[Patch(color=gcolor[g], label=g) for g in groups], frameon=False, loc="lower right", labelcolor=A.TEXT_2)
    A.style_axes(ax, title); ax.grid(axis="y", visible=False); ax.grid(axis="x", color=A.GRID)
    fig.tight_layout(); plt.show()

def failures_table(df, resp):
    nf = df[~df["passed"]]
    return nf.assign(answer=nf["id"].map(lambda r: answer_of(resp, r)))[
        ["eval", "id", "verdict", "failure_mode", "question", "answer", "rationale"]].sort_values(["eval", "failure_mode"]).reset_index(drop=True)
""")

# ------------------------------------------------------------------ main results
md("## 1. Headline: improved system on the current eval sets")
code("""
h = headline(main)
display(h.style.format({"pass rate": "{:.1%}", "ci_lo": "{:.0%}", "ci_hi": "{:.0%}"}))
fig, ax = plt.subplots(figsize=(9, 3.4))
x = np.arange(len(h))
ax.bar(x, h["pass rate"], color=A.SERIES[0], width=0.55)
ax.errorbar(x, h["pass rate"], yerr=[h["pass rate"] - h.ci_lo, h.ci_hi - h["pass rate"]], fmt="none", ecolor=A.TEXT_2, capsize=4, linewidth=1)
for xi, v, n in zip(x, h["pass rate"], h["n"]):
    ax.text(xi, 0.03, f"{v:.0%}\\nn={n}", ha="center", va="bottom", color="white", fontsize=8, fontweight="bold")
ax.set_xticks(x, h.index); ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
A.style_axes(ax, "Pass rate by eval set (95% Wilson CI)")
fig.tight_layout(); plt.show()
""")
code("""
pm = A.pair_metrics(main).set_index("eval").drop(columns=["model", "repeat"])
display(pm.style.format("{:.0%}").set_caption("Paired sets: trap pass (catches the problem), control pass (no over-triggering), pair accuracy"))
""")

md("## 2. Remaining failure modes")
code("""
failure_bars(main, f"Primary failure modes ({(~main.passed).sum()} non-passing of {len(main)})")
sec = pd.Series([m for ms in main["secondary"] for m in ms]).value_counts()
print("Secondary failure modes on all items:", sec.to_dict())
display(failures_table(main, main_resp))
""")

if MAIN_FINDINGS.exists():
    md(MAIN_FINDINGS.read_text())

md("""
## 3. Before and after

The baseline (original system, previous eval sets) next to the improved system on the current, harder sets. **Not a controlled comparison:** both the system and the sets changed. Per-category tables below show where the fixes helped and where the new sets probe new ground.
""")
code("""
cmp = pd.DataFrame({"Baseline: original system, previous sets": headline(base)["pass rate"],
                    "Improved system, current sets": headline(main)["pass rate"]})
display(cmp.style.format("{:.1%}"))
fig, ax = plt.subplots(figsize=(10, 3.6))
x = np.arange(len(cmp)); w = 0.36
for k, col in enumerate(cmp.columns):
    ax.bar(x + (k - 0.5) * w, cmp[col], width=w - 0.02, color=A.SERIES[k], label=col)
    for xi, v in zip(x + (k - 0.5) * w, cmp[col]): ax.text(xi, v + 0.01, f"{v:.0%}", ha="center", fontsize=8, color=A.TEXT_2)
ax.set_xticks(x, cmp.index); ax.set_ylim(0, 1.1)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
A.style_axes(ax, "Pass rate: baseline vs improved (different eval versions)")
ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12), labelcolor=A.TEXT_2)
fig.tight_layout(); plt.show()
fb = base[~base.passed]["failure_mode"].value_counts().rename("baseline")
fn = main[~main.passed]["failure_mode"].value_counts().rename("improved")
display(pd.concat([fb, fn], axis=1).fillna(0).astype(int).sort_values("baseline", ascending=False)
        .style.set_caption("Primary failure modes: counts per 250 items"))
""")

md("## 4. Breakdowns of the current run")
code("""
g = main[main["eval"] == "general"]
cat = g.groupby("category")["passed"].agg(["sum", "count"]); cat["rate"] = cat["sum"] / cat["count"]
display(cat.sort_values("rate").style.format({"rate": "{:.0%}"}).set_caption("General v3 by category"))
for ev in ["ambiguity", "scope", "aliases", "false_premise"]:
    d = main[main["eval"] == ev]
    t = d.groupby(["subtype", "role"])["passed"].mean().unstack("role")
    display(t.style.format("{:.0%}").set_caption(f"{EVAL_LABELS[ev]}: pass rate by subtype"))
""")
code("""
display(main.groupby("eval").agg(items=("id", "size"), cost_per_item=("cost", "mean"), tool_calls=("tool_calls", "mean"),
                                 wall_s=("wall_s", "mean"), asked=("asked", "mean"))
        .style.format({"cost_per_item": "${:.3f}", "tool_calls": "{:.1f}", "wall_s": "{:.0f}s", "asked": "{:.0%}"}))
print(f"System cost for the run: ${main['cost'].sum():.2f} (grading was manual, $0)")
""")

# ------------------------------------------------------------------ appendix A
md("""
---
# Appendix A: baseline run (original system, previous eval sets)

`evals/runs/suite_v2_sonnet46/`, graded under the same strict footnote rule (three items were re-graded from pass to partial when the rule was adopted). Item definitions are in `evals/data/archive/`.
""")
code("""
display(headline(base).style.format({"pass rate": "{:.1%}", "ci_lo": "{:.0%}", "ci_hi": "{:.0%}"}))
display(A.pair_metrics(base).set_index("eval").drop(columns=["model", "repeat"]).style.format("{:.0%}"))
failure_bars(base, f"Baseline primary failure modes ({(~base.passed).sum()} non-passing of {len(base)})")
display(failures_table(base, base_resp))
""")
if BASE_FINDINGS.exists():
    md("## Baseline findings (as written at the time)\n\n*Written before the strict footnote rule: the headline there (238/250) became 235/250 after fp-17t, gen-237 and scp-11t were regraded to partial.*\n\n" + BASE_FINDINGS.read_text().replace("## ", "### "))

# ------------------------------------------------------------------ appendix B
md("""
---
# Appendix B: composition of the current eval sets

How each set is built, from the item files and the `verified` blocks written by `evals/verify.py`.
""")
code("""
items = A.item_frame()
display(items.groupby(["eval", "role"], dropna=False).size().unstack(fill_value=0))
display(items[items["eval"] == "general"]["category"].value_counts().to_frame("general items"))
for ev in ["ambiguity", "scope", "aliases", "false_premise"]:
    display(items[(items["eval"] == ev) & (items["role"] == "trap")]["subtype"].value_counts().to_frame(f"{ev} traps"))
print("Alias traps by redirect status:", items[(items["eval"] == "aliases") & (items.role == "trap")]["alias_redirect"].value_counts().to_dict())
print("Answer location (all items):", items["answer_location"].value_counts().to_dict())
from evals.dataset import load_items
raw = load_items()
print("Items with as_of (time-sensitive):", sum("as_of" in i for i in raw))
print("Ambiguity traps whose disambiguation page is in the top 3 search results:",
      sum(bool(i.get("disambiguation_top3")) for i in raw if i["eval"] == "ambiguity" and i.get("role") == "trap"), "of 25")
""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = HERE / "results.ipynb"
nbf.write(nb, out)
print("wrote", out)
