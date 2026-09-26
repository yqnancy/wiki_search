"""Load judged runs into DataFrames and compute the metrics used by analysis.ipynb."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import EVAL_NAMES, load_items
from .run import RUNS_DIR
from .taxonomy import FAILURE_MODES

SCORE = {"pass": 1.0, "partial": 0.5, "fail": 0.0}

# $ per million tokens: (input, output). Cache reads bill at 0.1x input, cache writes at 1.25x.
PRICES = {
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-opus-5": (5.0, 25.0),
}
MODEL_LABELS = {"sonnet46": "Sonnet 4.6", "haiku45": "Haiku 4.5"}

# Reference palette (dataviz skill, light mode): fixed slot order, never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MODEL_COLORS = {"Sonnet 4.6": SERIES[0], "Haiku 4.5": SERIES[1]}
TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def load_archived_items(files: dict) -> list:
    """Items from specific data files, e.g. {"general": "archive/general_v2.jsonl"} (paths relative to evals/data)."""
    from .dataset import DATA_DIR
    out = []
    for ev, rel in files.items():
        for line in (DATA_DIR / rel).read_text().splitlines():
            if line.strip():
                it = json.loads(line)
                it.setdefault("eval", ev)
                it.setdefault("expected_behavior", "answer")
                it["eval"] = ev
                out.append(it)
    return out


def item_frame(items: list = None) -> pd.DataFrame:
    rows = []
    for i in (items if items is not None else load_items()):
        v = i.get("verified", {})
        rows.append({
            "id": i["id"], "eval": i["eval"], "role": i.get("role"), "pair_id": i.get("pair_id"),
            "subtype": i.get("subtype"), "expected_behavior": i["expected_behavior"],
            "category": i.get("category"), "language": i.get("language"), "as_of": i.get("as_of"),
            "subtle": i.get("subtle"), "answer_location": v.get("answer_location"),
            "popularity": v.get("popularity"), "alias_redirect": (v.get("alias_redirect") or "").split(":")[0] or None,
            "question": i["question"],
        })
    return pd.DataFrame(rows)


def _system_cost(usage: dict, model: str) -> float:
    pin, pout = PRICES.get(model, (0, 0))
    return (usage.get("input_tokens", 0) * pin + usage.get("cache_read_input_tokens", 0) * pin * 0.1
            + usage.get("cache_creation_input_tokens", 0) * pin * 1.25 + usage.get("output_tokens", 0) * pout) / 1e6


def load_runs(model_dirs=("sonnet46", "haiku45"), judgments_file="judgments.jsonl", items: list = None) -> pd.DataFrame:
    """One row per (model, repeat, item) that has a judgment."""
    items = item_frame(items).set_index("id")
    rows = []
    for mdir in model_dirs:
        base = RUNS_DIR / mdir
        run_dirs = [base] if (base / "responses.jsonl").exists() else sorted(base.glob("r*"))
        for run_dir in run_dirs:
            config = json.loads((run_dir / "config.json").read_text())
            responses = {r["id"]: r for r in _jsonl(run_dir / "responses.jsonl")}
            for j in _jsonl(run_dir / judgments_file):
                if j["id"] not in items.index:
                    continue
                rec = responses.get(j["id"], {})
                turns = [rec[t] for t in ("turn1", "turn2") if t in rec]
                tool_calls = sum(1 for t in turns for s in t["trace"] if s["type"] in ("search", "read"))
                cost = sum(_system_cost(t["usage"], config["model"]) for t in turns)
                jin, jout = PRICES.get(j.get("judge_model", "claude-opus-5"), PRICES["claude-opus-5"])
                judge_cost = (j.get("judge_input_tokens", 0) * jin + j.get("judge_output_tokens", 0) * jout) / 1e6
                rows.append({
                    "model": MODEL_LABELS.get(mdir, mdir), "repeat": run_dir.name if run_dir != base else "r1", "id": j["id"],
                    **items.loc[j["id"]].to_dict(),
                    "verdict": j["verdict"], "score": SCORE[j["verdict"]], "passed": j["verdict"] == "pass",
                    "behavior": j["behavior"], "failure_mode": j["primary_failure_mode"],
                    "failure_group": FAILURE_MODES[j["primary_failure_mode"]][0],
                    "secondary": j.get("secondary_failure_modes", []), "grounding": j["grounding"],
                    "citation_relevance": j.get("citation_relevance"),
                    "followup_verdict": j["followup_verdict"], "gold_disagreement": j["gold_disagreement"],
                    "rationale": j["rationale"], "tool_calls": tool_calls, "cost": cost, "judge_cost": judge_cost,
                    "wall_s": rec.get("wall_s"), "asked": bool(turns and turns[0].get("clarification")),
                })
    return pd.DataFrame(rows)


def pass_rates(df: pd.DataFrame, by=("model", "repeat", "eval")) -> pd.DataFrame:
    return df.groupby(list(by))["passed"].mean().rename("pass_rate").reset_index()


def pair_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, repeat, eval): trap pass, control pass, pair accuracy."""
    paired = df[df["pair_id"].notna()]
    out = []
    for (model, rep, ev), g in paired.groupby(["model", "repeat", "eval"]):
        wide = g.pivot_table(index="pair_id", columns="role", values="passed", aggfunc="first")
        out.append({"model": model, "repeat": rep, "eval": ev,
                    "trap_pass": g.loc[g.role == "trap", "passed"].mean(),
                    "control_pass": g.loc[g.role == "control", "passed"].mean(),
                    "pair_accuracy": (wide.get("trap", False) & wide.get("control", False)).mean()})
    return pd.DataFrame(out)


def summarize_repeats(rates: pd.DataFrame, value="pass_rate", by=("model", "eval")) -> pd.DataFrame:
    """Mean, SD and range across repeats."""
    g = rates.groupby(list(by))[value]
    return pd.DataFrame({"mean": g.mean(), "sd": g.std(ddof=1), "min": g.min(), "max": g.max(), "runs": g.count()}).reset_index()


def item_stability(df: pd.DataFrame) -> pd.DataFrame:
    """Per (model, item): how many repeats passed, and whether the outcome was unanimous."""
    g = df.groupby(["model", "eval", "id"])["passed"]
    out = pd.DataFrame({"passes": g.sum(), "runs": g.count()}).reset_index()
    out["pass_frac"] = out["passes"] / out["runs"]
    out["status"] = np.select([out.passes == out.runs, out.passes == 0], ["always pass", "always fail"], "flaky")
    return out


def binomial_sd(p: float, n: int) -> float:
    """SD of a pass rate over n independent items at true rate p (what one run's sampling noise would be
    if items were redrawn); compare with the observed run-to-run SD on a fixed item set."""
    return float(np.sqrt(p * (1 - p) / n)) if n else float("nan")


def paired_bootstrap(df: pd.DataFrame, a: str, b: str, eval_name=None, n_boot: int = 5000, seed: int = 0):
    """Difference in mean per-item pass fraction (a - b), resampling items; returns (diff, lo, hi)."""
    d = df if eval_name is None else df[df["eval"] == eval_name]
    per_item = d.groupby(["id", "model"])["passed"].mean().unstack("model").dropna()
    diffs = (per_item[a] - per_item[b]).to_numpy()
    rng = np.random.default_rng(seed)
    boots = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)
    return float(diffs.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def style_axes(ax, title=None, xlabel=None, ylabel=None):
    """Recessive grid and axes; text in ink tokens, never series colors."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    if xlabel is not None:
        ax.set_xlabel(xlabel, color=TEXT_2, fontsize=9)
    if ylabel is not None:
        ax.set_ylabel(ylabel, color=TEXT_2, fontsize=9)
