"""The research-brief eval: run research requests through WikiQA and grade the briefs.

This eval covers the research-starting-point use case only. Fact verification has its own,
separate eval (evals/judge.py, evals/run.py and the sets in evals/data/); don't merge them.

    .venv/bin/python evals/run_eval.py                             # run all items, write grading packets
    .venv/bin/python evals/run_eval.py --ids research-02
    .venv/bin/python evals/run_eval.py --regrade evals/runs/<run>  # fresh packets for saved answers
    .venv/bin/python evals/run_eval.py --collect evals/runs/<run>  # score the verdicts, write summary

Grading is split in two. The code graders (brief_format, freshness) need nothing extra. Judged graders
(citations, completeness, relevance, style) are written as Markdown packets to
<run>/grading/<item id>/<grader>.md. By default Claude grades them in the Claude Code session,
writing <grader>.json next to each packet; with --judge api, the Anthropic API fills them instead.
--collect then validates every verdict, computes the scores and writes results.jsonl and summary.md.

A run lives in evals/runs/<timestamp>_<model>/: answers.jsonl (agent outputs, usage, cost),
traces/<id>.json, grading/, and after collecting, results.jsonl and summary.md.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from evals.graders import (  # noqa: E402
    CODE_GRADERS, GRADER_NAMES, JUDGED_GRADERS, RESEARCH_GRADERS, JudgeRequest, validate,
)
from wiki_search import wikipedia  # noqa: E402
from wiki_search.agent import DEFAULT_MODEL, Answer  # noqa: E402

SET_PATH = ROOT / "evals" / "research_set.json"
# Older runs (the first pilot) mixed in fact-check items; look their ids up here when collecting.
LEGACY_SET_PATH = ROOT / "evals" / "eval_set.json"
RUNS_DIR = ROOT / "evals" / "runs"
BATCH = {"session": 30, "api": 10}  # claims per citation packet

# $ per million tokens (input, output). Cache writes bill at 1.25x input, cache reads at 0.1x.
PRICES = {
    "claude-sonnet-4-6": (3.0, 15.0), "claude-sonnet-5": (2.0, 10.0), "claude-haiku-4-5": (1.0, 5.0),
    "claude-opus-5": (5.0, 25.0), "claude-opus-5-5": (4.0, 20.0), "claude-opus-4-8": (5.0, 25.0),
}


def cost(model: str, usage: dict) -> float:
    inp, out = PRICES.get(model, (0.0, 0.0))
    return round((usage.get("input_tokens", 0) * inp
                  + usage.get("cache_creation_input_tokens", 0) * inp * 1.25
                  + usage.get("cache_read_input_tokens", 0) * inp * 0.1
                  + usage.get("output_tokens", 0) * out) / 1e6, 4)


def row_from_answer(item: dict, a: Answer) -> dict:
    return {
        "id": item["id"], "category": item["category"], "question": item["question"],
        "model": a.model, "mode": a.mode, "stop_reason": a.stop_reason,
        "needs_clarification": a.needs_clarification,
        "clarification": vars(a.clarification) if a.clarification else None,
        "text": a.text, "structured": a.structured, "warnings": a.warnings,
        "reads": [{"title": t["title"], "section": t["section"]} for t in a.trace if t["type"] == "read"],
        "searches": [t["query"] for t in a.trace if t["type"] == "search"],
        "elapsed_s": a.elapsed_s, "usage": a.usage, "cost_usd": cost(a.model, a.usage),
        "run_date": datetime.date.today().isoformat(),
        # The revision of each cited page the agent read, for the freshness grader.
        "source_revisions": {s.title: getattr(wikipedia.get_article(s.title), "revid", None) for s in a.sources},
    }


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


# ---------------------------------------------------------------------------- grading packets

def build_requests(item: dict, row: dict, name: str, batch: int):
    """A grader's JudgeRequests for one row, or a dict grade when it doesn't apply."""
    request_fn = JUDGED_GRADERS[name][0]
    return request_fn(item, row, batch) if name == "citations" else request_fn(item, row)


def packet_markdown(item_id: str, req: JudgeRequest) -> str:
    return (f"# Grading packet: {item_id} · {req.key}\n\n"
            f"Write the verdict to `{req.key}.json` in this folder: JSON only, matching the schema at the end.\n\n"
            f"## Instructions\n\n{req.system}\n\n## Input\n\n{req.user}\n\n"
            f"## Output schema\n\n```json\n{json.dumps(req.schema, indent=2)}\n```\n")


def all_requests(items: dict, rows: list[dict], names: list[str], batch: int):
    for row in rows:
        for name in names:
            if name in JUDGED_GRADERS:
                reqs = build_requests(items[row["id"]], row, name, batch)
                if not isinstance(reqs, dict):
                    for req in reqs:
                        yield row, req


def write_packets(items: dict, rows: list[dict], names: list[str], run_dir: Path, batch: int) -> list[Path]:
    pending = []
    for row, req in all_requests(items, rows, names, batch):
        folder = run_dir / "grading" / row["id"]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{req.key}.md").write_text(packet_markdown(row["id"], req))
        if not (folder / f"{req.key}.json").exists():
            pending.append(folder / f"{req.key}.md")
    return pending


def judge_with_api(items: dict, rows: list[dict], names: list[str], run_dir: Path, model: str) -> None:
    from evals.api_judge import Judge, JudgeError

    judge = Judge(model=model)

    def work(job):
        row, req = job
        path = run_dir / "grading" / row["id"] / f"{req.key}.json"
        try:
            verdict, usage = judge(req)
            path.write_text(json.dumps(verdict, indent=2, ensure_ascii=False))
            return cost(usage["model"], usage)
        except JudgeError as e:
            path.with_suffix(".error").write_text(str(e))
            return 0.0

    with ThreadPoolExecutor(max_workers=4) as pool:
        total = sum(pool.map(work, list(all_requests(items, rows, names, BATCH["api"]))))
    (run_dir / "grading" / "JUDGE").write_text(f"api: {model} (${total:.2f})\n")


# ---------------------------------------------------------------------------- collecting

def collect(items: dict, rows: list[dict], names: list[str], run_dir: Path, batch: int) -> None:
    for row in rows:
        item, folder = items[row["id"]], run_dir / "grading" / row["id"]
        grades = {}
        for name in names:
            if name in CODE_GRADERS:
                grades[name] = CODE_GRADERS[name](item, row)
                continue
            reqs = build_requests(item, row, name, batch)
            if isinstance(reqs, dict):
                grades[name] = reqs
                continue
            verdicts, problems = [], []
            for req in reqs:
                path = folder / f"{req.key}.json"
                if path.with_suffix(".error").exists():
                    problems.append(f"{req.key}: {path.with_suffix('.error').read_text().strip()}")
                elif not path.exists():
                    problems.append(f"{req.key}: pending")
                else:
                    verdict = json.loads(path.read_text())
                    problems += [f"{req.key}: {e}" for e in validate(verdict, req.schema)]
                    verdicts.append(verdict)
            grades[name] = ({"score": None, "error": "; ".join(problems)} if problems
                            else JUDGED_GRADERS[name][1](item, row, verdicts))
        row["grades"] = grades


def summarize(rows: list[dict], names: list[str]) -> str:
    def mean(vals):
        vals = [v for v in vals if v is not None]
        return f"{sum(vals) / len(vals):.2f} (n={len(vals)})" if vals else "–"

    cats = list(dict.fromkeys(r["category"] for r in rows))
    lines = ["| category | " + " | ".join(names) + " | agent $ | avg s |", "|---|" + "---|" * (len(names) + 2)]
    for cat in cats + ["all"]:
        rs = [r for r in rows if cat in ("all", r["category"])]
        cells = [mean(r["grades"].get(n, {}).get("score") for r in rs) for n in names]
        lines.append(f"| {cat} | " + " | ".join(cells)
                     + f" | {sum(r['cost_usd'] for r in rs):.2f} | {sum(r['elapsed_s'] for r in rs) / len(rs):.0f} |")
    errors = [f"- {r['id']} {n}: {g['error']}" for r in rows for n, g in r["grades"].items() if g.get("error")]
    return "\n".join(lines) + ("\n\nNot scored:\n" + "\n".join(errors) if errors else "")


# ---------------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--category")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Agent backbone model.")
    parser.add_argument("--set", type=Path, default=SET_PATH, help="Eval set to run (default: research_set.json).")
    parser.add_argument("--graders", nargs="*", default=RESEARCH_GRADERS, choices=GRADER_NAMES)
    parser.add_argument("--judge", choices=["session", "api"], default="session",
                        help="Who fills judged graders: Claude in the Claude Code session (default) or the API.")
    parser.add_argument("--judge-model", default="claude-opus-5", help="Model for --judge api.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--regrade", type=Path, help="Write fresh grading packets for a saved run.")
    group.add_argument("--collect", type=Path, help="Score the verdicts in a run and write the summary.")
    args = parser.parse_args()

    run_items = json.loads(args.set.read_text())
    items = {i["id"]: i for i in json.loads(LEGACY_SET_PATH.read_text()) + run_items}

    def selected(rows):
        return [r for r in rows if (not args.ids or r["id"] in args.ids)
                and (not args.category or r["category"] == args.category)]

    if args.collect:
        run_dir = args.collect.resolve()
        judge_file = run_dir / "grading" / "JUDGE"
        judge = judge_file.read_text().strip() if judge_file.exists() else "in-session Claude"
        rows = selected(load_jsonl(run_dir / "answers.jsonl"))
        collect(items, rows, args.graders, run_dir, BATCH["api" if judge.startswith("api") else "session"])
        write_jsonl(run_dir / "results.jsonl", rows)
        summary = summarize(rows, args.graders)
        (run_dir / "summary.md").write_text(f"# Eval run {run_dir.name}\n\nJudge: {judge}\n\n{summary}\n")
        print(f"{summary}\n\nResults: {run_dir.relative_to(ROOT)}/results.jsonl")
        return

    if args.regrade:
        run_dir = args.regrade.resolve()
        rows = selected(load_jsonl(run_dir / "answers.jsonl"))
    else:
        from wiki_search import WikiQA

        run_dir = RUNS_DIR / f"{datetime.datetime.now():%Y%m%d-%H%M%S}_{args.model}"
        (run_dir / "traces").mkdir(parents=True)
        qa = WikiQA(model=args.model)
        rows = []
        for item in selected(run_items):
            start = time.time()
            a = qa.ask(item["question"])
            print(f"  ran {item['id']} in {time.time() - start:.0f}s")
            rows.append(row_from_answer(item, a))
            (run_dir / "traces" / f"{item['id']}.json").write_text(json.dumps(a.trace, indent=2, ensure_ascii=False))
        write_jsonl(run_dir / "answers.jsonl", rows)

    rel = run_dir.relative_to(ROOT)
    pending = write_packets(items, rows, args.graders, run_dir, BATCH[args.judge])
    if args.judge == "api":
        judge_with_api(items, rows, args.graders, run_dir, args.judge_model)
        print(f"API judging done. Score with: .venv/bin/python evals/run_eval.py --collect {rel}")
    else:
        print(f"{len(pending)} grading packets pending in {rel}/grading/. Ask Claude to grade them "
              f"(each <grader>.md -> <grader>.json), then run: .venv/bin/python evals/run_eval.py --collect {rel}")


if __name__ == "__main__":
    main()
