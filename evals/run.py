"""Run WikiQA over eval items and save raw responses (graded separately by evals.judge).

When the system asks for clarification, the item's scripted `clarification_reply` is sent
as the user's answer and the second turn is saved too, so both the decision to ask and
the final answer can be graded.

    .venv/bin/python -m evals.run --name baseline
    .venv/bin/python -m evals.run --name smoke --evals ambiguity --limit 6
    .venv/bin/python -m evals.run --name baseline --resume     # skip ids already saved
    .venv/bin/python -m evals.run --name sonnet --repeats 5    # writes sonnet/r1 ... sonnet/r5

Repeats run in one process and share cached Wikipedia lookups, so every repeat sees the same
search results and article text: run-to-run variance then reflects the model, not Wikipedia.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from wiki_search import WikiQA, wikipedia
from wiki_search.agent import DEFAULT_MODEL

from .dataset import EVAL_NAMES, load_items

RUNS_DIR = Path(__file__).parent / "runs"
DEFAULT_REPLY = "Please go with the most common meaning."


def _answer_dict(answer) -> dict:
    d = dataclasses.asdict(answer)
    # Thinking text is bulky and the judge doesn't read it; keep tool steps only.
    d["trace"] = [t for t in d["trace"] if t["type"] != "thinking"]
    return d


def run_item(qa: WikiQA, item: dict) -> dict:
    record = {"id": item["id"], "eval": item["eval"], "question": item["question"]}
    start = time.time()
    try:
        first = qa.ask(item["question"])
        record["turn1"] = _answer_dict(first)
        if first.needs_clarification:
            reply = item.get("clarification_reply") or DEFAULT_REPLY
            record["clarification_reply"] = reply
            record["turn2"] = _answer_dict(qa.ask(item["question"], clarification=reply))
    except Exception as e:  # recorded, then dropped and retried by --resume
        record["error"] = f"{type(e).__name__}: {e}"
    # The agent turns tool exceptions into tool errors the model sees; an Anthropic API failure
    # there (e.g. in the disambiguation sub-call) is infrastructure noise, not system behavior.
    for turn in ("turn1", "turn2"):
        for step in record.get(turn, {}).get("trace", []):
            if step["type"] == "error" and "Error code:" in step["error"]:
                record["error"] = f"API error inside tool call: {step['error'][:200]}"
    record["wall_s"] = round(time.time() - start, 1)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="Run name; output goes to evals/runs/<name>/.")
    parser.add_argument("--evals", nargs="*", choices=EVAL_NAMES, help="Subset of evals (default: all).")
    parser.add_argument("--ids", nargs="*", help="Only these item ids.")
    parser.add_argument("--limit", type=int, help="First N items per eval (smoke tests).")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1, help="Run everything N times into <name>/r1..rN.")
    parser.add_argument("--resume", action="store_true", help="Skip items already in responses.jsonl.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--today", help="Date the agent treats as today (YYYY-MM-DD); pin it to the items' as_of.")
    parser.add_argument("--max-articles", type=int, default=10)
    parser.add_argument("--max-tool-calls", type=int, default=20)
    args = parser.parse_args()
    load_dotenv()

    items = load_items(args.evals)
    if args.ids:
        items = [i for i in items if i["id"] in set(args.ids)]
    if args.limit:
        per_eval: dict[str, int] = {}
        kept = []
        for i in items:
            per_eval[i["eval"]] = per_eval.get(i["eval"], 0) + 1
            if per_eval[i["eval"]] <= args.limit:
                kept.append(i)
        items = kept

    # Same query -> same results across items and repeats (see module docstring).
    wikipedia.search = lru_cache(maxsize=None)(wikipedia.search)
    wikipedia.get_article = lru_cache(maxsize=None)(wikipedia.get_article.__wrapped__)

    run_dirs = [RUNS_DIR / args.name] if args.repeats == 1 else [
        RUNS_DIR / args.name / f"r{k}" for k in range(1, args.repeats + 1)]
    today = datetime.date.fromisoformat(args.today) if args.today else None
    config = {"model": args.model, "effort": args.effort, "today": args.today, "max_articles": args.max_articles,
              "max_tool_calls": args.max_tool_calls, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    jobs, files = [], {}
    for run_dir in run_dirs:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(config, indent=2))
        out_path = run_dir / "responses.jsonl"
        done = set()
        if args.resume and out_path.exists():
            # Keep finished records; drop API/network failures so they are retried.
            kept = [r for r in (json.loads(l) for l in out_path.read_text().splitlines() if l.strip())
                    if "error" not in r]
            out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
            done = {r["id"] for r in kept}
        files[run_dir] = open(out_path, "a")
        jobs += [(run_dir, item) for item in items if item["id"] not in done]

    # Generous SDK retries: concurrent-request 429s are expected when both models run at once.
    client = anthropic.Anthropic(max_retries=10)
    qa = WikiQA(model=args.model, disambiguation_model=args.model, effort=args.effort, today=today,
                max_articles=args.max_articles, max_tool_calls=args.max_tool_calls, client=client)
    lock = threading.Lock()
    print(f"Running {len(jobs)} item-runs across {len(run_dirs)} run(s) with {args.model}")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_item, qa, item): run_dir for run_dir, item in jobs}
        for n, fut in enumerate(as_completed(futures), 1):
            run_dir = futures[fut]
            rec = fut.result()
            with lock:
                files[run_dir].write(json.dumps(rec, ensure_ascii=False) + "\n")
                files[run_dir].flush()
            status = rec.get("error") or ("clarify" if "turn2" in rec else rec["turn1"]["stop_reason"])
            print(f"[{n}/{len(jobs)}] {run_dir.name:4} {rec['id']:9} {rec['wall_s']:6.1f}s  {status}", flush=True)
    for f in files.values():
        f.close()


if __name__ == "__main__":
    main()
