"""Grade saved responses with an LLM judge that labels behavior, verdict, and failure mode.

The judge sees the eval item (question, expected behavior, gold answer, rubric), the
system's answer(s), a compact tool trace, and the text of every Wikipedia section the
answer cites. It returns labels drawn only from evals/taxonomy.py:

    behavior                 what the system did (asked, answered, corrected premise, ...)
    verdict                  pass | partial | fail against the item's expected behavior
    primary_failure_mode     one taxonomy code ("none" on a pass)
    secondary_failure_modes  any further codes, including grounding problems
    followup_verdict         for clarification turns: was the post-clarification answer right?
    grounding                whether cited sections support the answer's claims
    citation_relevance       whether cited sources bear on the question and their statements
    gold_disagreement        the answer contradicts the gold but Wikipedia supports the answer
                             (flags the item for review rather than blaming the system)

    .venv/bin/python -m evals.judge --name baseline
"""

from __future__ import annotations

import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import anthropic
from dotenv import load_dotenv

from wiki_search import wikipedia

from .dataset import load_items
from .run import RUNS_DIR
from .taxonomy import BEHAVIORS, FAILURE_MODES, taxonomy_text

JUDGE_MODEL = "claude-opus-5"
MAX_SECTION_CHARS = 5000
MAX_SECTIONS = 8

VERDICTS = ["pass", "partial", "fail"]
JUDGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "behavior": {"type": "string", "enum": list(BEHAVIORS)},
        "verdict": {"type": "string", "enum": VERDICTS},
        "primary_failure_mode": {"type": "string", "enum": list(FAILURE_MODES)},
        "secondary_failure_modes": {"type": "array", "items": {"type": "string", "enum": list(FAILURE_MODES)}},
        "followup_verdict": {"type": "string", "enum": VERDICTS + ["not_applicable"]},
        "grounding": {"type": "string", "enum": [
            "all_supported", "some_unsupported", "mostly_unsupported", "no_citations", "not_applicable"]},
        "citation_relevance": {"type": "string", "enum": [
            "all_relevant", "some_irrelevant", "mostly_irrelevant", "no_citations", "not_applicable"]},
        "gold_disagreement": {"type": "boolean"},
        "rationale": {"type": "string"},
    },
    "required": ["behavior", "verdict", "primary_failure_mode", "secondary_failure_modes",
                 "followup_verdict", "grounding", "citation_relevance", "gold_disagreement", "rationale"],
    "additionalProperties": False,
}

JUDGE_SYSTEM = f"""\
You grade a question-answering system that answers only from English Wikipedia using search and article-reading tools. Its answers are rendered from structured fields (answer, verdict for claim checks, reasoning with citations, assumptions, sources); `question_type="not-covered"` means it concluded Wikipedia can't answer. A citation warning means it cited an article it never retrieved. For each eval item you receive the question, the behavior the system was expected to show, a gold answer, sometimes a rubric, the system's response, a summary of its tool calls, and the text of the Wikipedia sections it cited.

Label the response using only the taxonomy below.

{taxonomy_text()}

How to grade:
- `behavior`: what the system actually did in its first turn. If the first turn is a clarification request, it is `asked_clarification` (this is given to you).
- `verdict`: judge against the expected behavior, the gold answer and the rubric. The rubric, when present, overrides general rules.
  - `pass`: the expected behavior, with the key facts right. Extra correct detail is fine. Numbers within the rubric's tolerance (or reasonable rounding) are fine.
  - `partial`: the right behavior with a real but limited flaw (e.g. a correct main answer with one wrong secondary detail, or a correct figure without the scope note the rubric asks for), or a case the expected-behavior definition calls partial.
  - `fail`: wrong behavior, or the main answer is wrong or missing.
- For expected behavior `clarify`: asking passes; answering without asking passes only if it explicitly covers each plausible reading (2-3 readings, each short); silently picking one reading fails with `missed_ambiguity`.
- For expected behavior `answer`: asking for clarification fails with `unnecessary_clarification`; correcting a true premise fails with `spurious_premise_correction`; saying Wikipedia doesn't cover it fails with `unwarranted_refusal`.
- For expected behavior `answer_with_perspectives`: the question is genuinely contested; pass requires the main positions and their standing. Stating one side as settled fact is `one_sided_contested`. Conversely, when the item is a settled consensus (category `consensus`), presenting it as an open debate is `false_balance`.
- Items with an `as_of` date are time-sensitive: treat `as_of` as today. An answer that was true earlier but is outdated per the gold/Wikipedia is `stale_answer`; an age or elapsed-time answer computed for the wrong year is `date_unaware`.
- If the question is not in English (`language`), the fact must still be right; answering in a different language adds `language_mismatch` as a secondary mode but does not by itself change the verdict.
- Footnote rule: Critical information must not live only in the `assumptions` footnote: a premise correction, a caveat the rubric requires, or the scope/metric an answer depends on must appear in the headline answer or the reasoning. If it appears only under 'Assumptions & gaps', the item is at most partial (the headline went along with the premise / omitted the caveat).
- `primary_failure_mode`: exactly `none` when the verdict is `pass`; otherwise the single most specific code that explains the failure. Prefer the eval-specific codes (ambiguity, scope, naming, premise) over generic content codes when they apply. Use the trace to tell `retrieval_miss` (never found the right article/section) from `reasoning_error` (had the facts, combined them wrongly).
- `secondary_failure_modes`: other codes that also apply, including grounding problems. Leave empty if none. Do not repeat the primary.
- `followup_verdict`: if there is a second turn (after the user's clarification reply), grade that answer against `gold_answer_after_clarification` (or the gold answer); otherwise `not_applicable`.
- `grounding`: check the load-bearing claims of the final answer against the cited section texts provided. `all_supported` if each is supported; `some_unsupported` or `mostly_unsupported` otherwise (also add `unsupported_citation` or `unsourced_claim` as a secondary mode); `no_citations` if the answer cites nothing; `not_applicable` for a pure clarification request with no answer. A search-result intro counts as the article's lead. Grounding does not change the verdict unless the answer's key fact is unsupported and wrong.
- `citation_relevance`: check whether each cited source bears on the question and on the statement it is attached to. A citation is relevant when the cited article/section is about the entity or topic its statement concerns and helps answer the question. It is irrelevant when it is padding or tangential: a different entity with a similar name, a section unrelated to the claim, a generic article cited for a specific fact, or a source listed but used by no statement. `all_relevant` if every citation is relevant; `some_irrelevant` or `mostly_irrelevant` otherwise (also add `irrelevant_citation` as a secondary mode); `no_citations` if the answer cites nothing; `not_applicable` for a pure clarification request with no answer. This is separate from `grounding`: a citation can be relevant yet not support the exact claim (a grounding problem), or support a trivially true side remark while being irrelevant to the question (a relevance problem). Citation relevance does not change the verdict.
- `gold_disagreement`: true only if the system's answer contradicts the gold answer AND the cited Wikipedia text clearly supports the system (the gold may be outdated or wrong). In that case, grade the verdict by Wikipedia's text, not the gold.
- `rationale`: two to four sentences explaining the verdict and the failure mode.
"""


# --------------------------------------------------------------------- evidence

_CITE_RE = re.compile(r"^\s*\[(\d+)\]\s*(.+?)\s*(?:§\s*(.+?))?\s*[—–-]+\s*https?://", re.MULTILINE)
_lock = threading.Lock()


def cited_sections(answer_text: str, sources: list[dict]) -> list[tuple[str, str]]:
    """(title, section) pairs cited in the answer's Sources list, plus sections it read."""
    pairs = []
    for m in _CITE_RE.finditer(answer_text or ""):
        title, section = m.group(2).strip(), (m.group(3) or "(lead)").strip()
        if title.endswith("(lead)"):
            title, section = title[: -len("(lead)")].strip(), "(lead)"
        pairs.append((title, section))
    for s in sources or []:
        pairs.append((s["title"], s["section"]))
    seen, out = set(), []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out[:MAX_SECTIONS]


def section_text(title: str, section: str) -> str:
    if section.lower() == "infobox":  # pseudo-section the agent can read and cite
        rows = wikipedia.get_infobox(title)
        return ("\n".join(f"{k}: {v}" for k, v in rows) or "(no infobox)")[:MAX_SECTION_CHARS]
    with _lock:  # get_article is lru_cached; serialize to avoid duplicate fetches
        article = wikipedia.get_article(title)
    if article is None:
        return "(article not found)"
    if section.lower() in ("(lead)", "lead"):
        text = article.lead
    else:
        found = article.find_section(section)
        text = found[1] if found else f"(section not found; lead follows)\n{article.lead}"
    return text[:MAX_SECTION_CHARS]


def trace_summary(trace: list[dict]) -> str:
    lines = []
    for t in trace:
        if t["type"] == "search":
            lines.append(f"search {t['query']!r} -> {', '.join(t['results'][:6])}")
        elif t["type"] == "read":
            lines.append(f"read {t['title']} § {t['section']}")
        elif t["type"] == "disambiguation":
            d = "ambiguous" if t.get("ambiguous") else f"resolved to {t.get('most_likely_title')}"
            lines.append(f"disambiguation check on {t['page']}: {d}")
        elif t["type"] == "error":
            lines.append(f"tool error in {t['tool']}: {t['error'][:200]}")
    return "\n".join(lines) or "(no tool calls)"


# --------------------------------------------------------------------- judging

ITEM_FIELDS = ["id", "eval", "category", "language", "as_of", "role", "subtype", "question", "expected_behavior", "gold_answer",
               "acceptable_variants", "rubric", "clarification_reply", "gold_answer_after_clarification",
               "alias", "canonical_title"]


def _answer_block(tag: str, turn: dict) -> str:
    if turn.get("clarification"):
        c = turn["clarification"]
        return f"<{tag} kind=\"clarification_request\">\n{c['question']}\nOptions: {c['options']}\n</{tag}>"
    meta = f'stop_reason="{turn["stop_reason"]}"'
    if turn.get("structured"):
        meta += f' question_type="{turn["structured"].get("question_type", "")}"'
    warnings = "".join(f"\n[citation warning] {w}" for w in turn.get("warnings") or [])
    return f"<{tag} kind=\"answer\" {meta}>\n{turn['text']}{warnings}\n</{tag}>"


def build_prompt(item: dict, rec: dict) -> str:
    parts = ["<eval_item>", json.dumps({k: item[k] for k in ITEM_FIELDS if k in item}, ensure_ascii=False, indent=1),
             "</eval_item>"]
    if rec.get("error"):
        parts.append(f"<system_error>{rec['error']}</system_error>")
        return "\n".join(parts)

    t1 = rec["turn1"]
    parts.append(_answer_block("turn1", t1))
    parts.append(f"<turn1_tool_trace>\n{trace_summary(t1['trace'])}\n</turn1_tool_trace>")

    final = t1
    if rec.get("turn2"):
        t2 = rec["turn2"]
        parts.append(f"<user_clarification_reply>{rec['clarification_reply']}</user_clarification_reply>")
        parts.append(_answer_block("turn2", t2))
        parts.append(f"<turn2_tool_trace>\n{trace_summary(t2['trace'])}\n</turn2_tool_trace>")
        final = t2

    sections = cited_sections(final.get("text", ""), final.get("sources", []))
    if sections:
        parts.append("<cited_wikipedia_sections>")
        for title, section in sections:
            parts.append(f"<section title=\"{title}\" heading=\"{section}\">\n{section_text(title, section)}\n</section>")
        parts.append("</cited_wikipedia_sections>")
    return "\n".join(parts)


def judge_one(client: anthropic.Anthropic, item: dict, rec: dict, model: str) -> dict:
    prompt = build_prompt(item, rec)
    response = client.messages.create(
        model=model,
        max_tokens=16000,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
        thinking={"type": "adaptive"},
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": JUDGMENT_SCHEMA}},
        # Re-run on Anthropic's recommended fallback model if a safety classifier declines.
        extra_headers={"anthropic-beta": "server-side-fallback-2026-07-01"},
        extra_body={"fallbacks": "default"},
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("judge refused")
    judgment = json.loads(next(b.text for b in response.content if b.type == "text"))

    # Deterministic facts override the judge where the run record is unambiguous.
    if rec.get("error"):
        judgment.update(behavior="declined", verdict="fail", primary_failure_mode="tool_error")
    elif rec["turn1"].get("clarification"):
        judgment["behavior"] = "asked_clarification"
    elif rec["turn1"]["stop_reason"] == "turn_limit" and judgment["verdict"] != "pass":
        judgment["primary_failure_mode"] = "budget_exhausted"
    if judgment["verdict"] == "pass":
        judgment["primary_failure_mode"] = "none"
    elif judgment["primary_failure_mode"] == "none":
        judgment["primary_failure_mode"] = "other"

    u = response.usage
    usage = {"judge_input_tokens": u.input_tokens + (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0),
             "judge_output_tokens": u.output_tokens}
    return {"id": item["id"], "eval": item["eval"], "judge_model": response.model, **judgment, **usage}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="Run name under evals/runs/.")
    parser.add_argument("--model", default=JUDGE_MODEL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rejudge", action="store_true", help="Re-grade items already judged.")
    parser.add_argument("--tag", help="Write judgments_<tag>.jsonl instead (e.g. a second grading pass).")
    args = parser.parse_args()
    load_dotenv()

    run_dir = RUNS_DIR / args.name
    records = [json.loads(l) for l in (run_dir / "responses.jsonl").read_text().splitlines() if l.strip()]
    items = {i["id"]: i for i in load_items()}
    out_path = run_dir / (f"judgments_{args.tag}.jsonl" if args.tag else "judgments.jsonl")

    done: dict[str, dict] = {}
    if out_path.exists() and not args.rejudge:
        done = {j["id"]: j for j in (json.loads(l) for l in out_path.read_text().splitlines() if l.strip())}
    todo = [r for r in records if r["id"] not in done and r["id"] in items]

    client = anthropic.Anthropic()
    results = dict(done)
    print(f"Judging {len(todo)} responses ({len(done)} already judged) with {args.model}")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(judge_one, client, items[r["id"]], r, args.model): r["id"] for r in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            rid = futures[fut]
            try:
                j = fut.result()
            except Exception as e:
                print(f"[{n}/{len(todo)}] {rid:9} JUDGE ERROR {type(e).__name__}: {e}")
                continue
            results[rid] = j
            print(f"[{n}/{len(todo)}] {rid:9} {j['verdict']:7} {j['primary_failure_mode']}", flush=True)

    order = [r["id"] for r in records if r["id"] in results]
    with open(out_path, "w") as f:
        for rid in order:
            f.write(json.dumps(results[rid], ensure_ascii=False) + "\n")
    print(f"Wrote {len(order)} judgments -> {out_path}")


if __name__ == "__main__":
    main()
