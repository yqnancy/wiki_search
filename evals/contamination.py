"""Check eval items for overlap with the system's prompts and its development set.

Three signals, per eval item (question + gold answer):
    ngram       longest word sequence shared with a source (>= NGRAM_FLAG words is flagged)
    entity      a named entity from the question that also appears in a source. It is serious
                only when the entity IS one of the prompts' quoted examples (e.g. "Georgia"):
                the prompt then effectively shows the model how to handle that exact case
    fact        the item shares an entity AND an answer token (year, number, or name not
                already in the question) with a dev-set expected answer, i.e. the same fact
                was likely used while tuning the prompts

Severity: high = prompt example entity, dev-set fact, or >= 8 shared words;
medium = >= 5 shared words (usually a generic question stem); low = incidental entity.

Sources: SYSTEM_PROMPT and DISAMBIGUATION_PROMPT (wiki_search/prompts.py), the tool and
decision-tool descriptions (wiki_search/agent.py), and evals/eval_set.json (the original
18-question set the prompts were developed against).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from wiki_search.agent import DECISION_TOOL, TOOLS
from wiki_search.prompts import DISAMBIGUATION_PROMPT, SYSTEM_PROMPT

from .dataset import load_items

NGRAM_FLAG = 5
_GENERIC = {
    # Capitalized words that are formatting/instructions, not subject-matter entities.
    "answer", "answers", "assumptions", "sources", "reasoning", "verdict", "plan", "read", "select", "search",
    "check", "stop", "rules", "use", "when", "for", "never", "if", "the", "a", "an", "it", "this", "that",
    "wikipedia", "english wikipedia", "markdown", "url", "urls", "claude", "decide", "prefer", "record",
    "example", "also", "each", "don't", "do", "how", "what", "who", "which", "where", "why", "is", "was",
    "reply", "section", "lead", "table", "results", "disambiguation", "disambiguation page", "supported",
    "contradicted", "partially supported", "not addressed by wikipedia", "should", "yes", "no", "user",
    "in", "on", "of", "and", "or", "by", "to", "at", "as", "with", "from", "into", "claim", "should not ask",
    "should ask for clarification", "search operators", "wikipedia's", "you", "your", "they", "their",
    "cite", "mention", "follow", "results come", "searches", "skip", "hop", "hops", "optional",
}

_ENTITY_RE = re.compile(r"\b([A-Z][\w'\-]*(?:\s+(?:of|the|de|van|von)?\s*[A-Z][\w'\-]*)*)")
_TOKEN_RE = re.compile(r"\b(\d{3,4}|[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*)\b")


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def entities(text: str) -> set[str]:
    out = set()
    for m in _ENTITY_RE.finditer(text):
        e = m.group(1).strip()
        if e.lower() not in _GENERIC and len(e) > 2:
            out.add(e)
    return out


def longest_shared_ngram(a: str, b: str) -> tuple[int, str]:
    wa, wb = _words(a), _words(b)
    best, best_seq = 0, ""
    index: dict[str, list[int]] = {}
    for j, w in enumerate(wb):
        index.setdefault(w, []).append(j)
    for i, w in enumerate(wa):
        for j in index.get(w, []):
            k = 0
            while i + k < len(wa) and j + k < len(wb) and wa[i + k] == wb[j + k]:
                k += 1
            if k > best:
                best, best_seq = k, " ".join(wa[i:i + k])
    return best, best_seq


def load_sources() -> dict[str, str]:
    tools = "\n".join(t["description"] + json.dumps(t["input_schema"]) for t in TOOLS + [DECISION_TOOL])
    dev = json.loads((Path(__file__).parent / "eval_set.json").read_text())
    sources = {"system_prompt": SYSTEM_PROMPT, "disambiguation_prompt": DISAMBIGUATION_PROMPT, "tool_descriptions": tools}
    for d in dev:
        # Every answer-bearing field, so a leaked fact is caught whichever field holds it.
        answers = [d.get(k) for k in ("expected_answer", "acceptable_answers", "also_accept", "evidence")]
        flat = [a if isinstance(a, str) else json.dumps(a, ensure_ascii=False) for a in answers if a]
        sources[f"dev:{d['id']}"] = "\n".join([d["question"], *flat])
    return sources


def prompt_examples() -> set[str]:
    """Whole quoted strings in the prompts, e.g. "Georgia", "Great Fire of London"."""
    text = SYSTEM_PROMPT + DISAMBIGUATION_PROMPT
    return {q.strip() for q in re.findall(r'"([^"]{3,80})"', text)}


def check(items: list[dict] | None = None) -> list[dict]:
    """One row per (item, source) with any signal, most serious first."""
    items = items or load_items()
    sources = load_sources()
    source_entities = {name: entities(text) for name, text in sources.items()}
    examples = prompt_examples()
    rows = []
    for item in items:
        text = f"{item['question']}\n{item.get('gold_answer', '')}"
        q_ents = entities(item["question"])
        gold_tokens = set(_TOKEN_RE.findall(item.get("gold_answer", "")))
        for name, src in sources.items():
            n, seq = longest_shared_ngram(text, src)
            shared = sorted(q_ents & source_entities[name])
            fact = []
            if name.startswith("dev:") and shared:
                fact = sorted(t for t in gold_tokens
                              if t in src and t not in shared and t.lower() not in item["question"].lower())
            if n >= NGRAM_FLAG or shared or fact:
                kind = "prompt" if not name.startswith("dev:") else "dev_set"
                example_hit = kind == "prompt" and any(e in examples for e in shared)
                severity = ("high" if example_hit or fact or n >= 8
                            else "medium" if n >= NGRAM_FLAG else "low")
                rows.append({"id": item["id"], "eval": item["eval"], "source": name, "kind": kind,
                             "severity": severity, "shared_entities": shared, "shared_answer_tokens": fact,
                             "longest_ngram": n, "ngram": seq if n >= NGRAM_FLAG else "",
                             "question": item["question"]})
    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(rows, key=lambda r: (order[r["severity"]], r["id"]))


if __name__ == "__main__":
    for r in check():
        print(f"{r['severity']:6} {r['id']:9} {r['source']:22} ents={r['shared_entities']} "
              f"facts={r['shared_answer_tokens']} ngram={r['longest_ngram']} {r['ngram']!r}")
