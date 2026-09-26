"""The submit_answer tool: the agent's final answer as structured fields, rendered to Markdown here.

Keeping the answer structured means formatting is enforced by code rather than prompt wording,
source URLs are built from titles rather than copied by the model, and evals can grade fields
(e.g. `answer`, `verdict`) without parsing Markdown.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .wikipedia import page_url

VERDICTS = ["Supported", "Contradicted", "Partially supported", "Disputed", "Not addressed by Wikipedia"]
LEAD = "(lead)"
INFOBOX = "Infobox"  # pseudo-section: the article's infobox, read via read_article(section="Infobox")

# submit_answer is strict (grammar-constrained), which requires additionalProperties: false
# on every object. Every property is also listed in `required`; optional fields take an
# empty list or null instead of being omitted. submit_report has the same shape but isn't
# strict: with both strict, the API rejects the request ("compiled grammar is too large"),
# even with $defs and without the nullable table. The renderers tolerate malformed fields.

SUBMIT_TOOL = {
    "name": "submit_answer",
    "strict": True,
    "description": (
        "Submit your final answer. Call this exactly once, when you are done researching; it ends "
        "the task. The answer is formatted for the reader from these fields, so put no Markdown "
        "headings or citation lists in them."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "question_type": {
                "type": "string",
                "enum": ["fact", "claim", "multi-hop", "not-covered"],
                "description": (
                    "not-covered: no factual answer Wikipedia can give: a forecast, an opinion or value "
                    "judgment, a personal decision, private info, or a fact Wikipedia doesn't record. Say so "
                    "in `answer` and put what Wikipedia can offer (typical climate, candidate definitions, "
                    "relevant facts) in `reasoning`, cited. Not for false premises: if the question assumes something untrue, "
                    "e.g. an event that never happened, use fact or claim and state the correction in `answer`."
                ),
            },
            "answer": {
                "type": "string",
                "description": (
                    "Only the bare answer, so a reader finds it at a glance: a name, number, date, place "
                    "or yes/no, at most about 10 words, e.g. 'Jean Sibelius'. For claim checks, a gist of "
                    "a few words to follow the verdict, e.g. 'it opened a decade later', without repeating "
                    "the verdict (no 'True'/'False'). If the question's premise is false, the correction "
                    "itself, e.g. 'No such treaty was signed'. If the answer depends on which reading or "
                    "definition is meant, give the answer for the one you used here, e.g. '2.1 million', and "
                    "state that reading and the main alternative (with its answer if it differs) in the "
                    "first reasoning item. Extra facts, context and explanation also go in `reasoning`. "
                    "No citations."
                ),
            },
            "verdict": {
                "anyOf": [{"type": "string", "enum": VERDICTS}, {"type": "null"}],
                "description": ("Claim checks only; null for other question types. 'Disputed': Wikipedia describes the "
                                "question as contested, with no consensus either way."),
            },
            "reasoning": {
                "type": "array",
                "description": (
                    "The evidence behind the answer, one fact per item, in order. For multi-hop "
                    "questions, one item per hop. Use as many items as the evidence needs and no "
                    "more: a simple fact usually needs one. Don't restate the answer."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "statement": {
                            "type": "string",
                            "description": "One sentence, about 25 words or fewer, with no citation markers.",
                        },
                        "citations": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "1-based indexes into `sources` that support this statement.",
                        },
                    },
                    "required": ["statement", "citations"],
                    "additionalProperties": False,
                },
            },
            "assumptions": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Method notes and limits of the sources only: what Wikipedia didn't cover, conflicting "
                    "sources, how a figure was computed, facts from outside Wikipedia (labelled as such). "
                    "Readers skim past this, so anything they need to interpret the answer (another reading "
                    "of the question, the definition or scope the answer depends on and its main "
                    "alternative, a premise correction) goes in the answer and reasoning, not here. No new "
                    "factual claims unless labelled as background knowledge. Empty list if none."
                ),
            },
            "sources": {
                "type": "array",
                "description": "Every article section cited in `reasoning`, in citation order.",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Exact article title as returned by the tools."},
                        "section": {
                            "type": "string",
                            "description": (
                                f'Section heading where the fact appears, "{LEAD}" for the article '
                                f'opening, or "{INFOBOX}" for its infobox.'
                            ),
                        },
                    },
                    "required": ["title", "section"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["question_type", "answer", "verdict", "reasoning", "assumptions", "sources"],
        "additionalProperties": False,
    },
}


_CITED_POINT = {
    "type": "object",
    "properties": {
        "statement": {
            "type": "string",
            "description": (
                "One sentence, about 30 words or fewer, with no citation markers. A synthesis that "
                "combines facts from several sources is fine if it cites all of them."
            ),
        },
        "citations": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "1-based indexes into `sources` that support this statement.",
        },
    },
    "required": ["statement", "citations"],
    "additionalProperties": False,
}
_SOURCES = SUBMIT_TOOL["input_schema"]["properties"]["sources"]
_ASSUMPTIONS = SUBMIT_TOOL["input_schema"]["properties"]["assumptions"]

REPORT_TOOL = {
    "name": "submit_report",
    "description": (
        "Submit a research brief: use instead of submit_answer when the user asked for an overview, "
        "history, explanation or comparison of a topic rather than a specific fact or claim. Call "
        "exactly once, when done researching; it ends the task. The brief is formatted for the "
        "reader from these fields, so put no Markdown headings or citation lists in them."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "A short title for the brief."},
            "overview": {
                "type": "array",
                "description": ("Two or three points giving the big picture for a reader who stops here: what the "
                                "topic is and the ideas that tie it together. Don't preview each section."),
                "items": _CITED_POINT,
            },
            "sections": {
                "type": "array",
                "description": (
                    "The body, organised the way that best answers the request: e.g. eras for a "
                    "history, one section per item or per dimension for a comparison. Overview plus "
                    "sections should total roughly 300-500 words. Some overlap with a table is fine, "
                    "but give sections what the table can't show. If the user stated a goal, the last "
                    "section is a short one for that goal (see the system prompt)."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "heading": {"type": "string"},
                        "points": {"type": "array", "items": _CITED_POINT},
                    },
                    "required": ["heading", "points"],
                    "additionalProperties": False,
                },
            },
            "table": {
                "description": (
                    "Usually null. A table only when several items (options, types, positions) are each "
                    "described on the same attributes and the reader would scan across rows to compare "
                    "them. Not for eras, causes, steps or traits that are already the sections. Keep "
                    "cells short."
                ),
                "anyOf": [{
                    "type": "object",
                    "properties": {
                        "purpose": {
                            "type": "string",
                            "description": ("One sentence: the comparison the reader makes by scanning across "
                                            "the rows, e.g. 'which option trades range for cost and safety'."),
                        },
                        "columns": {"type": "array", "items": {"type": "string"}},
                        "rows": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "cells": {"type": "array", "items": {"type": "string"}},
                                    "citations": {"type": "array", "items": {"type": "integer"}},
                                },
                                "required": ["cells", "citations"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["purpose", "columns", "rows"],
                    "additionalProperties": False,
                }, {"type": "null"}],
            },
            "further_reading": {
                "type": "array",
                "description": "Three to six Wikipedia articles to explore next, each with a short phrase on why it's worth reading.",
                "items": {
                    "type": "object",
                    "properties": {"title": {"type": "string"}, "why": {"type": "string"}},
                    "required": ["title", "why"],
                    "additionalProperties": False,
                },
            },
            "selection_basis": {
                "type": "object",
                "description": (
                    "How you chose which items, eras or subtopics to cover. If the request asks for a "
                    "selection ('the most common', 'the key milestones'), cite the sources that support "
                    "that choice; if none do, say it is your own judgment and leave citations empty. "
                    "Use 'Not applicable' when the request involves no selection."
                ),
                "properties": {
                    "explanation": {"type": "string"},
                    "citations": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["explanation", "citations"],
                "additionalProperties": False,
            },
            "assumptions": _ASSUMPTIONS,
            "sources": _SOURCES,
        },
        "required": ["title", "overview", "sections", "table", "further_reading", "selection_basis",
                     "assumptions", "sources"],
        "additionalProperties": False,
    },
}


@dataclass
class Source:
    title: str
    section: str
    url: str
    retrieved: bool  # False if the agent cites an article it never saw this run


def _items(value) -> list:
    """A list field as a list, even if the model sent a bare value (e.g. one string, which
    would otherwise render one bullet per character) or null."""
    if value is None or value == "":
        return []
    return value if isinstance(value, list) else [value]


def _text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def _build_sources(data: dict, retrieved_titles: set[str], warnings: list[str]) -> tuple[list[Source], dict[int, int]]:
    """Sources with duplicates merged, plus a map from the model's 1-based source
    numbers to the rendered ones."""
    seen = {t.lower() for t in retrieved_titles}
    sources: list[Source] = []
    remap: dict[int, int] = {}
    for i, s in enumerate(_items(data.get("sources")), 1):
        if not isinstance(s, dict):
            s = {"title": str(s)}
        title = _text(s.get("title"))
        section = _text(s.get("section")) or LEAD
        existing = next((j for j, src in enumerate(sources, 1) if (src.title, src.section) == (title, section)), None)
        if existing:
            remap[i] = existing
            continue
        retrieved = title.lower() in seen
        if not retrieved:
            warnings.append(f'Cited "{title}" but never retrieved it.')
        sources.append(Source(title, section, page_url(title, None if section in (LEAD, INFOBOX) else section), retrieved))
        remap[i] = len(sources)
    return sources, remap


def _cite(citations, remap: dict[int, int], warnings: list[str]) -> str:
    marks = []
    for c in _items(citations):
        c = int(c) if isinstance(c, str) and c.strip().isdigit() else c
        if c in remap:
            mark = f"[{remap[c]}]"
            if mark not in marks:
                marks.append(mark)
        else:
            warnings.append(f"Citation [{c}] doesn't match any source.")
    return "".join(marks)


def _point(item, bullet: str, remap: dict[int, int], warnings: list[str]) -> str:
    if not isinstance(item, dict):
        item = {"statement": str(item)}
    statement = _text(item.get("statement"))
    if not _items(item.get("citations")):
        warnings.append(f'No citation for "{statement[:60]}".')
    return f"{bullet} {statement} {_cite(item.get('citations'), remap, warnings)}".rstrip()


def _points(value) -> list[dict]:
    return [p if isinstance(p, dict) else {"statement": str(p)} for p in _items(value)]


def report_word_count(data: dict) -> int:
    """Words in a report's prose (overview and sections); tables and lists don't count."""
    points = _points(data.get("overview")) + [
        p for sec in _items(data.get("sections")) if isinstance(sec, dict) for p in _points(sec.get("points"))]
    return sum(len(_text(p.get("statement")).split()) for p in points)


# Assumptions entries that carry something the reader needs to interpret the answer (another
# reading, the definition or scope a figure depends on, a premise correction) rather than a method
# note. A phrase heuristic tuned on the 250 responses in evals/runs/suite_v3_sonnet46. It is English
# only and misses caveats phrased other ways; a miss just means no retry.
_READING_OR_SCOPE = re.compile("|".join([
    # another reading or referent
    r"\b(could|can|may|might) (also )?(mean|refer)\b", r"\brefers? to either\b", r"\bnot to be confused\b",
    r"\balso (known as|called|named|uses?|used)\b", r"\bother [\w-]+( [\w-]+)? (also|share|named|called)\b",
    r"\b(also|other)\b[^.;]*\bexists?\b",
    r"\balternative(ly)?\b", r"\bambiguous\b",
    # the definition or scope a figure depends on
    r"\bdepend(s|ing)? (heavily |largely |entirely )?on\b(?! the (source|methodology))",
    r"\bdefinitions? of\b", r"\bdefined as\b",
    r"\bin the [\w-]+( [\w-]+)? sense\b", r"\bif\b[^.;]*\b(is|are) (counted|included|excluded|considered|treated)\b",
    r"\bif '[^']+' means\b", r"\b(figure|population|total|count|number|area)s? (includes?|excludes?|covers?|counts?)\b",
    r"\binclud(e|es|ing) (those|both|only)\b", r"\b(this|the) answer (covers|uses|assumes|refers to|is for)\b",
    r"\b(sometimes|often|popularly|commonly) (popularly )?(cited|considered|regarded|described|called) as\b",
    r"\bsome (historians|scholars|sources|authors) (consider|regard|cite|argue|count)\b",
]), re.I)
_PREMISE = re.compile(r"\bthe question('s)? (assumes|premise|states)\b|\bpremise\b|\bmisconception\b", re.I)
_NEGATION = re.compile(r"\b(no|not|never|none|neither|nor)\b|n't\b", re.I)
# A reading chosen on priors ("taken as X, the most common meaning"): the alternative goes unnamed,
# so this fires even when the entry adds little to the answer.
_CHOICE = re.compile(r"\b(interpreted|taken|assumed|understood|read) (as|to (mean|refer to|be))\b|\brefers? to\b", re.I)
_PRIOR = re.compile(r"\b((most|more|far|overwhelmingly)( \w+ly)? (common|prominent|likely|widely|popular|famous"
                    r"|well[- ]known|usual)|predominant|primary|usual|default|best[- ]known|dominant)\b", re.I)
# The entry says the readings are already all covered, or that the user's own words settled it.
_SETTLED = re.compile(r"\b(both|all)\b[^.;]*\b(provided|given|addressed|covered|listed|included|presented)\b"
                      r"|\bclarifi|\bthe user (specified|mentioned|said)\b|\bbased on the context\b", re.I)
# Share of an entry's content words that the question, answer and reasoning don't already contain,
# at or above which a caveat-like entry counts as buried rather than a recap.
NOVELTY = 0.6
_STOPWORDS = set("""the and for that this with from which what when where while have has had been were was are
its their there these those than then they them into onto also only both some other such more most many much
not but can could may might would should will here question user answer wikipedia article""".split())


def _words(text: str) -> set[str]:
    """Content-word stems (first five letters), for a rough "does the answer already say this" check."""
    return {w[:5] for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in _STOPWORDS}


def buried_caveats(data: dict, question: str = "") -> list[str]:
    """Assumptions entries that look like something the reader needs to interpret the answer and
    that the headline text doesn't already say. Works on submit_answer and submit_report input."""
    head = _text(data.get("answer")) + " " + " ".join(_text(p.get("statement")) for p in _points(data.get("overview")))
    body = [_text(p.get("statement")) for p in _points(data.get("reasoning"))] + [
        _text(p.get("statement")) for sec in _items(data.get("sections")) if isinstance(sec, dict)
        for p in _points(sec.get("points"))]
    shown = _words(" ".join([question, head] + body))
    found = []
    for entry in _items(data.get("assumptions")):
        if not isinstance(entry, str) or _SETTLED.search(entry):
            continue
        words = _words(entry)
        new = len(words - shown) / len(words) if words else 0
        prior = any(_CHOICE.search(s) and _PRIOR.search(s) for s in re.split(r"(?<=[.;])\s", entry))
        caveat = _READING_OR_SCOPE.search(entry) or (_PREMISE.search(entry) and not _NEGATION.search(head))
        if prior or (caveat and new >= NOVELTY):
            found.append(entry.strip())
    return found


def _tail(data: dict, sources: list[Source]) -> list[str]:
    """Assumptions and sources, shared by answers and reports."""
    parts = []
    assumptions = [a.strip() for a in _items(data.get("assumptions")) if isinstance(a, str) and a.strip()]
    if assumptions:
        parts.append("**Assumptions & gaps:**\n" + "\n".join(f"- {a}" for a in assumptions))
    if sources:
        # Trailing double space = Markdown line break, so sources render one per line.
        parts.append("**Sources:**  \n" + "  \n".join(
            f"[{i}] {s.title}{' (lead)' if s.section == LEAD else f' § {s.section}'} — {s.url}"
            + ("" if s.retrieved else "  ⚠️ not retrieved")
            for i, s in enumerate(sources, 1)
        ))
    return parts


def render(data: dict, retrieved_titles: set[str]) -> tuple[str, list[Source], list[str]]:
    """Render submit_answer input to Markdown. Returns (markdown, sources, warnings)."""
    warnings: list[str] = []
    sources, remap = _build_sources(data, retrieved_titles, warnings)

    verdict = _text(data.get("verdict"))
    answer = _text(data.get("answer")).rstrip(".")
    if verdict:
        head = f"**Answer:** **{verdict}**" + (f": {answer}." if answer else "")
    else:
        head = f"**Answer:** **{answer}**"
    parts = [head]

    numbered = data.get("question_type") == "multi-hop"
    lines = [_point(item, f"{i}." if numbered else "-", remap, warnings)
             for i, item in enumerate(_points(data.get("reasoning")), 1)]
    if lines:
        parts.append("**Reasoning:**\n" + "\n".join(lines))
    parts += _tail(data, sources)
    return "\n\n".join(parts), sources, warnings


def render_report(data: dict, retrieved_titles: set[str]) -> tuple[str, list[Source], list[str]]:
    """Render submit_report input to Markdown. Returns (markdown, sources, warnings)."""
    warnings: list[str] = []
    sources, remap = _build_sources(data, retrieved_titles, warnings)

    parts = [f"## {_text(data.get('title'))}"]
    overview = " ".join(_point(p, "", remap, warnings).strip() for p in _points(data.get("overview")))
    if overview:
        parts.append(overview)

    for sec in _items(data.get("sections")):
        if not isinstance(sec, dict):
            continue
        body = "\n".join(_point(p, "-", remap, warnings) for p in _points(sec.get("points")))
        parts.append(f"### {_text(sec.get('heading'))}\n{body}")

    table = data.get("table")
    if isinstance(table, dict) and _items(table.get("columns")) and _items(table.get("rows")):
        cols = [str(c) for c in _items(table["columns"])] + ["Sources"]
        rows = [
            [str(c).replace("|", "\\|") for c in _items(r.get("cells"))] + [_cite(r.get("citations"), remap, warnings)]
            for r in _items(table["rows"]) if isinstance(r, dict)
        ]
        parts.append("\n".join(
            ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
            + ["| " + " | ".join(r) + " |" for r in rows]
        ))

    basis = data.get("selection_basis") if isinstance(data.get("selection_basis"), dict) else {}
    explanation = _text(basis.get("explanation"))
    if explanation and explanation.lower().rstrip(".") != "not applicable":
        parts.append(f"**How these were chosen:** {explanation} {_cite(basis.get('citations'), remap, warnings)}".rstrip())

    further = [f for f in _items(data.get("further_reading")) if isinstance(f, dict)]
    if further:
        parts.append("**Further reading:**\n" + "\n".join(
            f"- [{_text(f.get('title'))}]({page_url(_text(f.get('title')))}) — {_text(f.get('why'))}"
            for f in further
        ))
    parts += _tail(data, sources)
    return "\n\n".join(parts), sources, warnings
