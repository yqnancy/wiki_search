"""Reusable graders for wiki_search outputs.

Every grader takes an eval item (from eval_set.json) and a result row (see run_eval.row_from_answer)
and produces a dict with a `score` in [0, 1] (None when it doesn't apply) plus details.

    correctness    code: expected answer / verdict / clarification behaviour      (answers)
    format         code: the fixed answer layout is followed                       (answers)
    brief_format   code: not far over length, citations, selection basis, further reading, table
    freshness      code: time-sensitive items only; every cited page edited within a year of the run
    citations      judged, per claim: supported by the cited text? a faithful summary (no
                   distortion or overstatement)? the most relevant section the agent read?
    completeness   judged: requested and `must_cover` topics, help toward a stated goal, and
                   limits communicated honestly (what can't be answered, thin sources)  (briefs)
    relevance      judged, per point (incl. further reading): essential / supporting /
                   marginal / off-topic; name collisions are off-topic
    style          judged rubric: conciseness, clarity, table fit (a table only where a side-by-side
                   comparison beats lists), fit to the reader's level
                   (item's reader_level, else the wording, else curious beginner-intermediate)

Judged graders are split into a request (rubric + inputs + JSON schema) and a scorer that turns
verdicts into a grade. Verdicts can come from two places:
    - in session: run_eval.py writes each request to a Markdown packet; Claude, in the Claude Code
      session, reads the packets and writes JSON verdict files; `run_eval.py --collect` scores them.
    - through the API: `Judge` sends each request to a model with structured outputs.
"""

from __future__ import annotations

import datetime
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wiki_search import wikipedia  # noqa: E402
from wiki_search.verify import cited_text  # noqa: E402
from wiki_search.agent import MAX_LEAD_CHARS, MAX_SECTION_CHARS, REPORT_MAX_WORDS, REPORT_WORD_TARGET  # noqa: E402
from wiki_search.render import VERDICTS  # noqa: E402


def as_list(value) -> list:
    """The agent occasionally sends a single string where the schema asks for a list."""
    if not value:
        return []
    return [value] if isinstance(value, str) else list(value)

ANSWER_CATEGORIES = {"fact", "claim", "multi-hop", "ambiguous", "out-of-scope"}
# Categories in the research-brief eval (evals/research_set.json).
RESEARCH_CATEGORIES = {"research", "goal", "reader-level", "controversy", "limits"}


# ---------------------------------------------------------------------------- requests and schemas

@dataclass
class JudgeRequest:
    key: str  # "<grader>" or "<grader>.<part>", unique within one eval item
    system: str
    user: str
    schema: dict


def _obj(properties: dict) -> dict:
    """JSON-schema object with every property required and no extras."""
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def _enum(*values: str) -> dict:
    return {"type": "string", "enum": list(values)}


def validate(value, schema: dict, path: str = "$") -> list[str]:
    """Minimal validator for the schema subset used here; returns a list of problems."""
    t = schema.get("type")
    if t == "object":
        if not isinstance(value, dict):
            return [f"{path}: expected object"]
        errs = [f"{path}: missing {k}" for k in schema["required"] if k not in value]
        errs += [f"{path}: unexpected {k}" for k in value if k not in schema["properties"]]
        for k, sub in schema["properties"].items():
            if k in value:
                errs += validate(value[k], sub, f"{path}.{k}")
        return errs
    if t == "array":
        if not isinstance(value, list):
            return [f"{path}: expected array"]
        return [e for i, v in enumerate(value) for e in validate(v, schema["items"], f"{path}[{i}]")]
    if t == "string":
        if not isinstance(value, str):
            return [f"{path}: expected string"]
        if "enum" in schema and value not in schema["enum"]:
            return [f"{path}: {value!r} not in {schema['enum']}"]
        return []
    if t == "integer":
        return [] if isinstance(value, int) and not isinstance(value, bool) else [f"{path}: expected integer"]
    if t == "boolean":
        return [] if isinstance(value, bool) else [f"{path}: expected boolean"]
    return []


# ---------------------------------------------------------------------------- inputs

def claims(row: dict) -> list[dict]:
    """Every cited (or citable) statement in an answer or brief, with where it appears."""
    s = row.get("structured") or {}
    out = []

    def add(text, citations, context):
        if text and text.strip():
            out.append({"id": len(out) + 1, "text": text.strip(), "citations": list(citations or []), "context": context})

    if row.get("mode") == "research":
        for p in s.get("overview", []):
            add(p.get("statement"), p.get("citations"), "Overview")
        for sec in s.get("sections", []):
            for p in sec.get("points", []):
                add(p.get("statement"), p.get("citations"), sec.get("heading", ""))
        table = s.get("table") or {}
        for r in table.get("rows", []):
            cells = "; ".join(f"{c}: {v}" for c, v in zip(table.get("columns", []), r.get("cells", [])))
            add(cells, r.get("citations"), "Comparison table row")
        basis = _selection_basis(s)
        if basis.get("citations"):  # uncited = declared as own judgment, nothing to check
            add(basis.get("explanation"), basis["citations"], "How items were chosen")
    else:
        for p in s.get("reasoning", []):
            add(p.get("statement"), p.get("citations"), "Reasoning")
    return out


def source_text(title: str, section: str) -> str:
    """The cited section's text, truncated as the agent saw it (lead, infobox or section)."""
    return cited_text(title, section)


def _prose(row: dict) -> str:
    """The rendered answer without its Sources list."""
    return row.get("text", "").split("**Sources:**")[0].strip()


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().casefold()


def _selection_basis(structured: dict) -> dict:
    """The agent sometimes sends selection_basis as a plain string; treat that as uncited."""
    basis = structured.get("selection_basis") or {}
    return {"explanation": basis, "citations": []} if isinstance(basis, str) else basis


def _research_answered(item: dict, row: dict) -> Optional[dict]:
    """Like _answered, but a "not covered" reply is still graded: for research requests, how the
    tool explains what it can't answer is part of what's being evaluated."""
    if row.get("needs_clarification") or not row.get("text"):
        return _skip("no answer to grade (asked for clarification)")
    return None


def _skip(reason: str) -> dict:
    return {"score": None, "reason": reason}


def _answered(row: dict) -> Optional[dict]:
    """A skip result when there's no answer to judge, else None."""
    if row.get("needs_clarification") or not row.get("text"):
        return _skip("no answer to grade (asked for clarification)")
    if (row.get("structured") or {}).get("question_type") == "not-covered":
        return _skip("answer says Wikipedia doesn't cover this; nothing cited to check")
    return None


# ---------------------------------------------------------------------------- correctness (code)

def grade_correctness(item: dict, row: dict) -> dict:
    """Did the tool reach the expected outcome?"""
    if item["category"] not in ANSWER_CATEGORIES:
        return _skip(f"not applied to {item['category']}")
    clarified = row.get("needs_clarification", False)
    s = row.get("structured") or {}

    if item.get("should_clarify"):
        if clarified:
            return {"score": 1.0, "reason": "asked for clarification as expected"}
        if item.get("also_accept_mentions"):  # an answer covering every listed meaning also passes
            text = _norm(row.get("text", ""))
            ok = all(_norm(m) in text for m in item["also_accept_mentions"])
            return {"score": float(ok), "reason": "answered covering every meaning" if ok else "answered without covering every meaning"}
        return {"score": 0.0, "reason": "should have asked for clarification"}
    if clarified:
        return {"score": 0.0, "reason": "asked for clarification unnecessarily"}

    if item["category"] == "out-of-scope":
        ok = s.get("question_type") == "not-covered"
        return {"score": float(ok), "reason": "declined as not covered" if ok else "did not mark as not covered"}

    checks = {}
    if item.get("expected_verdict"):
        checks["verdict"] = s.get("verdict") == item["expected_verdict"]
    if item.get("acceptable_answers"):
        haystack = _norm(s.get("answer", "") or row.get("text", ""))
        checks["answer"] = any(_norm(a) in haystack for a in item["acceptable_answers"])
    if not checks:
        return _skip("no expected answer to check")
    return {"score": sum(checks.values()) / len(checks), "checks": checks}


# ---------------------------------------------------------------------------- format (code)

ANSWER_MAX_WORDS = 12        # a name, number or short phrase
CLAIM_GIST_MAX_WORDS = 15    # the few words after a verdict
POINT_MAX_WORDS = 35         # one reasoning sentence


def grade_format(item: dict, row: dict) -> dict:
    """Does a Q&A answer follow the fixed layout: short answer line, verdict only for claims,
    reasoning points that are short and cited, sources that were actually retrieved?"""
    if item["category"] not in ANSWER_CATEGORIES:
        return _skip(f"not applied to {item['category']}")
    if row.get("needs_clarification"):
        return _skip("asked for clarification")
    s = row.get("structured")
    if not s:
        return {"score": 0.0, "failures": ["answered in prose instead of submit_answer"]}

    qtype = s.get("question_type")
    answer_words = len(str(s.get("answer", "")).split())
    reasoning = s.get("reasoning", [])
    checks = {
        "answer is short": answer_words <= (CLAIM_GIST_MAX_WORDS if qtype == "claim" else ANSWER_MAX_WORDS),
        "verdict only on claim checks": (s.get("verdict") in VERDICTS) == (qtype == "claim"),
        "reasoning points are one short sentence": all(len(p.get("statement", "").split()) <= POINT_MAX_WORDS for p in reasoning),
    }
    if qtype != "not-covered":
        checks["has reasoning"] = bool(reasoning)
        checks["every reasoning point is cited"] = all(p.get("citations") for p in reasoning)
        checks["has sources"] = bool(s.get("sources"))
    checks["citations valid and retrieved"] = not row.get("warnings")
    failures = [k for k, ok in checks.items() if not ok]
    if not checks["answer is short"]:
        failures[failures.index("answer is short")] = f"answer is short ({answer_words} words)"
    return {"score": round(sum(checks.values()) / len(checks), 3), "failures": failures}


# ---------------------------------------------------------------------------- brief format (code)

# The length is a recommendation (about 300-500 words of overview + sections; tables and lists
# don't count). A brief fails only when it is well over: 25% past the upper end, the same point at
# which the agent sends a brief back to shorten.
BRIEF_MAX_WORDS = REPORT_MAX_WORDS
FURTHER_READING_RANGE = (3, 6)
FURTHER_READING_WHY_MAX_WORDS = 20
_JUDGMENT_WORDS = ("judgment", "judgement", "my own", "not ranked", "doesn't rank", "does not rank")


def brief_word_count(structured: dict) -> int:
    points = list(structured.get("overview", [])) + [p for sec in structured.get("sections", []) for p in sec.get("points", [])]
    return sum(len(str(p.get("statement", "")).split()) for p in points)


def grade_brief_format(item: dict, row: dict) -> dict:
    """Does a research brief follow its structural rules? Exact, free checks that leave the
    style grader to judge only how the brief reads."""
    if item["category"] not in RESEARCH_CATEGORIES:
        return _skip(f"not applied to {item['category']}")
    if row.get("needs_clarification"):
        return _skip("asked for clarification")
    s = row.get("structured")
    if item.get("expected_mode") == "answer":  # definition, value, forecast, image or personal questions
        if s and row.get("mode") != "research":
            return _skip("answered with a short answer, as expected")
        return {"score": 0.0, "failures": ["expected a short answer (submit_answer), got a research brief"
                                           if row.get("mode") == "research" else "no structured answer submitted"]}
    if not s or row.get("mode") != "research":
        return {"score": 0.0, "failures": ["expected a research brief, but submit_report wasn't used"]}

    words = brief_word_count(s)
    points = list(s.get("overview", [])) + [p for sec in s.get("sections", []) for p in sec.get("points", [])]
    uncited = sum(1 for p in points if not p.get("citations"))
    basis = _selection_basis(s)
    explanation = str(basis.get("explanation", "")).strip()
    applicable = explanation and explanation.lower().rstrip(".") != "not applicable"
    reading = s.get("further_reading", [])
    long_whys = sum(1 for f in reading if len(str(f.get("why", "")).split()) > FURTHER_READING_WHY_MAX_WORDS)
    table = s.get("table") or {}

    checks = {
        f"prose not far over the recommended {REPORT_WORD_TARGET} words ({words}; fails above {BRIEF_MAX_WORDS})":
            words <= BRIEF_MAX_WORDS,
        f"every overview and section point cited ({uncited} uncited)": uncited == 0,
        "selection basis filled in": bool(explanation),
        "selection cited or labelled as own judgment":
            not applicable or bool(basis.get("citations")) or any(w in explanation.lower() for w in _JUDGMENT_WORDS),
        f"further reading has {FURTHER_READING_RANGE[0]}-{FURTHER_READING_RANGE[1]} entries ({len(reading)})":
            FURTHER_READING_RANGE[0] <= len(reading) <= FURTHER_READING_RANGE[1],
        f"further-reading notes are short ({long_whys} over {FURTHER_READING_WHY_MAX_WORDS} words)": long_whys == 0,
        "no citation warnings": not row.get("warnings"),
    }
    failures = [k for k, ok in checks.items() if not ok]
    return {"score": round(sum(checks.values()) / len(checks), 3), "words": words, "failures": failures}


# ---------------------------------------------------------------------------- freshness (code)

STALE_AFTER_DAYS = 365


def grade_freshness(item: dict, row: dict) -> dict:
    """For time-sensitive items: was every cited page updated within a year of the run?

    Uses the revision the agent actually read (recorded at run time in `source_revisions`).
    A recent edit is a proxy, not proof: it may be a minor or bot edit, and a single stale
    section inside a recently edited page isn't detected. Completeness separately checks that
    the brief gives the latest information, dated."""
    if not item.get("time_sensitive"):
        return _skip("item isn't time-sensitive")
    if row.get("needs_clarification") or not row.get("structured"):
        return _skip("no answer to grade")
    titles = list(dict.fromkeys(src.get("title", "") for src in row["structured"].get("sources") or []))
    if not titles:
        return _skip("nothing cited")
    run_date = datetime.date.fromisoformat(row.get("run_date") or datetime.date.today().isoformat())
    recorded = row.get("source_revisions") or {}
    pages = []
    for title in titles:
        revid = recorded.get(title)
        if revid is None:
            article = wikipedia.get_article(title)
            revid = article.revid if article else None
        stamp = wikipedia.revision_timestamp(revid) if revid else None
        age = (run_date - stamp.date()).days if stamp else None
        pages.append({"title": title, "revid": revid, "last_updated": stamp.date().isoformat() if stamp else None,
                      "age_days": age, "stale": age is None or age > STALE_AFTER_DAYS})
    fresh = sum(not p["stale"] for p in pages)
    return {"score": round(fresh / len(pages), 3), "stale": [p for p in pages if p["stale"]], "pages": pages}


# ---------------------------------------------------------------------------- citations (judged)

CITATION_SYSTEM = """\
You audit citations in answers produced by a research assistant that may only use Wikipedia.

For each claim you get the claim text, where it appears, and the full text of every Wikipedia section it cites. You also get the list of all sections the assistant read. Judge each claim on three things:

1. support: does the cited text state or directly imply every factual assertion in the claim?
   - "full": everything in the claim is in the cited text.
   - "partial": some assertions are supported, others are not in the cited text (even if true in reality).
   - "none": the cited text doesn't support the claim.
   Judge only against the cited text, not your own knowledge. Reasonable paraphrase and simple arithmetic from stated facts count as supported. Synthesis that combines facts from several cited sections is fine if each fact is in one of them.

2. faithful: is the claim a fair summary of what the source says? Mark false if the claim distorts the source: overstates certainty or scope, adds false precision (e.g. turns "since 2022" into a specific release year), attributes to the source a ranking or judgment it doesn't make, or changes the meaning. Judge this separately from support: a claim that merely goes beyond the text is partially supported but can still be faithful.

3. relevance: is the cited section the most relevant source for this claim?
   - "primary": the section is where this fact naturally belongs (its main subject).
   - "adequate": it supports the fact, but another section the assistant read would be a clearly better citation.
   - "tangential": it only mentions the fact in passing, or is about something else.
   When a better source exists among the sections the assistant read, name it in better_source as "Title § Section"; otherwise leave it empty.
   For claims in a section serving the user's stated goal (e.g. "For your visit"), the cited section must bear on that goal specifically, not just on the topic in general: a general-history section cited for advice about which sites to see is "tangential".

Keep `issue` to one sentence explaining any non-full, unfaithful or non-primary verdict; leave it empty when all three are best. Return one verdict per claim."""

CITATION_SCHEMA = _obj({"verdicts": {"type": "array", "items": _obj({
    "claim_id": {"type": "integer"},
    "support": _enum("full", "partial", "none"),
    "faithful": {"type": "boolean"},
    "relevance": _enum("primary", "adequate", "tangential"),
    "better_source": {"type": "string"},
    "issue": {"type": "string"},
})}})

_SUPPORT = {"full": 1.0, "partial": 0.5, "none": 0.0}
_RELEVANCE = {"primary": 1.0, "adequate": 0.5, "tangential": 0.0}


def request_citations(item: dict, row: dict, batch_size: int = 10) -> Union[dict, list[JudgeRequest]]:
    skip = _answered(row)
    if skip:
        return skip
    cited = [c for c in claims(row) if c["citations"]]
    if not cited:
        return _skip("no cited claims")
    sources = (row.get("structured") or {}).get("sources", [])
    read = "\n".join(f"- {r['title']} § {r['section']}" for r in row.get("reads", [])) or "(none; cited search-result intros only)"

    requests = []
    for part, start in enumerate(range(0, len(cited), batch_size), 1):
        batch = cited[start:start + batch_size]
        used = sorted({i for c in batch for i in c["citations"] if 1 <= i <= len(sources)})
        texts = "\n\n".join(
            f'<source n="{i}" title="{sources[i-1]["title"]}" section="{sources[i-1]["section"]}">\n'
            f'{source_text(sources[i-1]["title"], sources[i-1]["section"])}\n</source>'
            for i in used
        )
        listing = "\n".join(f'<claim id="{c["id"]}" context="{c["context"]}" cites="{c["citations"]}">{c["text"]}</claim>'
                            for c in batch)
        user = (f"<question>{row['question']}</question>\n\n<sections_the_assistant_read>\n{read}\n"
                f"</sections_the_assistant_read>\n\n<cited_sources>\n{texts}\n</cited_sources>\n\n<claims>\n{listing}\n</claims>")
        requests.append(JudgeRequest(f"citations.{part}", CITATION_SYSTEM, user, CITATION_SCHEMA))
    return requests


def score_citations(item: dict, row: dict, verdicts: list[dict]) -> dict:
    all_claims = claims(row)
    by_id = {c["id"]: c for c in all_claims}
    judged = sorted((v for part in verdicts for v in part["verdicts"]), key=lambda v: v["claim_id"])
    uncited = [{"claim_id": c["id"], "support": "none", "faithful": True, "relevance": "tangential",
                "better_source": "", "issue": "no citation"} for c in all_claims if not c["citations"]]
    rows = sorted(judged + uncited, key=lambda v: v["claim_id"])
    for v in rows:
        v["claim"] = by_id.get(v["claim_id"], {}).get("text", "")
    n = len(rows)
    support = sum(_SUPPORT[v["support"]] for v in rows) / n
    faithful = sum(v["faithful"] for v in rows) / n
    relevance = sum(_RELEVANCE[v["relevance"]] for v in rows) / n
    return {
        "score": round((support + faithful + relevance) / 3, 3),
        "support": round(support, 3), "faithfulness": round(faithful, 3), "relevance": round(relevance, 3),
        "claims": n, "uncited": len(uncited),
        "problems": [v for v in rows if v["support"] != "full" or not v["faithful"] or v["relevance"] != "primary"],
    }


# ---------------------------------------------------------------------------- completeness (judged)

COMPLETENESS_SYSTEM = """\
You check whether a research brief covers everything the user needs: what they asked for, help toward any goal they stated, and an honest account of what the brief can't answer.

1. Requested topics. List the key topics the user's request explicitly or clearly implicitly asks for. Be concrete: "compare the five most common X and say what each is best suited to model" asks for five named items, what each is suited to model, and a comparison between them. Don't invent requirements the user didn't express. Then add each item from the reviewer's must-cover list (source "must_cover").

2. The user's goal. If the user states a purpose (a trip, a report, a purchase, a debate, a decision), add one topic with source "goal": does the brief help them go further toward it? "full" needs a short section tied to that goal, made up mostly of cited Wikipedia facts that matter for that goal specifically (for a trip, facts about the particular sites; for a debate, the facts behind the other side's likely arguments), with at most one or two light recommendations. A generic further-reading list with no link to the goal, or a section of uncited advice, is "partial".

3. Limits and gaps. Some requests can't be fully answered from Wikipedia: they need an image, a forecast, a personal decision, a value judgment, or a definition the user didn't give (what makes an actor "the best"); or Wikipedia's coverage of part of the topic is thin, dated or disputed. Add a topic with source "limitation" for each such limit: the reviewer's expected limitations plus any you identify. Judge whether the response communicates it clearly and honestly:
   - "full": it says plainly what can't be answered or is uncertain and why, and still offers what can be answered (e.g. several candidate definitions of "best", each with facts and its own limits; climate averages instead of a forecast; where sources are thin, saying so, e.g. "sources are thin on X").
   - "partial": mentioned only in passing, vaguely, or only in a footnote-like "assumptions" entry the reader may skip.
   - "missing": ignored, or the response overclaims, e.g. presents a contested, subjective or unknowable matter as settled fact.
   Don't add limitation topics for ordinary factual requests that Wikipedia covers well.

For each topic, judge coverage: "full" (addressed adequately for a brief meant as a starting point), "partial" (mentioned but thin, or only part of it), or "missing". Keep `note` to one short sentence."""

COMPLETENESS_SCHEMA = _obj({"topics": {"type": "array", "items": _obj({
    "topic": {"type": "string"},
    "source": _enum("request", "must_cover", "goal", "limitation"),
    "coverage": _enum("full", "partial", "missing"),
    "note": {"type": "string"},
})}})

_COVERAGE = {"full": 1.0, "partial": 0.5, "missing": 0.0}


def request_completeness(item: dict, row: dict) -> Union[dict, list[JudgeRequest]]:
    if item["category"] not in RESEARCH_CATEGORIES and not item.get("must_cover"):
        return _skip(f"not applied to {item['category']} (correctness covers it)")
    skip = _research_answered(item, row)
    if skip:
        return skip
    must = "\n".join(f"- {t}" for t in item.get("must_cover", [])) or "(none given)"
    limits = "\n".join(f"- {t}" for t in item.get("expected_limitations", [])) or "(none given)"
    goal = item.get("goal") or "(none stated by the reviewer; infer from the request if there is one)"
    user = (f"<request>{row['question']}</request>\n\n<user_goal>{goal}</user_goal>\n\n"
            f"<must_cover>\n{must}\n</must_cover>\n\n<expected_limitations>\n{limits}\n</expected_limitations>\n\n"
            f"<response>\n{_prose(row)}\n</response>")
    return [JudgeRequest("completeness", COMPLETENESS_SYSTEM, user, COMPLETENESS_SCHEMA)]


def score_completeness(item: dict, row: dict, verdicts: list[dict]) -> dict:
    topics = verdicts[0]["topics"]
    if not topics:
        return _skip("judge listed no topics")

    def part(source):
        ts = [t for t in topics if t["source"] == source]
        return round(sum(_COVERAGE[t["coverage"]] for t in ts) / len(ts), 3) if ts else None

    score = sum(_COVERAGE[t["coverage"]] for t in topics) / len(topics)
    return {"score": round(score, 3), "goal": part("goal"), "limitations": part("limitation"),
            "topics": topics, "gaps": [t for t in topics if t["coverage"] != "full"]}


# ---------------------------------------------------------------------------- relevance (judged)

RELEVANCE_SYSTEM = """\
You check whether each point in a response is highly relevant to what the user asked.

Classify every point:
- "essential": directly answers what was asked; the response would be incomplete without it.
- "supporting": useful context that helps the reader understand the answer or act on their goal.
- "marginal": loosely related; a reader wouldn't miss it, and it takes space from what matters.
- "off_topic": not about what was asked.

Name collisions: a point about a different thing that merely shares a name or word with the topic is "off_topic", however accurate, e.g. the Kyoto Protocol in a history of the city of Kyoto, or Greyhound Lines in a brief about the dog breed. The reviewer lists known namesakes to watch for, but flag any you spot. Further-reading entries count as points too: they must be about the user's topic.

Points under "How items were chosen" and "Assumptions & gaps" are a required transparency feature of this tool: label them "supporting" unless they are padded or off-topic. Judge relevance to the request, not accuracy. Keep `note` to one short sentence, only for marginal or off-topic points; for a name collision, say what it was confused with."""

RELEVANCE_SCHEMA = _obj({"points": {"type": "array", "items": _obj({
    "point_id": {"type": "integer"},
    "label": _enum("essential", "supporting", "marginal", "off_topic"),
    "name_collision": {"type": "boolean"},
    "note": {"type": "string"},
})}})

_RELEVANCE_LABEL = {"essential": 1.0, "supporting": 1.0, "marginal": 0.5, "off_topic": 0.0}


def _relevance_points(row: dict) -> list[dict]:
    points = claims(row)
    s = row.get("structured") or {}
    basis = _selection_basis(s)
    if basis.get("explanation") and not basis.get("citations") and basis["explanation"].strip().lower().rstrip(".") != "not applicable":
        points.append({"id": len(points) + 1, "text": basis["explanation"], "context": "How items were chosen"})
    points += [{"id": len(points) + i + 1, "text": a, "context": "Assumptions & gaps"}
               for i, a in enumerate(as_list(s.get("assumptions")))]
    points += [{"id": len(points) + i + 1, "text": f"{f.get('title', '')}: {f.get('why', '')}", "context": "Further reading"}
               for i, f in enumerate(s.get("further_reading") or [])]
    return points


def request_relevance(item: dict, row: dict) -> Union[dict, list[JudgeRequest]]:
    skip = _research_answered(item, row) if item["category"] in RESEARCH_CATEGORIES else _answered(row)
    if skip:
        return skip
    points = _relevance_points(row)
    if not points:
        return _skip("no points to grade")
    namesakes = "\n".join(f"- {n}" for n in item.get("must_not_include", [])) or "(none listed)"
    listing = "\n".join(f'<point id="{p["id"]}" section="{p["context"]}">{p["text"]}</point>' for p in points)
    user = (f"<request>{row['question']}</request>\n\n<known_namesakes>\n{namesakes}\n</known_namesakes>\n\n"
            f"<points>\n{listing}\n</points>")
    return [JudgeRequest("relevance", RELEVANCE_SYSTEM, user, RELEVANCE_SCHEMA)]


def score_relevance(item: dict, row: dict, verdicts: list[dict]) -> dict:
    labels = verdicts[0]["points"]
    by_id = {p["id"]: p["text"] for p in _relevance_points(row)}
    for l in labels:
        l["point"] = by_id.get(l["point_id"], "")
    score = sum(_RELEVANCE_LABEL[l["label"]] for l in labels) / len(labels)
    return {"score": round(score, 3), "counts": {k: sum(l["label"] == k for l in labels) for k in _RELEVANCE_LABEL},
            "name_collisions": [l for l in labels if l.get("name_collision")],
            "flagged": [l for l in labels if l["label"] in ("marginal", "off_topic")]}


# ---------------------------------------------------------------------------- style (judged)

DEFAULT_READER = "curious beginner-to-intermediate"

STYLE_SYSTEM = """\
You review how a research brief reads for its intended reader. Content accuracy, citations and coverage are checked elsewhere.

The brief's layout is fixed by design: a title, an overview, sections, an optional comparison table, "How these were chosen", "Further reading" and sources. Don't penalise the presence of any of these parts; do penalise bloat within them (e.g. a long-winded selection note, or further-reading entries that run to full paragraphs). If the response is a short answer rather than a brief (e.g. for a request Wikipedia can't answer), judge it as a short answer.

Reader level. If the reviewer gives the reader's level, use it. Otherwise infer it from the user's wording: "I'm an immunology PhD student" or "I've taken graduate quantum mechanics" means expert; "explain it simply", "my kid asked me", "I'm new to this" means beginner. When the wording gives no clear hint, assume a curious beginner-to-intermediate reader: someone who knows everyday concepts and perhaps the basics of the field, but not its specialist terms. That is neither a deep expert nor someone with zero background.

Score each dimension from 1 (poor) to 5 (excellent):
- conciseness: no padding, repetition or restating within the prose sections or within the table (a fact repeated in a later section, the same information in two table columns, an overview that previews every section in detail). Some overlap between the table and the sections is acceptable; don't penalise it on its own. The recommended length is roughly 300-500 words of prose plus any table; judge repetition and padding, not a modest overage in words.
- clarity: easy to follow; well organised; sentences are direct; the main points are easy to find.
- table_fit: whether the brief uses a table exactly when one helps. A table helps when seeing several items side by side across the same dimensions (options, types, positions, each described on the same few attributes) is clearer than lists, whether or not the user said "compare". Score 5 when a table is present and makes the comparison easier to scan, or when there's no table and none would help. Score 1-2 when a table only restates the sections or lists items without comparable attributes (it adds nothing lists don't), or when a clear side-by-side comparison is buried in lists with no table. Explain a low score in the issues.
- level_fit: vocabulary, depth and pace match the reader level. For a beginner: plain words, specialist terms explained or avoided, concrete examples, nothing condescending. For an expert: precise technical terms used without explaining basics they already know, and enough depth (mechanisms, specifics, caveats) to be worth their time; a beginner-level brief for an expert scores low. For the default reader: explain specialist terms briefly, don't explain everyday ones.

List the most important concrete issues (quote the offending text briefly), at most five."""

STYLE_SCHEMA = _obj({
    "reader_level": _enum("beginner", "intermediate", "expert"),
    "level_basis": _enum("reviewer", "wording", "default"),
    "conciseness": {"type": "integer"}, "clarity": {"type": "integer"}, "level_fit": {"type": "integer"},
    "table_fit": {"type": "integer"},
    "issues": {"type": "array", "items": {"type": "string"}},
})


def request_style(item: dict, row: dict) -> Union[dict, list[JudgeRequest]]:
    if item["category"] not in RESEARCH_CATEGORIES:
        return _skip(f"not applied to {item['category']} (format covers it)")
    skip = _research_answered(item, row)
    if skip:
        return skip
    words = len(_prose(row).split())
    level = item.get("reader_level") or f"(not given: infer from the wording, default {DEFAULT_READER})"
    user = (f"<request>{row['question']}</request>\n\n<reader_level>{level}</reader_level>\n\n"
            f"<response words=\"{words}\">\n{_prose(row)}\n</response>")
    return [JudgeRequest("style", STYLE_SYSTEM, user, STYLE_SCHEMA)]


def score_style(item: dict, row: dict, verdicts: list[dict]) -> dict:
    v = verdicts[0]
    dims = {k: max(1, min(5, int(v[k]))) for k in ("conciseness", "clarity", "level_fit", "table_fit")}
    return {"score": round((sum(dims.values()) / len(dims) - 1) / 4, 3),  # 1..5 -> 0..1
            **dims, "reader_level": v["reader_level"], "level_basis": v["level_basis"],
            "words": len(_prose(row).split()), "issues": v["issues"]}


# ---------------------------------------------------------------------------- registry

CODE_GRADERS: dict[str, Callable[[dict, dict], dict]] = {
    "correctness": grade_correctness,
    "format": grade_format,
    "brief_format": grade_brief_format,
    "freshness": grade_freshness,
}
JUDGED_GRADERS: dict[str, tuple[Callable, Callable]] = {
    "citations": (request_citations, score_citations),
    "completeness": (request_completeness, score_completeness),
    "relevance": (request_relevance, score_relevance),
    "style": (request_style, score_style),
}
GRADER_NAMES = list(CODE_GRADERS) + list(JUDGED_GRADERS)
# The research-brief eval (run_eval.py). correctness and format are for fact-check answers.
RESEARCH_GRADERS = ["brief_format", "freshness", "citations", "completeness", "relevance", "style"]
