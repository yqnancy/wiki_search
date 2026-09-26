"""Check that an answer's claims are backed, in meaning, by the Wikipedia sections they cite.

String checks (is the cited article one the agent retrieved?) can't tell whether a section
actually says what the claim says. This module gives a separate model call each claim together
with the full text of its cited sections, as the agent saw them, and asks it to flag claims the
text doesn't support or that distort it: e.g. a claim that contradicts the section it cites
that says otherwise.
"""

from __future__ import annotations

from typing import Optional

from . import wikipedia
from .render import INFOBOX

# Mirrors the limits the agent reads with (agent.MAX_LEAD_CHARS / MAX_SECTION_CHARS).
LEAD_CHARS = 4000
SECTION_CHARS = 8000

VERIFY_PROMPT = """\
You check a research assistant's answer before it reaches the reader. The assistant may only use Wikipedia, and every claim cites Wikipedia sections by number. You get each claim and the full text of the sections it cites, exactly as the assistant saw them.

Flag a claim when, judged by meaning against the cited text only (not your own knowledge):
- "unsupported": a factual assertion in it isn't stated or directly implied by the cited text, e.g. examples, dates, names or attributions the section doesn't give;
- "contradicted": the cited text says something different, e.g. the claim says a bridge is the longest in Europe today but the section says it was the longest when it opened;
- "distorted": it overstates or changes what the source says: "the most used" where the source says "often used", a specific year where the source says "in the early 1990s", a ranking or judgment the source doesn't make, a condition the source attaches that the claim drops.
Paraphrase, simple arithmetic from stated facts, and synthesis of facts that each appear in one of the cited sections are fine. Don't flag style, relevance, or claims that are merely incomplete.

For each flagged claim, quote or closely paraphrase what the cited text actually says in `source_says` (or say it's silent), and in `fix` say how to repair it: correct the wording to match the source, cite a different section you can see supports it, or drop the unsupported part. Record your findings with `record_claim_checks`; an empty list means every claim is backed."""

VERIFY_TOOL = {
    "name": "record_claim_checks",
    "description": "Record the claims that their cited sections don't back.",
    "input_schema": {
        "type": "object",
        "properties": {
            "problems": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "integer"},
                        "kind": {"type": "string", "enum": ["unsupported", "contradicted", "distorted"]},
                        "source_says": {"type": "string"},
                        "fix": {"type": "string"},
                    },
                    "required": ["claim_id", "kind", "source_says", "fix"],
                },
            },
        },
        "required": ["problems"],
    },
}


def claims(tool: str, data: dict) -> list[dict]:
    """The cited statements in a submit_answer or submit_report call, numbered from 1."""
    out = []

    def add(text, citations, where):
        if isinstance(text, str) and text.strip() and citations:
            out.append({"id": len(out) + 1, "text": text.strip(), "citations": list(citations), "where": where})

    if tool == "submit_report":
        for p in data.get("overview") or []:
            add(p.get("statement"), p.get("citations"), "overview")
        for sec in data.get("sections") or []:
            for p in sec.get("points") or []:
                add(p.get("statement"), p.get("citations"), f"section '{sec.get('heading', '')}'")
        table = data.get("table") or {}
        for r in table.get("rows") or []:
            cells = "; ".join(f"{c}: {v}" for c, v in zip(table.get("columns", []), r.get("cells", [])))
            add(cells, r.get("citations"), "table row")
        basis = data.get("selection_basis") if isinstance(data.get("selection_basis"), dict) else {}
        add(basis.get("explanation"), basis.get("citations"), "selection_basis")
    else:
        for p in data.get("reasoning") or []:
            add(p.get("statement"), p.get("citations"), "reasoning")
    return out


def cited_text(title: str, section: str) -> str:
    """A cited section's text, truncated as the agent read it."""
    article = wikipedia.get_article(title)
    if article is None:
        return "(article not found)"
    if section in ("(lead)", "", None):
        return article.lead[:LEAD_CHARS]
    if section.strip().lower() == INFOBOX.lower():
        fields = wikipedia.get_infobox(article.title)
        return "\n".join(f"{k}: {v}" for k, v in fields)[:SECTION_CHARS] or "(no infobox)"
    found = article.find_section(section)
    return found[1][:SECTION_CHARS] if found else "(section not found)"


def check_claims(client, model: str, question: str, tool: str, data: dict) -> tuple[list[dict], Optional[object]]:
    """Returns (problems, usage). Each problem carries the claim text, its kind, what the source
    says, and a suggested fix. Returns no problems if there is nothing cited to check."""
    items = claims(tool, data)
    sources = data.get("sources") or []
    if not items or not sources:
        return [], None
    used = sorted({i for c in items for i in c["citations"] if isinstance(i, int) and 1 <= i <= len(sources)})
    texts = "\n\n".join(
        f'<source n="{i}" title="{sources[i - 1].get("title", "")}" section="{sources[i - 1].get("section", "")}">\n'
        f'{cited_text(sources[i - 1].get("title", ""), sources[i - 1].get("section", ""))}\n</source>'
        for i in used
    )
    listing = "\n".join(f'<claim id="{c["id"]}" in="{c["where"]}" cites="{c["citations"]}">{c["text"]}</claim>'
                        for c in items)
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=VERIFY_PROMPT,
        tools=[VERIFY_TOOL],
        tool_choice={"type": "tool", "name": "record_claim_checks"},
        messages=[{"role": "user", "content": (
            f"<question>{question}</question>\n\n<cited_sources>\n{texts}\n</cited_sources>\n\n"
            f"<claims>\n{listing}\n</claims>")}],
    )
    result = next((b.input for b in response.content if b.type == "tool_use"), {"problems": []})
    by_id = {c["id"]: c for c in items}
    problems = []
    for p in result.get("problems") or []:
        claim = by_id.get(p.get("claim_id"))
        if claim:
            problems.append({**p, "claim": claim["text"], "where": claim["where"]})
    return problems, response.usage


def describe(problems: list[dict]) -> str:
    """The problems as instructions for the agent."""
    return "\n".join(
        f'- In {p["where"]}: "{p["claim"]}" is {p["kind"]}. The cited text: {p["source_says"]} Fix: {p["fix"]}'
        for p in problems
    )
