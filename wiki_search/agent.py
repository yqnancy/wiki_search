"""Wikipedia question-answering agent: a Claude tool-use loop over two tools.

    search_wikipedia(query)          ranked results with intros + disambiguation flags
    read_article(title, section?)    an article's lead + table of contents + "See also", or one section

The agent picks the mode. Fact questions and claim checks end with submit_answer. Research
requests (overviews, histories, comparisons) call plan_research, which raises the reading
budget, and end with submit_report.

When a search surfaces a disambiguation page, a separate Claude call decides whether the
user's question could reasonably mean more than one entry. If so, the run stops and returns
a clarifying question; otherwise the agent is told which meaning to use.
"""

from __future__ import annotations

import datetime
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

import anthropic

from . import verify, wikipedia
from .prompts import DISAMBIGUATION_PROMPT, SYSTEM_PROMPT
from .render import (INFOBOX, REPORT_TOOL, SUBMIT_TOOL, Source, buried_caveats, render, render_report,
                     report_word_count)

DEFAULT_MODEL = "claude-sonnet-4-6"

# Only disambiguation pages ranked this high trigger a check; lower-ranked ones are noise.
DISAMBIGUATION_RANK_CUTOFF = 3
MAX_SECTION_CHARS = 8000
MAX_LEAD_CHARS = 4000
# Disambiguation pages are sent whole as a compact entry list; 6,000 characters used to cut off
# entries on common pages (on 'Go', the programming language was past the cut).
MAX_DISAMBIGUATION_CHARS = 20000
DISAMBIGUATION_LINE_CHARS = 110
MAX_SEE_ALSO_CHARS = 1500
MAX_PARALLEL_TOOLS = 6
# Research briefs are recommended to be about 300-500 words. Only one well over that (25% past
# the upper end) is sent back once to shorten; the research eval uses the same threshold.
REPORT_WORD_TARGET = 500
REPORT_MAX_WORDS = int(REPORT_WORD_TARGET * 1.25)
# A submission can be sent back for several problems (length, citations, caveats left in assumptions);
# each check sends it back at most once, and at most this many corrective turns are spent in total.
MAX_CORRECTIONS = 2

TOOLS = [
    {
        "name": "search_wikipedia",
        "description": (
            "Search English Wikipedia (keyword full-text search). Returns the top results in "
            "Wikipedia's relevance order, each with title, URL, a short intro preview, and a "
            "DISAMBIGUATION PAGE flag where applicable. Use short keyword queries; operators such "
            "as intitle:, quoted phrases and -exclusions are supported."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keyword search query."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "read_article",
        "description": (
            "Read a Wikipedia article. Without `section`, returns the article's lead (opening "
            "summary), its table of contents, and its \"See also\" list of related articles. With "
            "`section`, returns that section's full text including subsections. Section names match "
            f"case-insensitively, and partial names work. When the table of contents lists \"{INFOBOX}\", "
            f"section=\"{INFOBOX}\" returns the article's infobox (key facts such as dates, runtime, "
            "capacity, as key: value lines), citable like any section."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Exact article title from search results."},
                "section": {
                    "type": "string",
                    "description": "Section heading from the table of contents. Omit to get the lead.",
                },
            },
            "required": ["title"],
            "additionalProperties": False,
        },
    },
    {
        "name": "plan_research",
        "description": (
            "Call this first when the request is research-style (an overview, history or explanation "
            "of a topic, or a comparison of several things) rather than a specific fact or claim. "
            "Records your plan and raises your reading budget. Finish research with submit_report."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "subtopics": {
                    "type": "array", "items": {"type": "string"},
                    "description": "The subtopics, eras or items the brief will cover.",
                },
                "candidate_articles": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Articles you expect to read, starting with hub or list articles.",
                },
            },
            "required": ["subtopics", "candidate_articles"],
            "additionalProperties": False,
        },
    },
    SUBMIT_TOOL,
    REPORT_TOOL,
]
FINAL_TOOLS = {"submit_answer": render, "submit_report": render_report}

DECISION_TOOL = {
    "name": "record_decision",
    "description": "Record whether the user's question is ambiguous with respect to the disambiguation page.",
    "input_schema": {
        "type": "object",
        "properties": {
            "ambiguous": {"type": "boolean"},
            "most_likely_title": {"type": "string"},
            "candidate_titles": {"type": "array", "items": {"type": "string"}},
            "clarifying_question": {"type": "string"},
            "rationale": {"type": "string"},
        },
        "required": ["ambiguous", "most_likely_title", "candidate_titles", "clarifying_question", "rationale"],
        "additionalProperties": False,
    },
}


@dataclass
class Clarification:
    term: str
    question: str
    options: list[str]
    rationale: str


@dataclass
class Answer:
    question: str
    text: str = ""  # Markdown rendered from `structured`
    structured: Optional[dict] = None  # fields of the submit_answer or submit_report call
    mode: str = "answer"  # "answer" or "research"
    clarification: Optional[Clarification] = None
    sources: list[Source] = field(default_factory=list)  # sources cited in the answer
    warnings: list[str] = field(default_factory=list)  # citation problems found while rendering
    trace: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    elapsed_s: float = 0.0
    model: str = DEFAULT_MODEL
    stop_reason: str = ""

    @property
    def needs_clarification(self) -> bool:
        return self.clarification is not None


class _NeedsClarification(Exception):
    def __init__(self, clarification: Clarification):
        self.clarification = clarification


class WikiQA:
    """Answers questions from Wikipedia.

    Args:
        model: backbone model for the agent loop.
        disambiguation_model: model for the disambiguation sub-call.
        max_articles: results returned per search, and the maximum number of distinct
            articles the agent may open per question. Raise for complex, many-entity questions.
        max_tool_calls: total tool-call budget per question (searches + reads).
        research_max_articles, research_max_tool_calls: the budgets that apply instead once
            the agent decides a request is research (by calling plan_research).
        effort: output_config effort for the agent loop ("low" | "medium" | "high" | "max").
        today: the date the agent treats as today (for ages, "how long ago", "current");
            defaults to datetime.date.today(). Pin it for reproducible evals.
        verify_claims: which submissions get a claim check before they're accepted: a separate
            model call compares each claim with the text of the sections it cites, and sends
            unsupported or distorted claims back once to fix. "research" (research briefs, the
            default), "all" (briefs and answers) or "off".
        verifier_model: model for the claim check.
        on_event: optional callback receiving progress strings (for the CLI's verbose mode).
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        disambiguation_model: str = DEFAULT_MODEL,
        max_articles: int = 10,
        max_tool_calls: int = 20,
        research_max_articles: int = 25,
        research_max_tool_calls: int = 40,
        effort: str = "medium",
        today: Optional[datetime.date] = None,
        verify_claims: str = "research",
        verifier_model: str = DEFAULT_MODEL,
        on_event: Optional[Callable[[str], None]] = None,
        client: Optional[anthropic.Anthropic] = None,
    ):
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.disambiguation_model = disambiguation_model
        self.max_articles = max_articles
        self.max_tool_calls = max_tool_calls
        self.research_max_articles = research_max_articles
        self.research_max_tool_calls = research_max_tool_calls
        self.effort = effort
        self.today = today
        self.verify_claims = verify_claims
        self.verifier_model = verifier_model
        self.on_event = on_event or (lambda msg: None)

    # ------------------------------------------------------------------ public

    def ask(self, question: str, clarification: Optional[str] = None) -> Answer:
        """Answer a question. If the answer requests clarification, call again with the
        user's reply as `clarification`."""
        run = _Run(self, question, clarification)
        return run.execute()


class _Run:
    """State for answering one question."""

    def __init__(self, qa: WikiQA, question: str, clarification: Optional[str]):
        self.qa = qa
        self.question = question
        self.full_question = (
            f"{question}\n\n(Clarification from the user: {clarification})" if clarification else question
        )
        self.answer = Answer(question=self.full_question, model=qa.model)
        self.usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
                      "cache_creation_input_tokens": 0, "llm_calls": 0}
        self.max_articles = qa.max_articles  # raised by plan_research
        self.max_tool_calls = qa.max_tool_calls
        self.tool_calls = 0
        self.lock = threading.RLock()  # tools run in parallel; guards the state below
        self.articles_opened: list[str] = []
        self.retrieved_titles: set[str] = set()  # articles whose text the agent has seen
        self.verify_errors: list[str] = []  # claim checks that failed to run, surfaced as warnings
        self.disambiguation_checked: dict[str, str] = {}  # title -> note given to the agent

    def execute(self) -> Answer:
        start = time.time()
        try:
            self._loop()
        except _NeedsClarification as e:
            self.answer.clarification = e.clarification
            self.answer.stop_reason = "needs_clarification"
        self.answer.elapsed_s = round(time.time() - start, 1)
        self.answer.usage = self.usage
        return self.answer

    # ------------------------------------------------------------------ loop

    def _loop(self) -> None:
        messages: list[dict] = [{"role": "user", "content": self.full_question}]
        today = self.qa.today or datetime.date.today()
        # The date goes in its own block after the cached prompt, so SYSTEM_PROMPT stays byte-identical.
        system = [
            {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": f"Today's date is {today:%A, %d %B %Y} ({today.isoformat()})."},
        ]
        nudged = False
        revised = False  # a report over REPORT_MAX_WORDS was sent back once
        rechecked = False  # a submission with citation warnings was sent back once
        promoted = False  # a submission with caveats buried in assumptions was sent back once
        sourced = False  # a brief whose selection rests on the agent's own judgment was sent back once
        tabled = False  # a brief with a table that doesn't compare items was sent back once
        completed = False  # a cut-off or incomplete submission was sent back once
        verified = False  # the claim check sent back unsupported claims once
        corrections = 0
        turns = 0
        # Budget of tool calls plus a few turns for planning and the final answer.
        while turns < self.max_tool_calls + 5:
            turns += 1
            response = self.qa.client.messages.create(
                model=self.qa.model,
                max_tokens=16000,
                system=system,
                tools=TOOLS,
                messages=messages,
                cache_control={"type": "ephemeral"},
                **_thinking_params(self.qa.model, self.qa.effort),
            )
            self._add_usage(response.usage)
            self.answer.stop_reason = response.stop_reason

            for block in response.content:
                if block.type == "thinking" and block.thinking:
                    self.answer.trace.append({"type": "thinking", "text": block.thinking})
                elif block.type == "text" and block.text.strip():  # e.g. the readings note before a search
                    self.answer.trace.append({"type": "text", "text": block.text.strip()})

            if response.stop_reason == "refusal":
                self.answer.text = "The model declined to answer this request."
                return

            submit = next((b for b in response.content
                           if b.type == "tool_use" and b.name in FINAL_TOOLS), None)
            if submit is not None:
                problems = []
                incomplete = _incomplete_submission(submit.name, submit.input, response.stop_reason)
                if incomplete and not completed:
                    # Never accept a submission cut off by the output limit, or one missing its
                    # evidence: it would render as an answer with no reasoning or sources.
                    completed = True
                    self.qa.on_event(f"submission incomplete ({incomplete}); asking to resubmit")
                    self.answer.trace.append({"type": "incomplete_submission", "issue": incomplete})
                    problems.append(
                        f"Your submission was incomplete: {incomplete}. Call {submit.name} again with every "
                        "field filled in. Keep it concise: short statements, only the reasoning the answer "
                        "needs, and no long deliberation before the call.")
                words = report_word_count(submit.input) if submit.name == "submit_report" else 0
                if words > REPORT_MAX_WORDS and not revised and corrections < MAX_CORRECTIONS:
                    revised = True
                    self.qa.on_event(f"brief is {words} words; asking for a shorter version")
                    self.answer.trace.append({"type": "revise", "words": words})
                    problems.append(
                        f"The overview and sections total {words} words, well over the recommended "
                        f"{REPORT_WORD_TARGET}. Shorten: state each fact once within the sections, merge or cut "
                        "minor points, keep every remaining point cited.")
                if not rechecked and corrections < MAX_CORRECTIONS:
                    _, _, warnings = FINAL_TOOLS[submit.name](submit.input, self.retrieved_titles)
                    if warnings:
                        rechecked = True
                        self.qa.on_event(f"{len(warnings)} citation warning(s); asking for a corrected submission")
                        self.answer.trace.append({"type": "recheck", "warnings": warnings})
                        problems.append(
                            "Citation problems:\n" + "\n".join(f"- {w}" for w in warnings) + "\nFix them: "
                            "every citation must be a 1-based index into `sources`, every source must be an "
                            "article you retrieved (exact title from the tools), and every reasoning item or "
                            "point needs at least one citation. Move any statement you can't cite to "
                            "`assumptions`, labelled if it is background knowledge.")
                if not promoted and corrections < MAX_CORRECTIONS:
                    buried = buried_caveats(submit.input, self.full_question)
                    if buried:
                        promoted = True
                        self.qa.on_event(f"{len(buried)} caveat(s) only in assumptions; asking to move them up")
                        self.answer.trace.append({"type": "promote", "entries": buried})
                        body = "answer or reasoning" if submit.name == "submit_answer" else "overview or sections"
                        problems.append(
                            "These `assumptions` entries seem to hold something the reader needs to interpret "
                            "the answer (another reading of the question, the definition or scope the answer "
                            "depends on, or a correction to the question's premise):\n"
                            + "\n".join(f"- {b}" for b in buried) + "\n"
                            f"Readers skim past assumptions, so say it in the {body}: the reading, definition or "
                            "correction you used and the main alternative, with its answer if that differs. Keep it "
                            "to a short clause so the headline stays short, and cite it (search or read first if "
                            "you need a source). Leave only method notes and gaps in Wikipedia's coverage in "
                            "`assumptions`; an entry that is only a method note can stay. Don't fix this by "
                            "deleting the entry: if you picked the most likely reading, name the other one too.")
                table_issue = _table_problem(submit.input) if submit.name == "submit_report" else None
                if not tabled and table_issue:
                    tabled = True
                    self.qa.on_event(f"table doesn't compare items ({table_issue}); asking to drop or fix it")
                    self.answer.trace.append({"type": "table_check", "issue": table_issue})
                    problems.append(
                        f"The table {table_issue}. Keep a table only if several items are each described on the "
                        "same attributes and the reader would scan across rows to compare them; then give its "
                        "`purpose` in one sentence and at least two attribute columns besides the item name. "
                        "Otherwise set `table` to null: a table that restates the sections adds nothing to them.")
                if not sourced and submit.name == "submit_report" and _unsourced_selection(submit.input):
                    sourced = True
                    self.qa.on_event("selection rests on own judgment; asking to look for a Wikipedia source")
                    self.answer.trace.append({"type": "source_selection"})
                    problems.append(
                        "`selection_basis` rests on your own judgment, with no citation. Wikipedia usually has "
                        "something to base the choice on: search for an article or section that lists, groups or "
                        "names the common, main or key items (e.g. a \"List of ...\" article, or a section of the "
                        "topic's main or hub article on common types or major milestones). Read it, base the "
                        "selection on it even if it isn't a perfect ranking, and cite it. Keep your own judgment "
                        "only if nothing turns up; then say what you searched for.")
                if not problems and not verified and self._should_verify(submit.name):
                    issues = self._check_claims(submit.name, submit.input)
                    if issues:
                        verified = True
                        self.qa.on_event(f"{len(issues)} claim(s) not backed by their citations; asking for fixes")
                        problems.append(
                            "A check of each claim against the text of the sections it cites found claims the "
                            "sources don't back:\n" + verify.describe(issues) + "\nFix each one: say what the "
                            "source says, cite a section you read that supports it, or drop the unsupported part. "
                            "Prefer what Wikipedia says over your own knowledge.")
                if problems:
                    corrections += 1
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": b.id, "is_error": True, "content": (
                            "Not submitted.\n\n" + "\n\n".join(problems) + f"\n\nCall {submit.name} again."
                        ) if b.id == submit.id else "Not run."}
                        for b in response.content if b.type == "tool_use"
                    ]})
                    continue
                leftover = []
                if verified:  # the brief was revised after the claim check: check it once more
                    leftover = [f'Claim not backed by its citation ({p["kind"]}): "{p["claim"]}". '
                                f'The cited text: {p["source_says"]}'
                                for p in self._check_claims(submit.name, submit.input)]
                self._submit(submit.name, submit.input, leftover)
                return

            text = "".join(b.text for b in response.content if b.type == "text").strip()
            if response.stop_reason != "tool_use":
                if nudged:  # answered in prose twice; keep the prose rather than loop
                    self.answer.text = text
                    return
                nudged = True
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": (
                    "Submit your final answer with the submit_answer tool, or submit_report for a research brief.")})
                continue

            messages.append({"role": "assistant", "content": response.content})
            calls = [b for b in response.content if b.type == "tool_use"]
            # Independent calls in one turn (typically several reads) run concurrently;
            # Wikipedia requests stay paced by the client, and cached reads return instantly.
            with ThreadPoolExecutor(max_workers=MAX_PARALLEL_TOOLS) as pool:
                futures = [pool.submit(self._run_tool, b.name, b.input) for b in calls]
            outputs = [f.result() for f in futures]  # re-raises _NeedsClarification
            results = []
            for block, (output, is_error) in zip(calls, outputs):
                result = {"type": "tool_result", "tool_use_id": block.id, "content": output}
                if is_error:
                    result["is_error"] = True
                results.append(result)
            messages.append({"role": "user", "content": results})

        self.answer.text = "Stopped: the agent did not finish within its turn limit."
        self.answer.stop_reason = "turn_limit"

    def _should_verify(self, tool: str) -> bool:
        mode = self.qa.verify_claims
        return mode == "all" or (mode == "research" and tool == "submit_report")

    def _check_claims(self, tool: str, data: dict) -> list[dict]:
        """Claims the cited sections don't back, by meaning (see verify.py)."""
        try:
            problems, usage = verify.check_claims(self.qa.client, self.qa.verifier_model, self.full_question, tool, data)
        except Exception as e:  # a safeguard (API error, malformed output): don't fail the answer, don't hide the gap
            self.answer.trace.append({"type": "error", "tool": "verify", "error": str(e)})
            self.verify_errors.append(f"Claim check didn't run, so claims weren't checked against their sources: {e}")
            return []
        if usage is not None:
            self._add_usage(usage)
        self.answer.trace.append({"type": "verify", "problems": problems})
        return problems

    def _submit(self, tool: str, data: dict, extra_warnings: Optional[list[str]] = None) -> None:
        self.answer.structured = data
        if tool == "submit_report":
            self.answer.mode = "research"
        self.answer.text, self.answer.sources, self.answer.warnings = FINAL_TOOLS[tool](data, self.retrieved_titles)
        self.answer.warnings += (extra_warnings or []) + self.verify_errors
        self.answer.stop_reason = "answered"
        self.answer.trace.append({"type": "submit", "tool": tool, "warnings": self.answer.warnings})

    def _add_usage(self, usage) -> None:
        with self.lock:
            self._add_usage_locked(usage)

    def _add_usage_locked(self, usage) -> None:
        self.usage["llm_calls"] += 1
        for key in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            self.usage[key] += getattr(usage, key, 0) or 0

    # ------------------------------------------------------------------ tools

    def _run_tool(self, name: str, args: dict) -> tuple[str, bool]:
        if name == "plan_research":
            return self._plan(args), False
        with self.lock:
            self.tool_calls += 1
            over_budget = self.tool_calls > self.max_tool_calls
        if over_budget:
            return ("Tool budget exhausted. Submit now (submit_answer or submit_report) using what you "
                    "have already read, and list what you could not cover under assumptions."), True
        try:
            if name == "search_wikipedia":
                return self._search(args["query"]), False
            if name == "read_article":
                return self._read(args["title"], args.get("section")), False
            return f"Unknown tool: {name}", True
        except _NeedsClarification:
            raise
        except Exception as e:  # network errors, API errors: let the model recover
            with self.lock:
                self.answer.trace.append({"type": "error", "tool": name, "args": args, "error": str(e)})
            return f"Tool error: {e}", True

    def _plan(self, args: dict) -> str:
        self.qa.on_event("plan research: " + "; ".join(args.get("subtopics", [])))
        with self.lock:
            self.answer.mode = "research"
            self.max_articles = max(self.max_articles, self.qa.research_max_articles)
            self.max_tool_calls = max(self.max_tool_calls, self.qa.research_max_tool_calls)
            self.answer.trace.append({"type": "plan", **args})
        return (f"Research mode: you may open up to {self.max_articles} articles and make "
                f"{self.max_tool_calls} tool calls in total. Read independent articles in parallel. "
                "Finish with submit_report.")

    def _search(self, query: str) -> str:
        self.qa.on_event(f"search: {query}")
        results = wikipedia.search(query, limit=self.qa.max_articles)
        with self.lock:
            self.answer.trace.append({"type": "search", "query": query, "results": [r.title for r in results]})
            self.retrieved_titles.update(r.title for r in results)  # intros are citable as "(lead)"
        if not results:
            return f'No results for "{query}". Try different keywords.'

        lines = [f'Results for "{query}" (Wikipedia relevance order):']
        for r in results:
            flag = "  [DISAMBIGUATION PAGE]" if r.is_disambiguation else ""
            lines.append(f"\n[{r.rank}] {r.title}{flag}\n    URL: {r.url}\n    Intro: {r.intro}")

        notes = [
            self._check_disambiguation(r.title, query)
            for r in results
            if r.is_disambiguation and r.rank <= DISAMBIGUATION_RANK_CUTOFF
        ]
        if notes:
            lines.append("\n" + "\n".join(notes))
        return "\n".join(lines)

    def _read(self, title: str, section: Optional[str]) -> str:
        self.qa.on_event(f"read: {title}" + (f" § {section}" if section else ""))
        article = wikipedia.get_article(title)
        if article is None:
            return f'No article titled "{title}". Use an exact title from search results.'

        with self.lock:
            if article.title not in self.articles_opened:
                if len(self.articles_opened) >= self.max_articles:
                    return (f"Article limit reached ({self.max_articles} distinct articles per question). "
                            f"Already opened: {', '.join(self.articles_opened)}. Answer from these.")
                self.articles_opened.append(article.title)
            self.retrieved_titles.update({title, article.title})  # requested title may be a redirect

        if article.is_disambiguation:
            note = self._check_disambiguation(article.title, f"read_article({title})")
            entries = _disambiguation_entries(article)
            return f"{article.title} is a disambiguation page.\n{note}\n\nEntries:\n{entries}"

        if section and section.strip().lower() == INFOBOX.lower():
            fields = wikipedia.get_infobox(article.title)
            if not fields:
                return f"{article.title} has no infobox. Available sections:\n" + "\n".join(article.toc())
            self._record_read(article.title, INFOBOX)
            body = "\n".join(f"{k}: {v}" for k, v in fields)
            return f"# {article.title} § {INFOBOX}\nURL: {article.url}\n\n{_truncate(body, MAX_SECTION_CHARS)}"

        if not section:
            self._record_read(article.title, "(lead)")
            toc = "\n".join(([INFOBOX] if _has_infobox(article.title) else []) + article.toc()) or "(no sections)"
            out = (f"# {article.title}\nURL: {article.url}\n\n## Lead\n{_truncate(article.lead, MAX_LEAD_CHARS)}"
                   f"\n\n## Table of contents\n{toc}")
            see_also = article.find_section("See also")
            if see_also and see_also[0].heading.lower() == "see also" and see_also[1].strip():
                out += f"\n\n## See also (related articles)\n{_truncate(see_also[1], MAX_SEE_ALSO_CHARS)}"
            return out

        found = article.find_section(section)
        if found is None:
            return (f'No section matching "{section}" in {article.title}. Available sections:\n'
                    + "\n".join(article.toc()))
        head, text = found
        url = wikipedia.page_url(article.title, head.heading)
        self._record_read(article.title, head.heading)
        return f"# {article.title} § {head.heading}\nURL: {url}\n\n{_truncate(text, MAX_SECTION_CHARS)}"

    def _record_read(self, title: str, section: str) -> None:
        with self.lock:
            self.answer.trace.append({"type": "read", "title": title, "section": section})

    # ------------------------------------------------------------------ disambiguation

    def _check_disambiguation(self, title: str, context: str) -> str:
        """Ask a separate model call whether the question could mean several entries.

        Raises _NeedsClarification if so; otherwise returns a note steering the agent."""
        with self.lock:  # held for the whole check so concurrent searches don't repeat it
            return self._check_disambiguation_locked(title, context)

    def _check_disambiguation_locked(self, title: str, context: str) -> str:
        if title in self.disambiguation_checked:
            return self.disambiguation_checked[title]

        self.qa.on_event(f"disambiguation check: {title}")
        page = wikipedia.get_article(title)
        entries = _disambiguation_entries(page)

        response = self.qa.client.messages.create(
            model=self.qa.disambiguation_model,
            max_tokens=1024,
            system=DISAMBIGUATION_PROMPT,
            tools=[DECISION_TOOL],
            tool_choice={"type": "tool", "name": "record_decision"},
            messages=[{"role": "user", "content": (
                # Only the question: the agent's own query can already carry its guess at the
                # meaning ("2026 Masters Tournament golf"), which biased this check.
                f"<user_question>\n{self.full_question}\n</user_question>\n\n"
                f"<disambiguation_page title=\"{title}\">\n{entries}\n</disambiguation_page>"
            )}],
        )
        self._add_usage(response.usage)
        decision = next(b.input for b in response.content if b.type == "tool_use")
        self.answer.trace.append({"type": "disambiguation", "page": title, "context": context, **decision})

        if decision["ambiguous"]:
            raise _NeedsClarification(Clarification(
                term=title,
                question=decision["clarifying_question"],
                options=decision["candidate_titles"],
                rationale=decision["rationale"],
            ))

        note = (f'Disambiguation check for "{title}": the question most likely refers to '
                f'"{decision["most_likely_title"]}" ({decision["rationale"]}). Use that meaning.')
        self.disambiguation_checked[title] = note
        return note


# Haiku 4.5 has no adaptive thinking or effort parameter; it takes a fixed thinking budget.
_HAIKU_THINKING_BUDGET = {"low": 2048, "medium": 4096, "high": 8192, "max": 12000}


def _thinking_params(model: str, effort: str) -> dict:
    if model.startswith("claude-haiku-4-5"):
        return {"thinking": {"type": "enabled", "budget_tokens": _HAIKU_THINKING_BUDGET.get(effort, 4096)}}
    return {"thinking": {"type": "adaptive"}, "output_config": {"effort": effort}}


def _has_infobox(title: str) -> bool:
    try:
        return bool(wikipedia.get_infobox(title))
    except Exception:  # an optional extra; don't fail the lead read over it
        return False


def _incomplete_submission(tool: str, data: dict, stop_reason: str) -> Optional[str]:
    """Why a final submission can't be accepted as-is, or None."""
    if stop_reason == "max_tokens":
        return "it was cut off by the output limit before it finished"
    if tool == "submit_answer" and data.get("question_type") != "not-covered":
        missing = [f for f in ("reasoning", "sources") if not data.get(f)]
        if missing:
            return f"{' and '.join(missing)} missing"
    if tool == "submit_report":
        missing = [f for f in ("overview", "sections", "sources") if not data.get(f)]
        if missing:
            return f"{', '.join(missing)} missing"
    return None


def _disambiguation_entries(page) -> str:
    """A disambiguation page as a compact list: section headings and one line per entry, long
    lines trimmed, "See also" dropped. Ends with a count if the list still overflows."""
    if page is None:
        return "(page not found)"
    lines = []
    for sec in page.sections:
        if sec.heading.lower() in ("see also", "references", "external links"):
            continue
        if sec.level > 1:
            lines.append(f"{sec.heading}:")
        for line in sec.text.splitlines():
            line = line.strip()
            if line:
                lines.append(line if len(line) <= DISAMBIGUATION_LINE_CHARS
                             else line[:DISAMBIGUATION_LINE_CHARS].rsplit(" ", 1)[0] + " …")
    out, used = [], 0
    for i, line in enumerate(lines):
        if used + len(line) + 1 > MAX_DISAMBIGUATION_CHARS:
            out.append(f"[… {len(lines) - i} more lines not shown]")
            break
        out.append(line)
        used += len(line) + 1
    return "\n".join(out)


def _table_problem(data: dict) -> Optional[str]:
    """Why a brief's table isn't a real comparison, or None. A table needs a stated purpose,
    at least two attribute columns besides the item column, and at least two rows."""
    table = data.get("table")
    if not isinstance(table, dict) or not table.get("rows"):
        return None
    if not str(table.get("purpose") or "").strip():
        return "has no stated purpose (the comparison a reader makes across its rows)"
    if len(table.get("columns") or []) < 3:
        return "has fewer than two attribute columns besides the item name"
    if len(table.get("rows") or []) < 2:
        return "has fewer than two rows to compare"
    return None


def _unsourced_selection(data: dict) -> bool:
    """True when a brief makes a selection but cites nothing for it."""
    basis = data.get("selection_basis")
    if isinstance(basis, str):  # sent as plain text: an explanation with no citations
        basis = {"explanation": basis, "citations": []}
    elif not isinstance(basis, dict):
        basis = {}
    explanation = str(basis.get("explanation", "")).strip()
    return bool(explanation) and explanation.lower().rstrip(".") != "not applicable" and not basis.get("citations")


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + f"\n… [truncated; {len(text) - limit:,} more characters]"
