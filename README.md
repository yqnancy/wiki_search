# wiki_search

Answers questions, checks claims and writes research briefs using only Wikipedia, with Claude (Sonnet 4.6 by default) driving a tool-use loop. No hosted search or RAG tools are used.

## How it works

```
question ─► Claude (Sonnet 4.6, adaptive thinking)
              │ plans: fact / claim check / multi-hop; writes keyword queries
              ├─► search_wikipedia(query)       top-N results + intros, disambiguation pages flagged
              │     └─ disambiguation page in top 3? ─► Sonnet 4.6 sub-call:
              │           could the question mean >1 entry?  yes → stop, ask the user
              │                                               no  → tell the agent which meaning
              ├─► read_article(title, section?)  lead + table of contents, or one section (or the "Infobox")
              └─► submit_answer(answer, verdict, reasoning[], sources[]) ─► rendered by code
```

| File | Purpose |
|---|---|
| `wiki_search/wikipedia.py` | MediaWiki API client: search, article fetch, section splitting |
| `wiki_search/agent.py` | `WikiQA`: agent loop, tools, disambiguation sub-call, budgets |
| `wiki_search/prompts.py` | System prompt and disambiguation prompt |
| `wiki_search/render.py` | `submit_answer` and `submit_report` schemas; renders answers and briefs to Markdown, merges duplicate sources, checks citations |
| `wiki_search/__main__.py` | CLI |
| `evals/research_set.json` | Research requests for the research-brief eval, each with a `must_cover` topic list |
| `evals/eval_set.json` | 18 fact-check questions (facts, claims, multi-hop, ambiguous, out-of-scope), each with gradeable expectations (`expected_answer`, `acceptable_answers`, `expected_verdict`, `should_clarify`) and quoted Wikipedia evidence pinned to a revision |
| `evals/verify_evidence.py` | Re-checks every evidence quote against current Wikipedia; `--write` pins revision IDs |
| `demo.ipynb` | Runs a single question and the eval set, and renders answers with tool traces |

## Two modes, chosen by the agent

- **Answers** (facts, claim checks, multi-hop): end with `submit_answer`, which gives a one-line answer or verdict, cited reasoning, and sources.
- **Research briefs** (overviews, histories, comparisons): the agent calls `plan_research`, which raises its budget from 10 articles / 20 tool calls to `research_max_articles` / `research_max_tool_calls` (25 / 40). It reads hub articles, sections and "See also" links, running independent calls in parallel, then calls `submit_report`: an overview, sections, an optional comparison table, how items were chosen (`selection_basis`), further reading, and sources. Briefs are recommended to be about 300–500 words; only one well over that (above 625 words) is sent back once to shorten.
- **Wikipedia before the model's own knowledge.** A brief whose selection ("the five most common ...") cites nothing is sent back once to look for a Wikipedia list or hub section to base it on; an imperfect Wikipedia source is preferred to the model's own judgment.
- **Claims checked against their sources by meaning** (`wiki_search/verify.py`). Before a brief is accepted, a separate model call (Sonnet 4.6 by default) compares every cited claim with the full text of the sections it cites, as the agent read them, and flags claims that are unsupported, contradicted or distorted, e.g. "height is normally distributed" citing a section that says height is log-normal. Flagged claims go back to the agent once, with what the source actually says. If the revised brief still has unsupported claims, they're listed in the brief's warnings; if the check can't run, that's a warning too. `--verify-claims all` extends the check to fact-check answers; `off` disables it.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env
```

## Usage

```bash
.venv/bin/python -m wiki_search -v "Who directed the Best Picture winner for the year the Berlin Wall fell?"
.venv/bin/python -m wiki_search            # interactive mode
```

Options: `--verify-claims research|all|off`, `--cache on|refresh|off`, `--max-articles N` (results per search and max distinct articles opened per question; default 10), `--max-tool-calls N` (default 20), `--effort low|medium|high|max` (default medium), `--model`.

From Python:

```python
from wiki_search import WikiQA

qa = WikiQA(max_articles=10)
answer = qa.ask("When was Mercury discovered?")
if answer.needs_clarification:
    print(answer.clarification.question, answer.clarification.options)
    answer = qa.ask("When was Mercury discovered?", clarification='By "Mercury" I mean the planet.')
print(answer.text)
```

The agent is told today's date (for ages and "how long ago" questions). Pass `WikiQA(today=datetime.date(2026, 1, 15))` to pin it, e.g. for reproducible evals.

## Eval set: how expected answers are made

Expected answers come from Wikipedia, not from a model's memory. Each item quotes the sentence that supports its answer, naming the article and section, and `evals/verify_evidence.py` confirms that every quote appears verbatim in that section and records the revision ID. If an article changes, the script reports the quote as MISSING. Items whose expected outcome is a judgment call rather than a fact (a verdict label, whether to ask for clarification) have `review.status: "needs_decision"` until a person decides; the decision is then recorded in `review.note` with status `"decided"`. `also_accept` describes an alternative behaviour that also passes.

## Research-brief eval: how judging works

The research-starting-point use case has its own eval, separate from the fact-verification eval (`evals/run.py`, `evals/judge.py`, `evals/data/`). The two share infrastructure (the Wikipedia client and cache, `WikiQA`), not eval sets.

### Inputs
`evals/research_set.json` holds research requests, grouped by the failure mode they probe:

| Category | Items | Probes |
|---|---|---|
| `research` | 2 | General briefs (history of LLMs, common distributions) |
| `goal` | 6 | Requests with a purpose (a trip, a report, a purchase, a debate, a new dog): the brief should end by helping the user go further toward it. Several topics have namesakes that must stay out (Kyoto Protocol, Monty Python, Greyhound Lines) |
| `reader-level` | 8 | Four topics asked twice, once with beginner hints and once with expert hints (mRNA vaccines, quantum entanglement / Bell tests, inflation, CRISPR): depth and terminology should adjust |
| `controversy` | 6 | Real disputes (saturated fat, intermittent fasting, minimum wage, rent control, the atomic bombings, Columbus's legacy): present the positions and where the evidence or consensus stands, without picking a winner where there isn't one |
| `limits` | 5 | Requests Wikipedia can't fully answer, each in a different way: an image that was never sent ("what is this lamp?"), a personal-taste question, an undefined "best" that can be partly answered by proposing definitions, a weather forecast, a personal life decision. These expect a short answer, not a brief |

Each item has `expected_mode` (`brief`, or `answer` where a short answer is right), `must_cover` (topics a good brief must include), `time_sensitive` (the brief must give the latest information, dated, and cite recently updated pages), and where relevant a `goal`, a `reader_level` (`beginner` / `expert`; empty means the default curious beginner-to-intermediate reader), a `pair` (its other-level twin), `must_not_include` (namesakes that would be off-topic) and `expected_limitations` (what an honest brief must say it can't answer). A person reviews each item before it's relied on (`review.status`). There are no expected answers: many different briefs can be good, so briefs are judged against rubrics instead.

### Graders
A brief gets five scores, each from 0 to 1:

| Grader | How | What it checks | How it scores |
|---|---|---|---|
| `brief_format` | code | Right mode for the item (`expected_mode`: a brief, or a short answer for questions that turn on a definition, value judgment, forecast, image or personal decision); prose (overview + sections) not far over the recommended ~500 words: it fails only above 625 (25% over), the same point at which the agent sends a brief back; every overview and section point cited; "How these were chosen" filled in, and cited or labelled as the agent's own judgment; 3–6 further-reading entries with short notes (20 words or fewer); a comparison table when the request compares ("compare", "versus", ...); no citation warnings | share of checks passed |
| `freshness` | code | Time-sensitive items only: every cited page had been edited within a year of the run, judged on the exact revision the agent read (recorded at run time). A recent edit is a proxy, not proof: it may be minor, and a stale section inside a busy page isn't detected, so `completeness` also checks the brief gives the latest information, dated | share of cited pages updated within a year |
| `citations` | judged, per claim | Each claim (overview and section points, table rows, a cited selection basis) against the full text of the sections it cites: **support** (full / partial / none), **faithful** (no distortion, overstatement, false precision or invented ranking) and **relevance** (the section is where the fact belongs, a better section the agent read exists, or it's tangential) | mean of support (1 / 0.5 / 0), faithful (1 / 0) and relevance (1 / 0.5 / 0) |
| `completeness` | judged | The key topics the request asks for and every `must_cover` item; for a stated **goal**, whether the brief helps the user go further toward it (concrete next steps, not just a generic reading list); and **limits and gaps** communicated clearly and honestly: what can't be answered and why (an image, a forecast, a value judgment, an undefined "best"), where sources are thin, while still offering what can be answered (e.g. candidate definitions of "best", each with its limits). Overclaiming a contested or subjective matter counts as missing | full 1, partial 0.5, missing 0, averaged; goal and limitation parts are also reported separately |
| `relevance` | judged, per point | Each point, including further-reading entries, labelled essential, supporting, marginal or off-topic. A **name collision** (a different thing sharing the topic's name, like the Kyoto Protocol in a history of Kyoto) is off-topic; the item's `must_not_include` namesakes are passed to the judge. "How these were chosen" and "Assumptions & gaps" count as supporting unless padded | essential and supporting 1, marginal 0.5, off-topic 0 |
| `style` | judged | Conciseness (no repetition within the sections or within the table; some overlap between the two is fine), clarity, and **level fit**: depth and terminology matched to the reader, taken from the item's `reader_level`, else the user's wording, else a curious beginner-to-intermediate reader (not a deep expert, not someone with zero background). Each 1–5. The brief's fixed sections are expected; only bloat within them is penalised | mean of the three, rescaled from 1–5 to 0–1 |

The judge sees exactly what the agent saw: cited sections are fetched from the same Wikipedia cache and truncated to the same length. Every judged verdict must name the problem (e.g. "the cited section says height is log-normal, not normal"), so a score can always be traced to specific text.

`correctness` and `format` in `evals/graders.py` are fact-check graders (an expected answer, the one-line answer layout). They don't apply to briefs and aren't run here.

### Who judges
Judging happens **in the Claude Code session** by default, not through the API. The runner turns each judgment into a Markdown **grading packet**, holding the rubric, the inputs (request, brief, cited source text) and a JSON schema for the verdict, and writes it to `evals/runs/<run>/grading/<id>/<grader>.md`. Claude reads each packet and writes its verdict next to it as `<grader>.json`. `--collect` then checks every verdict against its schema, computes the scores, and writes `results.jsonl` and `summary.md`. Grades record which model judged them; in-session grades come from whichever model the session runs.

For unattended runs, `--judge api` fills the same packets through the Anthropic API (Claude Opus 5, structured outputs). Safety classifiers occasionally decline benign grading requests (in the pilot, "bio" false positives on answers about mercury and light bulbs), so a declined request is retried on Claude Opus 4.8 and then with a short framing note. If every attempt is declined, the grade is recorded as an error, never as a zero.

### Running it
```bash
.venv/bin/python evals/run_eval.py                              # run every request, write grading packets
.venv/bin/python evals/run_eval.py --ids research-02 --graders brief_format style
# in Claude Code: "grade the pending packets in evals/runs/<run>"
.venv/bin/python evals/run_eval.py --collect evals/runs/<run>   # score, write summary.md
.venv/bin/python evals/run_eval.py --regrade evals/runs/<run>   # fresh packets for saved briefs (after a rubric change)
```

A run directory holds `answers.jsonl` (each brief with its structured fields, sections read, usage and cost), `traces/` (searches, reads and thinking), `grading/` (packets and verdicts), and after collecting, `results.jsonl` and `summary.md`.

## Wikipedia rate limits and caching

Wikimedia throttles bursts with HTTP 429. The client:

- **caches responses on disk** in `.cache/wikipedia.sqlite`. Reruns make few requests, and every run (e.g. comparing backbone models) reads identical article text. Use `--cache refresh` to refetch and overwrite, or `--cache off` to bypass. In Python, set `wiki_search.wikipedia.CACHE_MODE`; the environment variable is `WIKI_CACHE`.
- **searches in one request** (`generator=search` returns ranked results together with intros and disambiguation flags). Short queries of 3 words or fewer make one more request to look up `"<query> (disambiguation)"` directly.
- **sends requests one at a time**, spaced out (`WIKI_MIN_REQUEST_INTERVAL`, default 0.2s), and retries 429s using `Retry-After`. The demo notebook runs the eval set sequentially for the same reason.

Identify yourself as Wikimedia's [User-Agent policy](https://meta.wikimedia.org/wiki/User-Agent_policy) asks by adding this to `.env`:

```
WIKI_USER_AGENT=wiki-search/0.1 (your-contact@example.com)
```
