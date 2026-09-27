# Evals

This folder holds two separate evals, one per use case. They share infrastructure (the Wikipedia client and cache, `WikiQA`) but not eval sets:

- **Fact verification**: five eval sets, `run.py`, and an LLM judge (`judge.py`) that labels every response with a **behavior**, a **verdict**, and a **failure mode** from a fixed taxonomy.
- **Research briefs** (the research-starting-point use case): `research_set.json`, `run_eval.py` and `graders.py`, with judging done in the Claude Code session. See [Research-brief eval](#research-brief-eval) below.

**Results and history.** `latest_results.ipynb` has the latest runs of both evals (for fact verification, the v7 ambiguity round: shorter answer lines, 45/50 + 1 partial vs v4's 48/50, with the follow-up fixes). `results.ipynb` has the last full fact-verification suite (v3, 234/250). `SUMMARY.md` explains how both evals were built, what they found and what changed in the system (§1-9 fact verification, §10 research briefs).

The sections from here to [Research-brief eval](#research-brief-eval) cover the fact-verification eval.

| Set | File | Items | What it measures |
|---|---|---|---|
| General (v3) | `data/general.jsonl` | 50 | Realistic hard cases: facts that changed in 2025-26, date-relative questions (the agent must know today's date), contested vs settled-consensus topics, lower-resource languages with local-script entity names, nicknames without redirects, subtle misconceptions, comparisons that flipped recently, calendar traps, messy phrasing |
| Ambiguity (v2) | `data/ambiguity.jsonl` | 25 pairs | Ask (or cover all readings) when a term is ambiguous, including hidden ambiguity, near-equal namesakes, weak context, time ambiguity, and terms whose disambiguation page isn't in the top-3 search results (`disambiguation_top3`); don't ask when a subtle detail settles it |
| Scope (v2) | `data/scope.jsonl` | 25 pairs | Answer at the right granularity or definition: city proper vs metro, competing official definitions, parts without their own article, lists users expect complete, then vs now |
| Aliases (v2) | `data/aliases.jsonl` | 25 pairs | Names that keyword search handles badly: no-redirect nicknames, misspellings, recent renames, local-language names, informal product names, time-correct historical names |
| False premise (v2) | `data/false_premise.jsonl` | 25 pairs | Subtle false premises (plausible wrong details, partially true, stale since 2025) (18 pairs) and not-on-Wikipedia questions where a nearby figure tempts fabrication (7 pairs) |

`eval_set.json` is the original 18-question smoke set, run by `smoke_eval.ipynb`. Earlier versions of every set are in `data/archive/` (v1 of each; v2 of general). Time-sensitive items carry `as_of` and must be re-verified before each run (`verify.py`).

## Pairs

Every failure-mode set is 25 **trap / control** pairs with matching surface form. A trap tests whether the system catches the problem; its control tests whether it over-triggers (asks, hedges, or "corrects" when it shouldn't). The report gives trap pass rate, control pass rate, and pair accuracy (both right). For aliases, the control asks the same question with the canonical name, so the trap-control gap isolates the cost of the alias.

## Pass rules (expected behaviors)

| `expected_behavior` | Pass | Fail |
|---|---|---|
| `answer` | Direct, correct answer | Asking, correcting a true premise, "not covered" |
| `clarify` | Asks which meaning, **or** answers all readings when there are only 2-3 and each is short | Silently picks one reading; a wall of text over many readings |
| `answer_with_scope_note` | Answers and names the scope used (plus the alternative when figures differ a lot) | Unlabeled figure at one scope |
| `answer_with_perspectives` | Gives the main positions on a genuinely contested question and their standing | States one side as settled fact |
| `correct_premise` | Flags the false premise, ideally with the nearest true facts | Answers as if true. "Not covered by Wikipedia" is only **partial** |
| `state_not_covered` | Says Wikipedia doesn't answer it | Calls the premise false; invents a specific answer |

**Footnote rule (strict, decided 2026-09-26).** Critical information must not live only in the `assumptions` footnote: a premise correction, a caveat the rubric requires, or the scope/metric an answer depends on must appear in the headline answer or the reasoning. If it appears only under 'Assumptions & gaps', the item is at most partial (the headline went along with the premise / omitted the caveat).

Items can add a `rubric` that tightens or overrides these (tolerances, which readings must be covered).

## Failure-mode taxonomy

Defined in `taxonomy.py`; the judge must choose from it. One primary mode per non-passing item (`none` on a pass), plus any secondary modes.

| Group | Modes |
|---|---|
| ambiguity | `missed_ambiguity`, `unnecessary_clarification`, `overlong_enumeration`, `wrong_referent` |
| scope | `scope_too_broad`, `scope_too_narrow`, `scope_unstated` |
| naming | `alias_not_resolved`, `anachronistic_answer` |
| premise | `accepted_false_premise`, `spurious_premise_correction`, `false_premise_called_not_covered`, `not_covered_called_false`, `fabricated_answer`, `unwarranted_refusal` |
| contested | `one_sided_contested`, `false_balance` |
| freshness | `stale_answer`, `date_unaware` |
| content | `incorrect_fact`, `multi_hop_break`, `reasoning_error`, `incomplete_answer`, `retrieval_miss` |
| grounding | `unsupported_citation`, `unsourced_claim`, `irrelevant_citation` |
| operational | `budget_exhausted`, `tool_error`, `format_violation`, `language_mismatch`, `other` |

The judge also records `grounding` (do the cited sections support the claims; it fetches every cited section's text), `citation_relevance` (are the cited sources about the question and the statements they are attached to, as opposed to padding, a same-name entity, or an unrelated section; `all_relevant`, `some_irrelevant`, `mostly_irrelevant`, `no_citations`, or `not_applicable`; judgments made before this field existed leave it blank), `followup_verdict` (the answer after a scripted clarification reply), and `gold_disagreement` (the answer contradicts the gold but Wikipedia supports it; these items are listed for review, not counted against the system).

## Item schema

```json
{"id": "scp-05t", "pair_id": "scp-05", "role": "trap", "subtype": "geographic_granularity",
 "question": "...", "expected_behavior": "answer_with_scope_note",
 "gold_answer": "...", "rubric": "...", "clarification_reply": "...",
 "gold_sources": [{"title": "Denmark", "contains": "42,9"}],
 "verified": {"sources": [{"resolved_title": "Denmark", "revid": 1234, "location": "lead"}],
              "answer_location": "lead", "popularity": "head", "checked": "2026-09-25"}}
```

General items carry `category` and `language` instead of pair fields, and time-sensitive ones carry `as_of` (the judge treats it as today). The earlier 100-item general set (domains x hops x question types) is archived in `data/archive/`. Alias items carry `alias` and `canonical_title`. False-premise items carry `subtle`.

`verified` is written by `verify.py`: it resolves each gold source on live Wikipedia, pins the revision id, and records where the evidence string appears (`lead`, `body`, or `markup` = only in infobox/table wikitext, which `read_article` never shows the agent). It also buckets popularity by last month's page views (head ≥ 100k, tail < 10k) and, for alias traps, whether the alias redirects and where search ranks the canonical article.

## Running

```bash
# 1. (after editing data) re-check gold sources; --write stores the `verified` blocks
.venv/bin/python -m evals.verify --write

# 2. run the system (resumable); a smoke test first is cheap
.venv/bin/python -m evals.run --name smoke --limit 2
.venv/bin/python -m evals.run --name baseline

# 3. grade and report
.venv/bin/python -m evals.judge --name baseline
.venv/bin/python -m evals.report --name baseline     # also writes evals/runs/baseline/report.md
```

To grade by hand instead of with the API judge (as for every full-suite run so far), use `evals/manual_grading/` in step 3: see its README.

Wikimedia rate-limits anonymous clients aggressively. Set `WIKI_USER_AGENT` to something with contact info (e.g. `wiki-search-evals/0.1 (you@example.com)`) and, if you still see HTTP 429s, `WIKI_MIN_REQUEST_INTERVAL=1.0`.

## Caveats

- 25 pairs gives roughly ±15-18 points of 95% CI around mid-range pass rates. That's enough to find failure modes but not to separate two close variants; add pairs or repeat runs for A/B comparisons.
- Gold answers are pinned to the revision ids in `verified`. Figures that drift (populations) carry tolerant rubrics; re-run `verify.py` periodically and review `gold_disagreement` items.
- `verify.py` only proves the evidence string appears in the article. Every gold was additionally read against the article text by hand; the outcome is in each item's `gold_review` (`verified`, or `fixed` with a note on what was wrong). The 7 "not on Wikipedia" traps were confirmed absent from the subject articles and top search results.
- The judge defaults to `claude-opus-5` with server-side refusal fallbacks enabled; `judge_model` in each judgment records which model actually graded.

## Research-brief eval

This eval covers `submit_report` briefs (overviews, histories, comparisons meant as a starting point) and is kept separate from the fact-verification eval above: different items, runner and graders. The design and results are in `SUMMARY.md` §10.

### Items

`research_set.json` holds 27 requests, grouped by the failure mode they probe:

| Category | Items | Probes |
|---|---|---|
| `research` | 2 | General briefs (history of LLMs, common distributions) |
| `goal` | 6 | Requests with a purpose (a trip, a report, a purchase, a debate, a new dog): the brief should help the user go further toward it. Several topics have namesakes that must stay out (Kyoto Protocol, Monty Python, Greyhound Lines) |
| `reader-level` | 8 | Four topics asked twice, with beginner and with expert hints (mRNA vaccines, Bell tests, inflation, CRISPR): depth and terminology should adjust |
| `controversy` | 6 | Real disputes (saturated fat, intermittent fasting, minimum wage, rent control, the atomic bombings, Columbus's legacy): present the positions and where the evidence stands, without declaring a winner where there isn't one |
| `limits` | 5 | Requests Wikipedia can't fully answer (an image never sent, a taste question, an undefined "best", a forecast, a life decision). These expect a short answer, not a brief |

There are no expected answers, since many different briefs can be good; briefs are judged against rubrics instead. Each item has:

- `expected_mode`: `brief`, or `answer` where a short answer is right;
- `must_cover`: topics a good brief must include;
- `time_sensitive` (8 items): the brief must give the latest information, dated, and cite recently updated pages;
- where relevant, `goal`, `reader_level` (`beginner` / `expert`; empty means a curious beginner-to-intermediate reader), `pair` (the other-level twin), `must_not_include` (off-topic namesakes) and `expected_limitations` (what an honest brief must say it can't answer);
- `review.status`: 2 items are `approved`; the other 25 are `draft` until a person reviews them.

```json
{"id": "research-01", "category": "research", "question": "Tell me about the history of LLMs",
 "expected_mode": "brief", "reader_level": null, "pair": null, "goal": null, "time_sensitive": true,
 "must_cover": ["the transformer (2017)", "BERT and the GPT series", "..."],
 "must_not_include": ["Master of Laws (the LLM degree)"], "expected_limitations": [],
 "review": {"status": "approved", "note": "..."}}
```

### Graders

Defined in `graders.py`. A brief gets up to six scores, each from 0 to 1:

| Grader | How | What it checks |
|---|---|---|
| `brief_format` | code | Right mode for the item; prose not above 625 words (25% over the recommended ~500, the point at which the agent sends a brief back); every overview and section point cited; "How these were chosen" filled in, and cited or labelled as the agent's judgment; 3–6 further-reading entries with notes of 20 words or fewer; no citation warnings. Score: share of checks passed |
| `freshness` | code | Time-sensitive items only: share of cited pages edited within a year of the run, on the revision the agent read. A recent edit is a proxy, so `completeness` also checks the brief gives the latest information, dated |
| `citations` | judged, per claim | Each claim against the full text of the sections it cites: support (full / partial / none), faithfulness (no distortion, overstatement, false precision or invented ranking) and relevance (is this the best section the agent read for the fact) |
| `completeness` | judged | The requested topics and every `must_cover` item; for a `goal`, whether the brief helps the user go further (concrete next steps, not a generic reading list); limits and gaps stated honestly while still offering what can be answered. Goal and limitation parts are also reported separately |
| `relevance` | judged, per point | Each point, including further reading: essential, supporting, marginal or off-topic. A name collision (the Kyoto Protocol in a history of Kyoto) is off-topic; `must_not_include` is passed to the judge |
| `style` | judged | Conciseness, clarity, table fit (a table only where seeing items side by side helps) and level fit (to `reader_level`, else the user's wording, else a curious beginner-to-intermediate reader), each 1–5, rescaled to 0–1 |

The judge sees exactly what the agent saw: cited sections come from the same Wikipedia cache, truncated to the same length. Every judged verdict must name the problem (e.g. "the cited section says height is log-normal, not normal"), so a score can be traced to specific text. `correctness` and `format` in `graders.py` are fact-check graders and aren't run on briefs.

### Judging

By default, judging happens **in the Claude Code session**, not through the API. `run_eval.py` writes each judgment as a Markdown grading packet (rubric, inputs, JSON schema for the verdict) at `runs/<run>/grading/<id>/<grader>.md`; Claude writes its verdict next to it as `<grader>.json`; `--collect` validates every verdict against its schema, computes the scores and writes `results.jsonl` and `summary.md`. Grades record which model judged them.

For unattended runs, `--judge api` fills the same packets through the Anthropic API (`--judge-model`, default `claude-opus-5`). Safety classifiers occasionally decline benign grading requests, so a declined request is retried on another model and then with a short framing note; if every attempt is declined, the grade is recorded as an error, never as a zero.

### Running

```bash
.venv/bin/python evals/run_eval.py                                  # run every request, write grading packets
.venv/bin/python evals/run_eval.py --ids research-02 --graders brief_format style
.venv/bin/python evals/run_eval.py --category goal                  # one category
# in Claude Code: "grade the pending packets in evals/runs/<run>"
.venv/bin/python evals/run_eval.py --collect evals/runs/<run>       # score, write summary.md
.venv/bin/python evals/run_eval.py --regrade evals/runs/<run>       # fresh packets for saved briefs (after a rubric change)
```

A run lives in `runs/<timestamp>_<model>/`: `answers.jsonl` (each brief with its structured fields, sections read, usage and cost), `traces/` (searches, reads and thinking), `grading/` (packets and verdicts), and after collecting, `results.jsonl` and `summary.md`.

### Caveats

- 27 items, one run each, graded by one judge (Claude, in session); items and system fixes were developed on the same set. Enough to find failure modes, not to compare close variants.
- 25 of the 27 items are drafts pending review, especially the controversy and limits `expected_limitations`.
- The judged grades for the latest goal and reader-level runs are still ungraded packets in their run folders, and the controversy and limits items haven't been run on the current system (`SUMMARY.md` §10.3-10.4).
