# Evals

Five eval sets for `WikiQA`, one runner, and an LLM judge that labels every response with a **behavior**, a **verdict**, and a **failure mode** from a fixed taxonomy.

| Set | File | Items | What it measures |
|---|---|---|---|
| General (v3) | `data/general.jsonl` | 50 | Realistic hard cases: facts that changed in 2025-26, date-relative questions (the agent must know today's date), contested vs settled-consensus topics, lower-resource languages with local-script entity names, nicknames without redirects, subtle misconceptions, comparisons that flipped recently, calendar traps, messy phrasing |
| Ambiguity (v2) | `data/ambiguity.jsonl` | 25 pairs | Ask (or cover all readings) when a term is ambiguous, including hidden ambiguity, near-equal namesakes, weak context, time ambiguity, and terms whose disambiguation page isn't in the top-3 search results (`disambiguation_top3`); don't ask when a subtle detail settles it |
| Scope (v2) | `data/scope.jsonl` | 25 pairs | Answer at the right granularity or definition: city proper vs metro, competing official definitions, parts without their own article, lists users expect complete, then vs now |
| Aliases (v2) | `data/aliases.jsonl` | 25 pairs | Names that keyword search handles badly: no-redirect nicknames, misspellings, recent renames, local-language names, informal product names, time-correct historical names |
| False premise (v2) | `data/false_premise.jsonl` | 25 pairs | Subtle false premises (plausible wrong details, partially true, stale since 2025) (18 pairs) and not-on-Wikipedia questions where a nearby figure tempts fabrication (7 pairs) |

`eval_set.json` is the original 18-question smoke set used by `demo.ipynb`. Earlier versions of every set are in `data/archive/` (v1 of each; v2 of general). Time-sensitive items carry `as_of` and must be re-verified before each run (`verify.py`).

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

Wikimedia rate-limits anonymous clients aggressively. Set `WIKI_USER_AGENT` to something with contact info (e.g. `wiki-search-evals/0.1 (you@example.com)`) and, if you still see HTTP 429s, `WIKI_MIN_REQUEST_INTERVAL=1.0`.

## Caveats

- 25 pairs gives roughly ±15-18 points of 95% CI around mid-range pass rates. That's enough to find failure modes but not to separate two close variants; add pairs or repeat runs for A/B comparisons.
- Gold answers are pinned to the revision ids in `verified`. Figures that drift (populations) carry tolerant rubrics; re-run `verify.py` periodically and review `gold_disagreement` items.
- `verify.py` only proves the evidence string appears in the article. Every gold was additionally read against the article text by hand; the outcome is in each item's `gold_review` (`verified`, or `fixed` with a note on what was wrong). The 7 "not on Wikipedia" traps were confirmed absent from the subject articles and top search results.
- The judge defaults to `claude-opus-5` with server-side refusal fallbacks enabled; `judge_model` in each judgment records which model actually graded.
