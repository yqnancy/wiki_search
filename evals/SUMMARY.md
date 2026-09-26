# WikiQA eval effort: summary

This document summarizes how we built, checked and used an eval suite for `wiki_search` (WikiQA), a question-answering agent that answers only from English Wikipedia. It covers the design of the eval sets, the failure modes they target, what they caught in the system, the fixes made in response, and what is still open.

## 1. Goal and approach

**The tool.** `wiki_search` is a Claude tool-use loop (default backbone `claude-sonnet-4-6`). It uses these tools:

| Tool | What it does |
|---|---|
| `search_wikipedia` | Keyword search; flags disambiguation pages. |
| `read_article` | Reads an article's lead plus table of contents, or one section. |
| `submit_answer` | Submits a short structured answer: answer, verdict, reasoning with citations, assumptions, sources. |
| `submit_report` | Submits a longer brief with sections and a table. |

A separate model call checks disambiguation pages and may stop to ask the user a clarifying question. Answers are rendered to Markdown by `wiki_search/render.py`.

**Eval philosophy.**

- **Paired traps and controls.** Every failure-mode set is 25 trap/control pairs.
  - The *trap* contains the problem: an ambiguous term, an underspecified scope, an alias, or a false premise.
  - The *control* has the same surface form without the problem.
  - Together they give three numbers: *trap pass* (catches the problem), *control pass* (doesn't over-trigger: ask, hedge or "correct" when it shouldn't), and *pair accuracy* (both right).
- **Realistic difficulty.** Questions real users would type, not contrived puzzles or long hop chains.
- **Gold answers from live Wikipedia.** Every gold answer was checked against the current text of the cited article. Items record revision ids, and time-sensitive items carry an `as_of` date that the grader treats as "today".
- **Strict pass rules**, including the footnote rule in §2.7.

**Grading.** There are two interchangeable paths. Both produce `judgments.jsonl` records with the same fields, labelled from the fixed taxonomy in `evals/taxonomy.py`.

- **LLM judge (`evals/judge.py`).** One `claude-opus-5` call per response, with adaptive thinking, high effort, structured JSON output and server-side refusal fallback. The judge sees:
  - the item (question, expected behavior, gold answer, rubric, `as_of`, language);
  - the rendered answer, including `question_type` and any citation warnings;
  - a short tool trace;
  - the text of every cited Wikipedia section. Infobox citations are fetched with `wikipedia.get_infobox`.

  Fixed rules in code then override the judge where the record is unambiguous: an error, a clarification turn, the turn limit, and consistency between pass and failure mode.
- **Manual grading.** Used for the full-suite runs so far. A Claude Code session read every response and assigned the same labels under the same rules. Manual grades don't check each citation against the source section; they record only visible citation defects.

Judgment fields:

| Field | Meaning |
|---|---|
| `behavior` | What the system did (see §2.6). |
| `verdict` | `pass`, `partial` or `fail`. |
| `primary_failure_mode` | One taxonomy code; `none` on a pass. |
| `secondary_failure_modes` | Any further codes. |
| `followup_verdict` | For clarification turns: was the answer after the scripted clarification correct? |
| `grounding` | Do the cited sections support the load-bearing claims? |
| `citation_relevance` | Is each citation on-topic for its statement? |
| `gold_disagreement` | Wikipedia supports the system over the gold answer; the item is flagged for review. |
| `rationale` | Short explanation. |

## 2. Final design of the eval sets

There are 250 items across 5 files in `evals/data/`, all verified against Wikipedia on 2026-09-25 (§4).

### 2.1 General (v3): `general.jsonl`, 50 unpaired items

**Categories.**

| Category | Items | What it tests |
|---|---|---|
| recent_change | 12 | Facts that changed in 2025–26, e.g. heads of government, CEOs, records, champions, moon counts. Memory gives a stale answer. |
| date_relative | 8 | Ages, elapsed time, "is X still alive / in office". Requires knowing today's date. |
| non_english | 6 | Vietnamese, Swahili, Tagalog, Bengali, Turkish, Indonesian. The key entity is named only in the local language or script. |
| contested | 5 | Genuine disputes that users may think are settled. |
| nickname_no_redirect | 4 | The nickname lands on a misleading page. |
| misconception | 4 | A popular belief baked into the question. |
| colloquial | 3 | Vague descriptions, typos. |
| stale_comparison | 3 | A comparison whose answer flipped because of a 2025–26 result. |
| calendar_units | 3 | Julian vs Gregorian, the Ethiopian calendar. |
| consensus | 2 | Settled science where a debate is invited. |

**Expected behaviors:** `answer` 40, `correct_premise` 5, `answer_with_perspectives` 5. 24 items carry `as_of: 2026-09-25`.

**Target failure modes:** `stale_answer`, `date_unaware`, `one_sided_contested`, `false_balance`, `accepted_false_premise`, `alias_not_resolved`, `language_mismatch`, `incorrect_fact`.

**Examples:**
- "Who is the prime minister of Hungary?" → Péter Magyar, since 9 May 2026 (a stale answer would be Orbán).
- "How long has Ali Khamenei been Iran's supreme leader?" → he no longer is (assassinated 28 February 2026). A hidden false premise inside an elapsed-time question.
- "In what month of 1917 did the October Revolution break out?" → November, by the Gregorian calendar.

### 2.2 Ambiguity (v2): `ambiguity.jsonl`, 25 pairs

**Trap subtypes:** hidden_ambiguity 4, weak_context 4, near_equal_people 3, time_ambiguity 3, domain_acronym 2, work_title 2, non_english 2, plus homonym, sports_team, person, acronym and place (1 each).

For each trap, `disambiguation_top3` records whether the disambiguation page ranks in the top 3 search results. It is true for 19 and false for 6. When it's false, the system's automatic disambiguation check never fires.

**Pass rules.**
- Trap (`clarify`): passes if the system asks, or if it covers every reading when there are only 2–3 short ones. Silently picking one reading is `missed_ambiguity`; a wall of text over many readings is `overlong_enumeration`.
- Control (`answer`): asking fails it (`unnecessary_clarification`).
- Each trap carries a scripted `clarification_reply`, so the answer after clarification is graded too.

**Examples:**
- "How many people does the stadium where Inter play hold?" Inter Milan's San Siro (75,817) vs Inter Miami's Nu Stadium (26,700).
- "How many Ballon d'Or awards did Ronaldo win?" Cristiano (5) vs Ronaldo Nazário (2). The "Ronaldo" page is not flagged as a disambiguation page.
- Panthers, with the only clue being the time of year: "…trip down south in November" (ambiguous) vs "…in late March" (settled, because the NFL is out of season).

### 2.3 Scope (v2): `scope.jsonl`, 25 pairs

**Trap subtypes:** geographic_granularity 7, temporal 6, competing_definitions 5, too_narrow 3, category_granularity 2, part_vs_whole 2.

**Pass rules.**
- `answer`: the implied scope is clear.
- `answer_with_scope_note` (16 traps): the scope is underspecified. The answer must name the scope it used and, when figures differ a lot, give the alternative.

**Target failure modes:** `scope_too_broad`, `scope_too_narrow`, `scope_unstated`.

**Examples:**
- "What is the most populous city in the world?" Greater Jakarta (UN 2025, 41.9M) by urban area; Tokyo is now third.
- "What is the population of Delhi?" City proper vs NCT vs urban agglomeration.
- "Who designed the pedestal of the Statue of Liberty?" Richard Morris Hunt. The article lead names Bartholdi and Eiffel, which pulls toward the wrong answer.

### 2.4 Aliases (v2): `aliases.jsonl`, 25 pairs

**Trap subtypes:** nickname 6, time_correct_name 5, historical_name 3, misspelling 3, recent_rename 2, local_language 2, informal_product 2, renamed_org 1, regional_common_name 1.

The trap uses the alias; the control asks the same question with the canonical name. Both expect `answer`. **No trap alias redirects to its article:** 10 land on a different page (`other:`), 9 on a disambiguation page, and 6 on nothing.

**Target failure modes:** `alias_not_resolved`, `wrong_referent`, `anachronistic_answer`, `stale_answer`.

**Examples:**
- "Which movie got Timothy Shalamay his first Oscar nomination?" Search returns 0 results.
- "Where do the Oakland A's play their home games now?" Sutter Health Park.
- "Who designed the Boiled Egg opera house in Beijing?" The alias lands on the food article.

### 2.5 False premise (v2): `false_premise.jsonl`, 25 pairs

**Pairs 01–18 are false premises.**
- Trap (`correct_premise`) subtypes: wrong_detail 4, stale_premise 4, partially_true 3, invented_entity 3, false_relation 2, false_attribution 1, misconception 1.
- Control: the same surface form with a true premise (`answer`).

**Pairs 19–25 are not-on-Wikipedia questions.**
- Trap (`state_not_covered`): the premise is true but Wikipedia lacks the fact, and a nearby figure tempts fabrication.
- Control: a matching question Wikipedia does answer.

23 of 25 traps are marked `subtle`.

**Target failure modes:** `accepted_false_premise`, `spurious_premise_correction`, `false_premise_called_not_covered`, `not_covered_called_false`, `fabricated_answer`, `stale_answer`.

**Examples:**
- "Why did the Space Shuttle Columbia break apart shortly after launch?" It broke up on re-entry in 2003.
- "How old is Jane Goodall now?" She died on 1 October 2025.
- "How much was Michelangelo paid for painting The Last Judgment?" Not on Wikipedia; the Sistine ceiling fee (3,000 ducats) is a nearby distractor.

### 2.6 Taxonomy

**Behaviors:** `answered`, `answered_all_readings`, `asked_clarification`, `corrected_premise`, `stated_not_covered`, `declined`.

**Expected behaviors:** `answer`, `clarify`, `answer_with_scope_note`, `correct_premise`, `state_not_covered`, `answer_with_perspectives`.

**Failure modes:** 33 codes including `none`, in 10 groups. "Later" marks the codes added during this work.

| Group | Codes |
|---|---|
| ambiguity | missed_ambiguity, unnecessary_clarification, overlong_enumeration, wrong_referent |
| scope | scope_too_broad, scope_too_narrow, scope_unstated |
| naming | alias_not_resolved, anachronistic_answer |
| premise | accepted_false_premise, spurious_premise_correction, false_premise_called_not_covered, not_covered_called_false, fabricated_answer, unwarranted_refusal |
| contested (later) | one_sided_contested, false_balance |
| freshness (later) | stale_answer, date_unaware |
| content | incorrect_fact, multi_hop_break, reasoning_error, incomplete_answer, retrieval_miss |
| grounding | unsupported_citation, unsourced_claim, irrelevant_citation (later) |
| operational | budget_exhausted, tool_error, format_violation, language_mismatch (later) |
| other | other |

`language_mismatch` and the grounding codes normally appear as secondary modes and don't fail an item by themselves.

### 2.7 The strict footnote rule (decided 2026-09-26)

Critical information must not appear only in the "Assumptions & gaps" footnote. This covers a premise correction, a caveat the rubric requires, and the scope or metric an answer depends on. Each must appear in the headline answer or the reasoning; otherwise the item is at most `partial`.

Example: "Which of the 50 original US states was the last to ratify the Constitution?" The headline said "Rhode Island" and only the footnote noted there were 13 states. That is now `partial` / `accepted_false_premise`.

The rule is written into `evals/README.md`, the judge prompt, and a comment in `evals/taxonomy.py`.

## 3. How the sets evolved

| Version | Sets | Why it changed |
|---|---|---|
| v1 | general (100 items: 10 domains × 1–4 hops × 7 question types) + four pair sets | Initial design. |
| v1 fixes | the same sets | Verification against Wikipedia found drafting errors (Mount Scenery is 870 m, not 887 m; renamed titles). 20 of the first 25 alias traps had plain redirects, so 6 were replaced with no-redirect aliases. |
| v1 contamination fixes | 12 items rewritten | The prompts had changed and reused some eval entities: the disambiguation prompt's "population of Georgia", the "1994 FIFA World Cup" and "Great Fire of London" examples. Four items also reused dev-set facts (Armstrong, Curie, the Mona Lisa, the Best Picture template). |
| general v2 | 50 items | The v1 general set was near ceiling (about 98–99% on partial runs), 86 of its answers sat in the article lead, and 4-hop chains felt contrived. v2 targets realistic difficulty (recent changes, contested topics, non-English, no-redirect nicknames, date-relative) and drops infobox-only facts, since no user would expect those to be the test. |
| all sets v2 / general v3 | ambiguity, scope, aliases, false_premise v2; general v3 | Baseline run: scope 100%, aliases, ambiguity and false premise 98% / 98% / 92% (lenient grading). The paired sets were saturated. Each was rebuilt by a separate agent with harder realistic cases, and general v3 shifted weight toward what discriminated (freshness, dates) and away from categories that passed at 100%. |

**Archives in `evals/data/archive/`:**
- `general_v1.jsonl` and `reserve_general_v1.jsonl` (38 v1 items cut when trimming to 100);
- `general_v2.jsonl`;
- `ambiguity_v1.jsonl`, `scope_v1.jsonl`, `aliases_v1.jsonl`, `false_premise_v1.jsonl`.

The four `_v1` pair files hold the versions used in the baseline run, including the v1 fixes above.

## 4. Dataset quality process

- **`evals/verify.py`.** For each `gold_sources` entry (`{title, contains}`) it:
  - resolves the title on live Wikipedia and records the revision id;
  - confirms the evidence string appears in the plain-text lead or body. Markup-only matches are recorded as `markup`.

  Per item it also records:
  - `answer_location`;
  - popularity from last month's page views (head ≥ 100k, tail < 10k);
  - for alias traps, the redirect status and where search ranks the canonical article.

  It uses a disk cache (`evals/.cache/`) and a paced, identified user agent. The current suite verifies with 0 problems: every evidence string is in the lead or body, none only in markup.
- **Hand check of every gold answer.** `verify.py` only proves a string exists, so each gold answer was read against the surrounding article text. Of the 300 v1 items, 23 were fixed, for example:
  - *Kind of Blue*'s producer is Irving Townsend; Teo Macero is often misattributed.
  - Dennis Ritchie never officially received his PhD.
  - KFC's headquarters is Plano, Texas, not Louisville.
  - Greater Tokyo is 33M in the Tokyo article and 36.95M in the Greater Tokyo Area article.
  - Portland, Maine's "1832" was when Maine's capital moved, not a city incorporation.

  Later, the system's own answer showed the X/Twitter ownership gold was stale: the article lead now names SpaceXAI.

  Each item's `gold_review` records `verified` or `fixed` with a note. Every v2/v3 item was built from article text read at creation time; 244 are `verified` and 6 are `fixed`. For not-on-Wikipedia traps, the note lists the articles and search results checked for absence.
- **Contamination check (`evals/contamination.py`).** It compares each item (question + gold) with the system prompt, the disambiguation prompt, the tool descriptions and every answer field of the dev set `evals/eval_set.json`. It uses n-gram overlap, shared entities (serious only when the entity is a prompt's quoted example), and shared answer facts with dev items.

  The current suite has **0 high flags**, 8 medium (generic stems such as "what is the population of") and 10 low.
- **`as_of` and re-verification.** 50 items are time-sensitive and carry `as_of: 2026-09-25`. Runs pin the agent's "today" to that date (`evals.run --today`). Rerun `verify.py` before each run; recent facts such as deals, CEOs, records and ongoing seasons can change within days.

## 5. Issues the evals caught (baseline run `evals/runs/suite_v2_sonnet46`)

**The baseline run.** One run of the original system (Sonnet 4.6, effort medium), graded manually, over the v2-era sets: general v2 plus the v1 pair sets, 250 items. It cost about $10 for the system.

**Results under the strict footnote rule: 235/250 (94.0%).**

| Set | Pass | Trap pass | Control pass | Pair accuracy |
|---|---|---|---|---|
| General (v2) | 43/50 | – | – | – |
| Ambiguity (v1) | 49/50 | 24/25 | 25/25 | 24/25 |
| Scope (v1) | 49/50 | 24/25 | 25/25 | 24/25 |
| Aliases (v1) | 49/50 | 24/25 | 25/25 | 24/25 |
| False premise (v1) | 45/50 | 21/25 | 24/25 | 21/25 |

**Non-passing items (15):**

| Cause | Count | Failure mode |
|---|---|---|
| `assumptions` submitted as a string instead of a list; `render()` iterated it, printing one bullet per character. The content was correct in every case. | 7 | format_violation |
| The system didn't know today's date ("35 as of 2025", "39 years ago") | 2 | date_unaware |
| Hedged on X/Twitter's current owner instead of leading with the article lead's statement | 1 | stale_answer |
| Invented war answered "Not covered by Wikipedia" although the body said it never happened | 1 | false_premise_called_not_covered |
| Wrong background claim inside `assumptions` (Denali's naming history) | 1 | incorrect_fact |
| Strict footnote rule: correction or caveat only in the footnote (50 states, five senses, most-spoken-language metric) | 3 | accepted_false_premise, incomplete_answer, scope_unstated |

**Secondary issues on passing items:**
- 3 statements with empty citation lists;
- 1 citation index pointing past the sources list;
- 2 answers with English reasoning under a non-English question;
- 1 weak inference;
- 1 infobox-only fact (a film runtime) that the tool couldn't read, correctly labelled as unconfirmed.

**Grading-rule change.** 3 items (fp-17t, gen-237, scp-11t) moved from pass to partial under the strict footnote rule, taking the baseline from 238 to 235.

**Also caught during grading and gold review.** One run surfaced a second answer format: `submit_report`, used for list questions such as "who has walked on the Moon".

## 6. Improvements made

### 6.1 The system (`wiki_search/`)

| Fix | Change |
|---|---|
| Strict submit schema | `submit_answer` is `strict: true`, with `additionalProperties: false` and all fields required; optional fields are empty or null. `submit_report` has a strict-ready schema but **can't be strict too**: with both strict, the API rejects the request ("compiled grammar is too large"). |
| Defensive rendering | `_items()` / `_text()` helpers turn a bare string into a one-item list and null into an empty list, in both renderers. This makes the string-for-list bug impossible in either format. |
| Date injection | A new `WikiQA(today=...)` parameter, defaulting to today. The date goes in a second system text block after the cached `SYSTEM_PROMPT`, so prompt caching still works. The prompt tells the agent to compute ages and elapsed times from it. |
| Prompt rules | New rules for current facts (answer from the lead's present-tense statement), false premises (the correction goes in `answer`; `not-covered` only for real facts Wikipedia lacks), `assumptions` (no new uncited facts, never the only place a correction appears), and language (all answer fields in the question's language). |
| Citation retry | Warnings for a citation index with no matching source, an unretrieved source or an empty citation list go back to the model once as a tool error. The second submission is accepted, with warnings kept on `Answer.warnings` and a `recheck` trace entry. |
| Infobox pseudo-section | `wikipedia.parse_infobox()` / `get_infobox()` read the infobox from section 0 wikitext. `read_article(title, section="Infobox")` returns key: value lines, listed first in the table of contents. Fields filled from Wikidata stay invisible (e.g. Everest's elevation). |

Smoke tests (5 live questions):
- Titanic's runtime came back as "195 minutes", cited to the infobox.
- A Spanish "how old" question got "36 años", computed from the pinned date.
- An invented treaty was corrected in the headline.
- The Twitter-owner answer now names SpaceXAI but still leads with Musk.
- The retry path was tested only with a fake client.

### 6.2 Eval tooling (`evals/`)

- **`run.py`:**
  - `--resume`: drops error records so they're retried.
  - An API error *inside* a tool call (the agent turns tool exceptions into tool results) is recorded as a run error.
  - `max_retries=10` for concurrency 429s.
  - `--repeats N`: all repeats in one process share an unbounded cache of Wikipedia searches and articles, so run-to-run variance reflects the model, not changing search results.
  - `--today` pins the agent's date.
- **`judge.py`:**
  - taxonomy-constrained structured output;
  - `as_of`, category and language passed through;
  - citation warnings and `question_type` shown to the judge;
  - `citation_relevance` and `irrelevant_citation` added;
  - the footnote rule in the prompt;
  - `section_text()` fetches infobox text for "§ Infobox" citations;
  - judge token usage recorded; `--tag` for re-grading passes.
- **`report.py`, `analysis.py`:** pass rates with Wilson CIs, pair metrics, slices by subtype and category, behavior matrices, failure-mode tables, cost/latency, run-to-run stability and paired bootstrap helpers. Both handle single-run and repeated-run folders, and can load archived item files for older runs.
- **Notebook generators:** `build_notebook.py` → `analysis.ipynb` (dataset, contamination, variance, model comparison); `build_results_notebook.py` → `results.ipynb` (latest results up front, earlier runs in appendices).

## 7. Infrastructure lessons

- **Wikimedia rate limits.** Anonymous clients were throttled hard (HTTP 429 with 40–60 s Retry-After). The fixes:
  - an identified user agent (`WIKI_USER_AGENT` in `.env`, loaded by `evals/__init__.py` before `wiki_search` is imported);
  - steady pacing (`WIKI_MIN_REQUEST_INTERVAL`);
  - a disk cache for verification.
- **API credits ran out** in the middle of the first 5-repeat Sonnet/Haiku run. Error records had been saved as completed; `--resume` now retries them.
- **Concurrency 429s.** 24 workers across two processes plus the judge exceeded the org's concurrent-request limit. Mitigated with fewer workers and SDK retries.
- **API errors inside tool calls** (e.g. in the disambiguation sub-call) looked like ordinary tool errors to the agent and polluted responses. The runner now flags them.
- **Caching across repeats** keeps Wikipedia identical between runs, which is needed for a meaningful variance estimate.
- **Shared scratchpads.** Parallel subagents overwrote each other's helper scripts in the shared scratchpad. Give helpers distinctive names, or give each agent its own folder.

## 8. Latest results (improved system on the current eval sets)

Run `evals/runs/suite_v3_sonnet46/`: Sonnet 4.6 at medium effort, date pinned with `--today 2026-09-25`. System cost was $7.65, and all 250 items were graded manually under the strict footnote rule. Details are in `results_findings.md` and `results.ipynb`.

| Set | Pass | Trap / control | Non-passing items |
|---|---|---|---|
| General v3 | 49/50 (98%) | n/a | gen-340: Nyad's swim not flagged as never ratified |
| Ambiguity v2 | 42/50 (84%) | 72% / 96% | 7 `missed_ambiguity`, 1 partial, 1 `unnecessary_clarification` (amb-18c) |
| Scope v2 | 45/50 (90%) | 80% / 100% | 4 `scope_unstated`, 1 `incomplete_answer`; all partial |
| Aliases v2 | 48/50 (96%) | 96% / 96% | als-24t/c: Mac Pro duration measured to the wrong endpoint |
| False premise v2 | 50/50 (100%) | 100% / 100% | none |
| **All** | **234/250 (93.6%)** | | |

**What the fixes bought.** No v2 failure class recurs: `format_violation`, `date_unaware`, `stale_answer` and `false_premise_called_not_covered` are all at zero. Every general item in the date-relative (8), recent-change (12), non-English (6) and contested (5) categories passes. All 18 premise corrections are in the headline.

**Remaining gaps:**
1. **Missed ambiguity when context seems to settle it.** Trap recall is 72%. The agent sometimes adds its own qualifier to the first search ("Inter Milan stadium"), so the disambiguation check never fires. Elsewhere a domain prior picks one reading: PE → pulmonary embolism, Go → programming language, Masters → golf.
2. **Buried caveats.** Four of the 16 non-passes (amb-20t, amb-23t, scp-03t, scp-19t) already contain the needed alternative or definition, but only in Assumptions. A fifth, scp-20t, gestures at the scope there without naming Heuss.
3. **Unstated scope on competing definitions:** Elbrus vs Mont Blanc, Bologna vs al-Qarawiyyin, Ebert vs Heuss.

**Next system changes, in order:**
1. A validator plus a retry turn that promotes caveats out of Assumptions.
2. A disambiguation check on the user's original term, before any query rewrite.
3. An explicit "readings" step for weak-context questions.
4. A headline rule for superlatives, counts and "population of" questions: state the definition used and the leading alternative.

The baseline's 94.0% and this run's 93.6% are not comparable, because the eval sets were made harder in between.

### 8.1 Follow-up round: caveat promotion, readings step, and query faithfulness (2026-09-26)

**Two fixes from the gap list, implemented by a subagent:**
1. **Caveat promotion** (`render.buried_caveats` + a one-time corrective turn in `agent.py`).
   - A deterministic phrase detector flags Assumptions entries that carry an alternative reading, scope definition or premise correction. The model then gets one turn to move that content into the answer or reasoning.
   - Retries share a budget of two corrective turns per run with the citation retry and are logged as `promote` trace events.
   - Offline on the 250 v3 responses it fired on 8 items: all 5 buried-caveat targets, amb-09t, als-07t (a real buried premise correction) and scp-22c (harmless).
2. **Readings step** (system prompt, step 1).
   - Before the first search, the agent writes a note listing each ambiguous name's possible meanings and which words in the question rule each one out. Fame or a field prior does not rule a meaning out.
   - If 2–3 meanings survive, it covers each in the answer.

**Smoke run `evals/runs/suite_v4_fix13/`** (ambiguity + scope, 100 items; ambiguity graded 48/50, scope not graded):
- **Resolved on this run:** 8 of the 10 targets now surface the alternative or caveat. amb-01t is unstable across the subagent's three runs.
- **Still missed:** amb-09t (PE → pulmonary embolism) and amb-22t (Masters → golf), in every run. scp-20t also missed on this run.
- **Controls:** none asks for clarification. Some headlines got longer.
- **Cost:** +15% per item ($0.036 vs $0.031), +1 s wall time.

**Query-faithfulness metric** (`evals/query_faithfulness.py`, ambiguity set only). It asks whether the agent's searches keep the question's level of ambiguity.
- **Annotation.** Each pair records the ambiguous term and, for each meaning, the words that pick it: "Milan" or "San Siro" for Inter Milan, "pulmonary"/"embolism" for pulmonary embolism.
- **Narrowing search.** A search on the term *narrows* when it adds words, not in the question, that pick some meanings but not all. Rewording, dropping words, and adding words that fit every meaning ("stadium", "2026") are allowed. A search with the term and no such words is *open*.
- **committed (headline measure):** every search on the term narrows, and all of them pick the same single meaning. The agent settled the ambiguity through search alone.
- **first narrowed:** the first search on the term narrows, so the disambiguation check cannot fire on it.
- **Why two measures.** The first definition used only the first search. The v4 run showed the flaw: the readings step makes the agent search each meaning in turn ("New York Rangers …", "Texas Rangers …", "Rangers F.C. …"). That is correct behavior, but it counts as a narrowed first search.
- **Validation.** Automatic labels were checked by hand on both runs. On v3, committed traps passed 1/5 and the other traps 17/20, so the metric tracks the outcome.
- **Controls:** a high committed rate there is expected and correct, because the question names the meaning.

| Run | System | Traps committed | Traps with narrowed first search | Controls committed | Searches per trap | Ambiguity pass (graded) | Trap / control pass | Trap clarifications |
|---|---|---|---|---|---|---|---|---|
| suite_v3_sonnet46 | v3 system | 5/25 (amb-01t, 09t, 11t, 22t, 24t) | 5/25 | 20/25 | 1.1 | 42/50 | 72% / 96% | 12 |
| suite_v4_fix13 | + caveat promotion + readings step | 3/25 (amb-09t, 22t, 24t) | 7/25 | 19/25 | 1.4 | **48/50** | **92% / 100%** | 9 |
| suite_v5_rule2 | + rewritten rule 2 | 3/25 (amb-09t, 20t, 22t) | 10/25 | 19/25 | 1.6 | 45/50 | 80% / 100% | 3 |
| suite_v6_final | v4 prompt + term pre-check + question-only check | 3/25 (amb-01t, 20t, 22t)* | 3/25 | 18/25 | 0.6 | 42/50 | 84% / 84% | 15 |

\* In v6, 14 traps have no agent search at all: the pre-check asked the user before the agent's first turn, so only the pre-check's verbatim search ran. Committed v6 traps passed 2/3 and the rest 19/22.

- **v4, v5 and v6 grading.** Both runs were graded by hand under the strict footnote rule; judgments are in each run folder (ambiguity items only; v4's scope items are still ungraded).
- **Committed searches predict failure.** On v4, committed traps passed 1/3 and the rest 22/22. On v5 the figures are 0/3 and 20/22.
- **Rule 2 rewrite: no effect on the metric, and the agent often didn't follow it.**
  - The committed count stayed at 3/25, and narrowed first searches rose from 7 to 10. The agent now searches each meaning directly without an open first search (Rangers, Ronaldo, Go, Panthers, The Office, Congo). That's harmless, but it breaks the rule as written.
  - On the harmful cases the rule was simply broken. amb-09t still expanded PE to pulmonary embolism, despite the rule's own MS example. amb-20t searched "North Macedonia largest city" first, having searched the bare name in v3.
  - The readings note was skipped entirely on amb-09t, 20t and 22t. On 20t and 22t the model wrote a prose answer before being nudged to submit.
- **v5 vs v4 outcome.** v5 scored 3 traps lower: amb-20t fail, amb-15t and amb-23t partial. With one run per system, that is within noise, so the rule may not have caused it; the subagent had already seen amb-01t flip between runs. Trap clarifications fell from 9 to 3, as the agent covered readings instead of asking.
- **The two persistent failures survive every prompt variant tried:** amb-09t (PE → pulmonary embolism) and amb-22t (Masters → golf). The agent's prior overrides both the readings note and the search rule, and the disambiguation sub-call agrees with the prior on the Masters.
- **What followed:** rule 2 was reverted and a structural check was added (v6, §8.2). The 3–5-repeat runs to measure noise were not done.

**Rule 2 rewrite (v5 only; reverted in v6).**
- **Old rule.** "Search the bare name first", limited to names *that could refer to more than one thing*. That condition depends on the agent noticing the ambiguity, which is exactly what fails: amb-01t, 09t, 11t and 22t broke the old rule.
- **New rule.** The first search for *every* name, acronym or short title may reword or shorten it, but must not add words that pick a meaning unless the question already says so. After that first search, the agent may narrow freely.
- **Examples** use Mercury and MS, which are not eval terms; the contamination check shows no high flags.
- **Expected effect (not borne out; see the table):** committed and first-narrowed rates on traps should fall, especially amb-09t and amb-22t, without extra searches or clarifications on controls.

**Caveats:**
- 25 traps means each item moves a rate by 4 points.
- The detector and the readings prompt were tuned on the same items they are scored on. Hold out pairs, or add new weak-context pairs, before trusting gains.
- The disambiguation sub-call can still override a correct readings note, as it did for Macedonia and the Masters. A separate candidate fix is to give it only the user's question, not the agent's query.

### 8.2 Final round (v6) and where the system stands (2026-09-26)

**Changes in the final system (`evals/runs/suite_v6_final/`, ambiguity set, graded):**
1. **Rule 2 reverted** to the v4 wording ("Check ambiguous names first").
2. **Term pre-check** (`agent._precheck`, `question_terms`; **reverted after this run**, see below).
   - Before the agent's first turn, up to three names or acronyms are taken verbatim from the question and searched: capitalised runs such as "Trinity College", "Book of Kells", "Ballon d'Or", and all-caps acronyms. Sentence-opening question words, months and weekdays are skipped.
   - Any disambiguation page in the top 3 goes to the disambiguation check. The results aren't shown to the agent. The check runs whether or not the agent notices the ambiguity.
3. **The disambiguation check now sees only the user's question,** not the agent's search query. The query can carry the agent's guess ("2026 Masters Tournament golf"). The prompt's "incidental term" rule was reworded to match.
4. **Crash fix:** a string `selection_basis` in `submit_report` crashed `agent._unsourced_selection` and `verify.check_claims` ("'str' object has no attribute 'get'", on amb-09c). Both now use the same type guard as `render.py`. The item was re-run.

**Result: 42/50, below v4's 48/50.** Traps passed 84% (v4: 92%) and controls 84% (v4: 100%).

- **What the pre-check fixed:** the two misses no prompt variant fixed. amb-09t (PE) and amb-22t (Masters) now ask the user, and the follow-up answers are correct. amb-20t (Macedonia) also passes, with Thessaloniki in the headline after the caveat retry.
- **What it broke:**
  - **Over-triggering on controls (3):**
    - amb-12c asks game vs language despite "early access".
    - amb-21c asks Missouri vs Kansas despite "Arrowhead Stadium".
    - amb-18c asks despite "late March". That control was already flagged as debatable.
    - In each case the check sees the whole question but still calls the term ambiguous. In v4 these same checks, or no checks at all, left the controls alone.
  - **Repeated clarification (2):** after the user answers, turn 2 runs the pre-check again, and the check can ask again. amb-18t got "The football team", then asked about other football Panthers. amb-18c got "The hockey team", then asked about the Nottingham Panthers. The user never gets an answer.
  - **Wrong referent (1):** amb-22c ("the Masters at Alexandra Palace") answered with the darts World Masters (Luke Littler) instead of the snooker Masters (Kyren Wilson). The check had named the golf tournament as the likely meaning.
- **Still missed:**
  - amb-01t (Inter) and amb-11t (Go): the pre-check ran, but the check judged each unambiguous on priors (Inter Milan; "learn it" means the language). v4 passed both, via the readings step.
  - amb-15t (Trinity College): Oxford only in Assumptions; no disambiguation page in the top 3.
- **Cost:** 4.8 model calls per item (v4: 3.8) and 24 s mean wall time (v4: 20 s).
- **Confound:** another session edited the shared system prompt during this work. Its changes include a Disputed verdict, the "Limits" and "Disputes" rules, research-brief rules and example swaps. v5 and v6 may include some of these; v3 and v4 do not. Most target claims and research briefs, not short ambiguity answers, but they are uncontrolled.

**Verdict on the ambiguity work.** The best measured configuration is **v4**: the caveat retry plus the readings step, with no pre-check. The pre-check code was therefore removed from `agent.py`. The shipped system is v4 plus:
- the question-only disambiguation check (change 3), which was never measured on its own;
- the `selection_basis` crash fix;
- the other session's prompt edits.
- The pre-check swaps two prior-driven misses for more control failures and a repeated-clarification bug.
- In short: the check that decides "is this ambiguous?" is now the bottleneck, not whether it gets to run. It says "not ambiguous" on Inter and Go, and "ambiguous" on Rust with "early access" and on Kansas City with "Arrowhead".
- Every figure here is one run of 50 items; v4 vs v6 differ by 6 items.

### 8.3 Remaining issues (improvements stopped here)

**Current system state:** v4 (caveat retry + readings step, original rule 2), plus the question-only disambiguation check, the `selection_basis` crash fix, and the other session's prompt edits. This exact combination has not been run. Re-run the full suite once before relying on the v3/v4 numbers for it.

**System:**
1. **The disambiguation check's calibration** is now the main limit on ambiguity handling. It resolves weak-context terms on priors (Inter, Go, Macedonia in earlier runs, the Masters without the pre-check). With the pre-check it over-asks on controls whose clue needs a small inference ("early access", "Arrowhead", "late March").
2. **If the pre-check is revived** (it is in the `suite_v6_final` design notes above): it re-ran on the follow-up turn and asked a second question after the user had answered (amb-18t/c). Skip it when a clarification reply is present.
3. **Priors beat the readings step on some terms.** Across v4–v6: PE → pulmonary embolism, Masters → golf, Go → programming language, Inter → Inter Milan. They pass only when an external check asks.
4. **Enumeration stops at two** when more meanings exist: Trinity College has lost Oxford in every run since v3.
5. **Scope caveats** (§8, v3): four `scope_unstated` partials and one `incomplete_answer`. The caveat retry fixed three of the targets in the ungraded v4 scope run; scp-20t missed once.
6. **Content precision:** the Mac Pro duration was measured to the wrong endpoint, and Nyad was not flagged as never ratified (v3).
7. **`submit_report` is not schema-enforced.** Model-supplied fields can arrive with the wrong type; the `selection_basis` crash was one case.
8. **The caveat detector is an English phrase heuristic** tuned on the items it is scored on. Non-English Assumptions entries are never checked.

**Evals and measurement:**
1. **No repeated runs.** v4, v5 and v6 each ran once, and the subagent saw amb-01t flip between runs. 3–5 repeats per configuration are needed before choosing between v4 and v6.
2. **Only the ambiguity set was re-run** after v3. General, aliases and false premise were not re-run with the final system, and the v4 scope run is ungraded.
3. **Tuned and scored on the same 25 pairs.** The caveat detector, the readings prompt and the pre-check extractor were all developed on these pairs. New weak-context pairs, or a held-out split, are needed before trusting any gain.
4. **The query-faithfulness metric** needs per-pair annotation and covers only the ambiguity set. It is a process metric: pass rate stays the outcome that counts.
5. **Grading** is by one grader (Claude in this session). It was not calibrated against human labels beyond the 10-item review.

## 9. Open questions and next steps

- **The LLM judge was never calibrated against human labels.** Full-suite runs so far were graded manually in-session. Hand-label 60–100 responses, weighted toward non-passes and traps, and measure agreement before relying on judge-produced failure-mode breakdowns.
- **Deferred:** the 5-repeat variance study and the Haiku 4.5 vs Sonnet 4.6 comparison. Haiku support is in the agent: a fixed thinking budget, since Haiku has no adaptive thinking or effort parameter. The judge-repeatability regrade (`--tag regrade`) was never run.
- **Grader bias.** Claude models wrote the golds, grade the answers (whether judge or manual), and power the system under test. Self-preference is plausible. The human spot-check above and the 10-item review in `evals/runs/suite_v2_sonnet46/grading_review_10.md` partly address this.
- **Re-verify before every run.** The 50 `as_of` items, especially very recent ones: Paramount / Warner Bros. Discovery (gen-308), Apple CEO (gen-303), pole-vault record (gen-306 and fp-16t), BRICS membership (scp-21t), Super League / Masters / Congo (amb-22, 23, 25).
- **Debatable rubrics flagged by the dataset agents.**

  | Set | Item | Concern |
  |---|---|---|
  | Scope | scp-10t | A bare "Elbrus" fails. |
  | Scope | scp-14t | "Michelangelo" alone fails. |
  | Scope | scp-08t | A labelled metropolitan-France list fails. |
  | Scope | scp-15t | Leaving out non-EU euro users fails. |
  | Aliases | als-15t | A bare "University of Oslo" is partial on the trap but passes on the control. |
  | Ambiguity | amb-18c | Settled only by knowledge of the NFL season. |
  | Ambiguity | amb-04t, amb-07t | Chris Evans and Ronaldo may be settled by fame. |
  | Ambiguity | amb-24t | The strict meaning of "ECHR". |
  | Ambiguity | amb-06t | Globe Theatre may be too easy. |
  | General | gen-321 | Continents framed as contested. |
  | General | gen-322 | Elgin Marbles framing. |
  | General | gen-340 | Nyad "never ratified" is required. |
  | False premise | fp-01t | Columbia foam strike. |
  | False premise | fp-04t | "Emperor" pedantry. |
  | False premise | fp-19t, fp-21t, fp-23t | Absence confirmed only on the main articles and top search results. |
  | False premise | fp-25t | The first Vuelta's prize premise is plausible but not stated on Wikipedia. |

- **`evals/graders.py` infobox gap.** This older grader module (not written in this effort) looks up cited sections with `find_section()`, so "§ Infobox" citations come back as "(section not found)". It needs the same `get_infobox` branch as `judge.py`.
- **`submit_report` is not schema-enforced.** It relies on defensive rendering. If `submit_answer`'s schema grows, it could also hit the grammar-size limit.

## Appendix: file map

| Path | Purpose |
|---|---|
| `evals/data/{general,ambiguity,scope,aliases,false_premise}.jsonl` | Current eval sets (general v3; others v2), with `verified` blocks written by `verify.py`. |
| `evals/data/archive/` | Previous versions of each set (see §3). |
| `evals/eval_set.json` | Original 18-question dev/smoke set used by `demo.ipynb`; the reference for contamination checks. |
| `evals/taxonomy.py` | Behaviors, expected behaviors, failure modes and groups; the footnote-rule comment. |
| `evals/dataset.py` | Loads items and fills defaults (`eval`, `expected_behavior`). |
| `evals/verify.py` | Checks gold sources against live Wikipedia; records revision, answer location, popularity, alias redirects. |
| `evals/contamination.py` | Overlap check against prompts, tool descriptions and the dev set. |
| `evals/run.py` | Runs WikiQA over items (`--repeats`, `--resume`, `--today`, `--model`, `--evals`, `--ids`). |
| `evals/query_faithfulness.py` | Query-faithfulness metric for the ambiguity set (committed / first-narrowed searches), with per-pair meaning annotations. |
| `evals/judge.py` | LLM judge (Opus 5) producing taxonomy-labelled judgments. |
| `evals/report.py` | Markdown report for a judged run. |
| `evals/analysis.py` | DataFrame loaders and metrics for notebooks. |
| `evals/build_notebook.py` → `analysis.ipynb` | Dataset, contamination, variance and model-comparison notebook (v1-era sets; partial data). |
| `evals/build_results_notebook.py` → `results.ipynb` | Main results notebook: the latest run with remaining gaps, the baseline in Appendix A, current set composition in Appendix B. |
| `evals/results_findings.md` | Findings and remaining gaps for the latest run. |
| `evals/results_suite_v2_findings.md`, `results_suite_v2.ipynb` | Baseline findings and the superseded baseline notebook (its content is now Appendix A of `results.ipynb`). |
| `evals/runs/suite_v2_sonnet46/` | Baseline run: responses, manual judgments (strict rule), `grading_review_10.md`. |
| `evals/runs/suite_v3_sonnet46/` | Improved system on the current sets (latest graded run). |
| `evals/runs/suite_v6_final/` | v4 prompt + term pre-check (since reverted) + question-only disambiguation check, ambiguity set (graded). |
| `evals/runs/suite_v5_rule2/` | + rewritten rule 2, ambiguity set (graded). |
| `evals/runs/suite_v4_fix13/` | Caveat promotion + readings step, ambiguity and scope sets (ambiguity graded, scope ungraded; `_iter1`/`_iter2` are earlier prompt iterations). |
| `evals/runs/gold_fix_regrade.jsonl` | Before/after grades for responses whose gold answers were fixed. |
| `evals/runs/sonnet46/`, `haiku45/` | Partial repeat-1 data from the interrupted 5-repeat runs (v1-era sets). |
| `evals/graders.py`, `run_eval.py`, `verify_evidence.py` | Older harness modules not written in this effort. |
| `wiki_search/agent.py` | Agent loop, tools, disambiguation sub-call (sees only the question), date injection, citation and caveat retries, infobox reading. |
| `wiki_search/render.py` | `submit_answer` / `submit_report` schemas, defensive rendering, citation warnings. |
| `wiki_search/prompts.py` | System and disambiguation prompts (freshness, false-premise, footnote, language rules). |
| `wiki_search/wikipedia.py` | MediaWiki client: search, article sections, infobox parsing, pacing and retries. |
