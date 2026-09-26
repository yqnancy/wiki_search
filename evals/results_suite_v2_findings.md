## Findings

**Grading note.** This run was graded by hand in the Claude Code session (no LLM-judge API calls), using the same taxonomy and pass rules as `evals/judge.py`. Citations were not checked section-by-section; only visible citation defects (invalid indexes, empty citation lists) were recorded. One gold answer was corrected during grading: gen-210 (Twitter/X ownership). The system's own answer surfaced that the *X* article lead now names SpaceXAI.

**Headline: 238/250 = 95.2%.** Scope 100%, aliases 98%, ambiguity 98%, false premise 92%, general 88%. Over-triggering is essentially absent: control pass rates are 96-100%. The system asks for clarification when a term is genuinely ambiguous (every clarification's follow-up answer was correct), names its scope, resolves nicknames with no Wikipedia redirect, handles contested topics with both sides, and never invented an answer on the seven "not on Wikipedia" traps.

**The 12 non-passing items come from five causes:**

| Cause | Items | Failure mode |
|---|---|---|
| `assumptions` submitted as a string, rendered one bullet per character | 7 (gen-211, gen-246, amb-25t, als-07t, fp-02t, fp-12c, fp-16t) | `format_violation` |
| No notion of today's date: "35 as of 2025", "39 years ago" | 2 (gen-239, gen-240) | `date_unaware` |
| Hedged on the current owner of X instead of leading with what the article lead says | 1 (gen-210) | `stale_answer` |
| Headline answer "Not covered by Wikipedia" for an invented war it had correctly identified as nonexistent | 1 (fp-12t) | `false_premise_called_not_covered` |
| Wrong background claim inside `assumptions` (Denali's federal name history) | 1 (gen-206) | `incorrect_fact` |

Secondary issues on passing items: statements with empty citation lists (3), a citation index pointing past the sources list (1), reasoning in English under a non-English question (2), a weak inference (1), and one infobox-only fact (the *Titanic* film's runtime) that the tool could not read and correctly labelled as unconfirmed.

## How to improve the system (in order of payoff)

1. **Enforce the `submit_answer` schema (fixes 7 of 12 failures).** Mark `submit_answer` and `submit_report` as `strict: true` so `assumptions`/`reasoning`/`sources` must be arrays, and make `render()` defensive (wrap a bare string in a list). This is a one-line class of bug that silently turns a correct answer into an unreadable one.
2. **Tell the agent today's date.** Put the current date in the system prompt (in a block after the cached prefix, or in the first user turn) and instruct it to compute ages and elapsed times from that date. Fixes both `date_unaware` failures and helps every "current / now / latest" question.
3. **Lead with the freshest statement for time-sensitive questions.** For "who owns / who is the current / latest" questions, instruct the agent to answer from the article lead's present-tense statement and mention superseded facts only as history, rather than hedging (gen-210).
4. **Keep premise corrections in the headline, and separate "false premise" from "not covered".** The answer field should state the correction ("No such war occurred"), and `question_type: not-covered` should be reserved for true facts Wikipedia lacks (fp-12t; also fp-17t, where the correction sat only in `assumptions`).
5. **Validate citations before accepting a submission.** `render()` already detects invalid indexes and uncited statements; feed those warnings back to the model for one corrective turn instead of only logging them. Also forbid new factual claims in `assumptions` unless cited (gen-206's error lived there).
6. **Expose infoboxes.** Add an "Infobox" pseudo-section to `read_article` (parsed from wikitext) so facts like film runtimes, elevations and capacities, which appear only in infoboxes, become citable.
7. **Answer entirely in the user's language**, not only the headline field.

## What this says about the eval suite

The paired failure-mode sets are now close to ceiling for Sonnet 4.6 (scope 100%, aliases and ambiguity 98%), so they mainly guard against regressions. The v2 general set is the most informative: its failures cluster exactly where designed (freshness, date awareness), and it surfaced a real rendering bug. Next steps: harder paired variants (ambiguity where context only weakly disambiguates, aliases with misspellings, subtler false premises), more date- and freshness-sensitive items, and checking citations section-by-section. With one run and 50-item sets, a single item is 2 points; treat differences under ~10 points as noise.
