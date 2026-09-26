## Findings: improved system on the current eval sets

**Grading note.** All 250 responses were graded by hand in the Claude Code session, with no LLM-judge API calls. The grading used the taxonomy in `evals/taxonomy.py` and the **strict footnote rule**: a premise correction, required caveat, alternative reading or scope/metric definition must appear in the headline answer or the reasoning bullets. If it appears only in "Assumptions & gaps", the item is at most partial. Citations were not checked section-by-section. Two gold answers are disputed; both responses were graded as passes:
- amb-13t: Córdoba. Wikipedia now gives 2022 census figures rather than the gold's "2020 census".
- scp-07c: Norway's coastline with islands. *Geography of Norway* gives 83,281 km, and the *Norway* article gives 100,915 km.

### Headline: 234/250 = 93.6%

| set | pass | what it says |
|---|---|---|
| General v3 | 49/50 (98%) | The date and freshness fixes work: date-relative 8/8, recent changes 12/12, non-English 6/6, contested 5/5. |
| Ambiguity v2 | 42/50 (84%) | **The main remaining gap.** Trap recall is 72% (18/25). Controls pass 96%. |
| Scope v2 | 45/50 (90%) | Trap recall 80%. All five misses are partial: the right dominant answer with the caveat buried or missing. |
| Aliases v2 | 48/50 (96%) | Every alias resolves, including misspellings with no redirect. The only miss is a content error that also fails on the control. |
| False premise v2 | 50/50 (100%) | 18/18 corrections are in the headline, and 7/7 not-covered traps have no fabricated figure. |

Over-triggering stays low: 99 of 100 controls pass. The one control failure is amb-18c, a clarification on a debatable control. The v2 failure classes from the baseline no longer appear: no `format_violation`, `date_unaware`, `stale_answer`, or `false_premise_called_not_covered` on any item.

The baseline's 94.0% (Appendix A; 235/250 after the strict-rule regrade) and this run's 93.6% are **not comparable**. The eval sets were rebuilt to be harder in between. Ambiguity went from 98% to 84% because the new traps are harder, not because the system regressed.

### Remaining performance gaps

**1. Missed ambiguity when context seems to settle the question (7 fails + 1 partial of 25 traps).** The misses cluster in the harder subtypes: time_ambiguity 1/3, weak_context 2/4, hidden_ambiguity 2/4, domain_acronym 1/2. The plain acronym, homonym and person traps all pass. Three mechanisms:
- *The agent narrows the query before checking.* On amb-01t ("Inter"), the first search was already `Inter Milan stadium`. The disambiguation-page check runs on the results of that query, so it never fires, and Inter Miami is never considered.
- *A domain prior wins.* On amb-09t, "PE in pregnancy" is answered as pulmonary embolism only, although pre-eclampsia is at least as common. On amb-11t, "Go" (learn it this weekend) is answered as the programming language only, although the board game fits. On amb-22t, "the Masters" is answered as golf only, although snooker also fits. In each case the context fits two readings, and the agent treats it as fitting one.
- *The agent noticed but footnoted.* On amb-20t (Macedonia) and amb-23t (Super League), the alternative readings are written, but only in Assumptions. amb-15t (Trinity College) covers two of the three required readings.

**2. The scope caveat is buried or omitted (5 partials of 25 scope traps).** Each response gives the right dominant figure but not the definition it depends on:
- scp-03t: Chongqing's 32M, with the municipality-vs-city note only in Assumptions.
- scp-10t: Elbrus, with no Caucasus caveat and no Mont Blanc.
- scp-19t: Bologna, with al-Qarawiyyin only in Assumptions.
- scp-20t: Ebert, with no Federal Republic and no Heuss.
- scp-12t: nine US time zones, without the six (states) or four (contiguous) alternatives.

Competing-definition traps are the weakest subtype (3/5).

**3. The footnote pattern is the single biggest lever.** Four of the 16 non-passing items (amb-20t, amb-23t, scp-03t, scp-19t) already contain the needed caveat in the response, but only in Assumptions. On scp-20t the Assumptions note gestures at the scope without naming Heuss. The agent treats the Assumptions section as the place to hedge, and the grading rule treats hedges placed there as not delivered.

**4. Content precision (3 items).**
- als-24t/24c: the "trash can" Mac Pro's unchanged run is measured to the *announcement* of its replacement (~5.5 years). The gold measures to its *discontinuation*: 2,182 days, about 6 years. The same miss appears on both trap and control, so the alias was not the problem.
- gen-340: Nyad's swim is named without noting that it was never ratified, which the rubric requires.

### How to improve the system next (in order of payoff)

1. **Promote buried caveats (targets 4 of the 16 non-passes directly and helps scp-20t).** In `render()` or a validator, detect Assumptions entries that name an alternative reading, definition or correction ("could also refer to", "depends on", "if X is counted", "not to be confused"). Then use one corrective turn, like the existing citation retry, to ask the model to move them into the answer or reasoning. Also tighten the `submit_answer` description: Assumptions is for method notes, not for anything the reader needs to interpret the answer.
2. **Check ambiguity on the user's own words, before rewriting the query.** Require the first search to use the bare ambiguous term from the question, or run a disambiguation lookup on every capitalised term and acronym the question contains. The agent's own qualified query should come only after that (amb-01t, amb-22t).
3. **Add an explicit "readings" step for short or weakly contextual questions.** For each proper noun or acronym, the agent lists the plausible referents and names the words in the question that rule each one out. If nothing rules a reading out, cover it or ask (amb-09t, amb-11t, amb-15t).
4. **Add a scope-note rule for superlatives, "population of", "first X" and counts.** State the definition used in the headline and name the leading alternative with its figure (all five scope partials).
5. **For duration questions, state the endpoints used** and prefer the article's own figure when it gives one (als-24).

### What this says about the eval suite

- **False premise (100%) and aliases (96%)** are at ceiling for this system. They now serve as regression guards. Harder variants would need subtler premises, such as off-by-one numbers or near-miss dates, and aliases that collide with a more popular entity.
- **General v3 (98%)** did its job: it confirmed the date and freshness fixes. It is now near ceiling too.
- **Ambiguity v2** is the most discriminative set. Its weak-context and time-ambiguity traps are where to add items.
- **Scope v2**'s competing-definition traps are the second most useful.

With one run and 50-item sets, one item is 2 points, so treat differences under about 10 points as noise.
