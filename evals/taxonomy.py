"""Predefined labels the judge must choose from.

BEHAVIORS describe what the system did; EXPECTED_BEHAVIORS (on each eval item) say what
it should have done. FAILURE_MODES name why a non-passing item failed. The judge picks
exactly one primary failure mode ("none" on a pass) plus any secondary ones.
"""

BEHAVIORS = {
    "answered": "Gave a direct answer under a single reading of the question.",
    "answered_all_readings": "Answered without asking, but explicitly covered each plausible reading.",
    "asked_clarification": "Stopped and asked the user which meaning they intended.",
    "corrected_premise": "Said the question's premise is false, with or without giving the nearest true facts.",
    "stated_not_covered": "Said Wikipedia does not contain the answer, without claiming the premise is false.",
    "declined": "Refused or gave up for another reason (budget exhausted, error, 'cannot determine').",
}

# Grading rule (decided 2026-09-26): Critical information must not live only in the `assumptions` footnote: a premise correction, a caveat the rubric requires, or the scope/metric an answer depends on must appear in the headline answer or the reasoning. If it appears only under 'Assumptions & gaps', the item is at most partial (the headline went along with the premise / omitted the caveat).
EXPECTED_BEHAVIORS = {
    "answer": "Answer directly. Asking, correcting the premise, or saying 'not covered' fails.",
    "clarify": "Ask which meaning, OR answer all readings when there are only 2-3 and each is short. "
               "Silently picking one reading fails.",
    "answer_with_scope_note": "Answer at the implied granularity; if the scope is underspecified, "
                              "name the scope used (and give the alternative when figures differ a lot).",
    "correct_premise": "Flag the false premise, ideally with the nearest true facts. "
                       "Saying 'not covered by Wikipedia' is only a partial pass.",
    "state_not_covered": "Say Wikipedia does not answer this. Calling the premise false fails; "
                         "a specific invented answer fails.",
    "answer_with_perspectives": "The question is genuinely contested (e.g. sovereignty disputes): give the main "
                                "positions and their standing as Wikipedia describes them. Stating one side as "
                                "settled fact fails.",
}

# code -> (group, definition). Groups let the report roll failures up.
FAILURE_MODES = {
    "none": ("none", "The item passed; no failure."),

    # Ambiguity / referent
    "missed_ambiguity": ("ambiguity", "Silently chose one reading of a genuinely ambiguous term or acronym."),
    "unnecessary_clarification": ("ambiguity", "Asked for clarification although the question's context settles the meaning."),
    "overlong_enumeration": ("ambiguity", "Answered many readings in a wall of text where asking was clearly better."),
    "wrong_referent": ("ambiguity", "Answered about a different entity than the one the question (or the clarification) meant."),

    # Scope
    "scope_too_broad": ("scope", "Answered about a broader entity, period, or category than asked."),
    "scope_too_narrow": ("scope", "Answered about a narrower subset or single instance than asked."),
    "scope_unstated": ("scope", "Scope was underspecified and figures differ materially, but the answer did not say which scope it used."),

    # Names
    "alias_not_resolved": ("naming", "Failed to connect an alias, nickname, or historical name to its article; answered 'not found' or about something else."),
    "anachronistic_answer": ("naming", "Used facts for the modern entity/name when the question is about the historical one (or vice versa)."),

    # Premise / coverage
    "accepted_false_premise": ("premise", "Answered as if a false premise were true."),
    "spurious_premise_correction": ("premise", "Claimed a true premise is false, or hedged/corrected a question that was fine."),
    "false_premise_called_not_covered": ("premise", "On a false-premise question, said 'Wikipedia doesn't cover this' instead of flagging the premise."),
    "not_covered_called_false": ("premise", "On a true-premise question Wikipedia can't answer, claimed the premise is false."),
    "fabricated_answer": ("premise", "Gave a specific answer that Wikipedia does not support (invented detail)."),
    "unwarranted_refusal": ("premise", "Said Wikipedia doesn't cover it, or declined, when the answer is available."),

    # Contested topics
    "one_sided_contested": ("contested", "Presented one side of a genuinely contested question as settled fact."),
    "false_balance": ("contested", "Presented a question with a clear consensus (as Wikipedia states it) as an open debate."),

    # Freshness
    "stale_answer": ("freshness", "Gave an outdated answer (e.g. from memory) where current Wikipedia states a newer fact."),
    "date_unaware": ("freshness", "Computed a date-relative answer (age, years since, still alive) against a wrong 'today'."),

    # Content
    "incorrect_fact": ("content", "The final answer is factually wrong (not attributable to a more specific mode)."),
    "multi_hop_break": ("content", "An intermediate hop was resolved wrongly or skipped, derailing the final answer."),
    "reasoning_error": ("content", "Retrieved the right facts but computed, compared, counted, or ordered them wrongly."),
    "incomplete_answer": ("content", "Correct as far as it goes but omits a part the question requires."),
    "retrieval_miss": ("content", "Never found or opened the article/section holding the answer (visible in the trace)."),

    # Grounding
    "unsupported_citation": ("grounding", "A cited section does not support the claim attached to it."),
    "unsourced_claim": ("grounding", "A load-bearing claim has no citation and is not labeled as background knowledge."),
    "irrelevant_citation": ("grounding", "A cited source is off-topic for the question or for the statement it is attached to."),

    # Operational
    "budget_exhausted": ("operational", "Ran out of tool calls or turns before answering."),
    "tool_error": ("operational", "A tool or API error prevented a proper answer."),
    "format_violation": ("operational", "Ignored the required answer format in a way that hurts usability."),
    "language_mismatch": ("operational", "Answered in a different language from the one the question was asked in."),
    "other": ("other", "None of the above; explain in the rationale."),
}

FAILURE_GROUPS = sorted({g for g, _ in FAILURE_MODES.values()})


def taxonomy_text() -> str:
    """Render the taxonomy for inclusion in the judge prompt."""
    lines = ["Behaviors:"]
    lines += [f"- {k}: {v}" for k, v in BEHAVIORS.items()]
    lines += ["", "Expected behaviors (from the eval item):"]
    lines += [f"- {k}: {v}" for k, v in EXPECTED_BEHAVIORS.items()]
    lines += ["", "Failure modes (group / code: definition):"]
    lines += [f"- {g} / {k}: {d}" for k, (g, d) in FAILURE_MODES.items()]
    return "\n".join(lines)
