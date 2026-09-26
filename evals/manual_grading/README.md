# Manual grading

Tools for grading a run by hand (or in a Claude Code session) instead of with the API judge in `evals/judge.py`. All full-suite runs in this project were graded this way. The output uses the judge's schema, so reports, notebooks and `evals.query_faithfulness` work the same either way.

## Workflow

1. **Run the system** through the Claude API as usual:
   ```
   .venv/bin/python -m evals.run --name suite_v7 --today 2026-09-25
   ```
2. **Read the responses next to the gold and rubric.** Use `--start`/`--end` to page through a set, or `--compare` to see two runs under each item:
   ```
   .venv/bin/python -m evals.manual_grading.dump --run suite_v7 --eval ambiguity
   ```
3. **Record grades**, one line per item: `id|behavior|verdict|primary|secondary,modes|rationale`. Values are checked against `evals/taxonomy.py` as you go.
   ```
   .venv/bin/python -m evals.manual_grading.record --out evals/runs/suite_v7/grades.jsonl <<'EOF'
   amb-01t|answered|fail|missed_ambiguity||San Siro only; Inter Miami never mentioned.
   amb-01c|answered|pass|none||75,817.
   EOF
   ```
4. **Convert to judgments** and report:
   ```
   .venv/bin/python -m evals.manual_grading.to_judgments --grades evals/runs/suite_v7/grades.jsonl --run suite_v7
   .venv/bin/python -m evals.report --name suite_v7
   ```
   Add `--partial` when only some of the run's sets were graded.

## Grading rules

- **Verdicts:** pass when the item's `expected_behavior` is met and the facts match the gold (or the rubric's accepted range); partial when the core is right but a required element is missing; fail otherwise. `verdict` is `pass` exactly when `primary_failure_mode` is `none`.
- **Strict footnote rule:** a premise correction, alternative reading, or the scope or definition the answer depends on must appear in the answer or reasoning. If it appears only in "Assumptions & gaps", the item is at most partial.
- **Clarifications:** grade the question itself (the `behavior` is `asked_clarification`) and mark the follow-up answer in the rationale with `FOLLOWUP-FAIL` or `FOLLOWUP-PARTIAL`; no marker means the follow-up passed.
- **Disputed golds:** if the system's answer is supported by Wikipedia but the gold differs, grade on the evidence and put `GOLD-DISPUTE` in the rationale, so the gold can be checked with `evals/verify.py`.
- **Not checked by hand:** whether each cited section supports its claim (grounding) and citation relevance. They are recorded as `not_assessed` unless a citation failure mode was noted.

Grades files are plain JSONL; keeping them with the run (`evals/runs/<run>/grades.jsonl`) means they are ignored by git like the rest of the run outputs.
