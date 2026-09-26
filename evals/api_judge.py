"""Optional: fill grading requests through the Anthropic API instead of in a Claude Code session."""

from __future__ import annotations

import json
from typing import Optional

import anthropic

from evals.graders import JudgeRequest

JUDGE_MODEL = "claude-opus-5"
# Safety classifiers occasionally decline benign grading requests: in the pilot, 'bio' false
# positives on answers about mercury the element and light bulbs, on every model tried, depending on
# the exact input. Declined requests are retried on a fallback model, then with a framing preface.
JUDGE_FALLBACK_MODEL = "claude-opus-4-8"
_RETRY_PREFACE = ("For context: the response below is an encyclopedic answer based on Wikipedia, "
                  "submitted for automated quality review.\n\n")


class JudgeError(Exception):
    pass


class Judge:
    def __init__(self, model: str = JUDGE_MODEL, client: Optional[anthropic.Anthropic] = None, effort: str = "medium"):
        self.model = model
        self.client = client or anthropic.Anthropic()
        self.effort = effort

    def __call__(self, request: JudgeRequest) -> tuple[dict, dict]:
        """Returns (verdict, usage). Raises JudgeError if every attempt is declined or truncated."""
        attempts = [(self.model, request.user), (JUDGE_FALLBACK_MODEL, request.user),
                    (self.model, _RETRY_PREFACE + request.user)]
        for model, content in attempts:
            response = self.client.messages.create(
                model=model,
                max_tokens=16000,
                system=request.system,
                messages=[{"role": "user", "content": content}],
                thinking={"type": "adaptive"},
                output_config={"format": {"type": "json_schema", "schema": request.schema}, "effort": self.effort},
            )
            if response.stop_reason != "refusal":
                break
        usage = {"model": model, "input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
        if response.stop_reason == "refusal":
            raise JudgeError(f"declined to grade ({getattr(response.stop_details, 'category', None)})")
        if response.stop_reason == "max_tokens":
            raise JudgeError("output truncated")
        return json.loads(next(b.text for b in response.content if b.type == "text")), usage
