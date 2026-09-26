"""Load eval items from evals/data/*.jsonl, filling defaults."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent / "data"
EVAL_NAMES = ["general", "ambiguity", "scope", "aliases", "false_premise"]


def load_items(evals: Optional[list[str]] = None, by_file: bool = False):
    """All items (or those in the named evals). With by_file, a {path: [items]} dict."""
    out = {}
    for name in evals or EVAL_NAMES:
        path = DATA_DIR / f"{name}.jsonl"
        items = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            item.setdefault("eval", name)
            item.setdefault("expected_behavior", "answer")
            items.append(item)
        out[path] = items
    if by_file:
        return out
    return [item for items in out.values() for item in items]
