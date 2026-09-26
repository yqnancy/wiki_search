"""Check every eval item's evidence quotes against Wikipedia and pin the revisions.

Expected answers in eval_set.json must be backed by quotes from the named article section.
This script fetches each article fresh (bypassing the response cache), confirms each quote
appears verbatim in that section, and with --write records the revision ID and date it was
verified against. Rerun it periodically: a MISSING quote means the article changed and the
expected answer needs another look.

    .venv/bin/python evals/verify_evidence.py            # check only
    .venv/bin/python evals/verify_evidence.py --write    # check and pin revisions
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from wiki_search import wikipedia  # noqa: E402

EVAL_PATH = ROOT / "evals" / "eval_set.json"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def section_text(article: wikipedia.Article, heading: str) -> str | None:
    """Exact heading match (unlike Article.find_section, which also matches substrings)."""
    if heading == "(lead)":
        return article.lead
    for i, s in enumerate(article.sections):
        if s.heading == heading:
            found = article.find_section(heading)
            return found[1] if found else s.text
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="Record revid and verified_on in eval_set.json.")
    args = parser.parse_args()

    wikipedia.CACHE_MODE = "refresh"  # verify against current Wikipedia, and refresh the cache
    items = json.loads(EVAL_PATH.read_text())
    today = datetime.date.today().isoformat()
    failures = 0

    for item in items:
        for ev in item["evidence"]:
            article = wikipedia.get_article(ev["title"])
            if article is None:
                status = "NO ARTICLE"
            elif article.title != ev["title"]:
                status = f"REDIRECTS to {article.title}"
            else:
                text = section_text(article, ev["section"])
                if text is None:
                    status = "NO SECTION"
                elif _norm(ev["quote"]) in _norm(text):
                    status = "ok"
                    if args.write:
                        ev["revid"] = article.revid
                        ev["verified_on"] = today
                else:
                    status = "MISSING"
            failures += status != "ok"
            print(f"{status:>10}  {item['id']:<9} {ev['title']} § {ev['section']}: \"{ev['quote'][:60]}\"")

    pending = [i for i in items if i["review"]["status"] == "needs_decision"]
    print(f"\n{failures} evidence problems. {len(pending)} items need a human decision: "
          + ", ".join(i["id"] for i in pending))
    if args.write:
        EVAL_PATH.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n")
        print(f"Pinned revisions written to {EVAL_PATH.relative_to(ROOT)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
