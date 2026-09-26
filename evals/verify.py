"""Check every eval item's gold sources against live Wikipedia and annotate the items.

For each gold source {title, contains} this resolves redirects, pins the revision id, and
records where the evidence string appears:
    lead   - in the article's opening section (plain-text extract)
    body   - in a later section of the plain-text extract
    markup - only in the wikitext (infobox, table, template), which the agent's
             read_article tool never sees
    missing - nowhere: the item needs fixing

Per item it also records the primary article's monthly page views (popularity bucket),
the answer location (location of the last gold source, i.e. the final hop), and for alias
traps whether the alias has a redirect and where search ranks the canonical article.

    .venv/bin/python -m evals.verify            # report only
    .venv/bin/python -m evals.verify --write    # also write a `verified` block into each item
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from urllib.parse import quote

from wiki_search import wikipedia

from .dataset import DATA_DIR, load_items

PAGEVIEWS_URL = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
                 "en.wikipedia/all-access/user/{title}/monthly/{start}/{end}")
HEAD_VIEWS = 100_000  # monthly views at or above: head
TAIL_VIEWS = 10_000   # below: tail; in between: mid


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    return re.sub(r"\s+", " ", text)


CACHE_DIR = Path(__file__).parent / ".cache"
_page_cache: dict[str, dict] = {}


def _disk_cached(kind: str, key: str, fetch):
    """Memoize Wikipedia lookups on disk for the day, so re-running after fixes is cheap."""
    path = CACHE_DIR / kind / (hashlib.sha1(f"{dt.date.today()}|{key}".encode()).hexdigest() + ".json")
    if path.exists():
        return json.loads(path.read_text())
    value = fetch()
    if value is not None:  # don't pin transient failures
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    return value


def fetch_page(title: str) -> dict:
    """Resolved title, revid, plain-text lead/body, and raw wikitext for an article."""
    if title not in _page_cache:
        _page_cache[title] = _disk_cached("pages", title, lambda: _fetch_page(title))
    return _page_cache[title]


def _fetch_page(title: str) -> dict:
    data = wikipedia._get({
        "titles": title, "redirects": 1, "prop": "revisions|pageprops",
        "rvprop": "ids|content", "rvslots": "main", "ppprop": "disambiguation",
    })["query"]
    page = data["pages"][0]
    info = {"requested": title, "exists": not page.get("missing")}
    if info["exists"]:
        rev = page["revisions"][0]
        article = wikipedia.get_article(page["title"])
        info.update(
            title=page["title"],
            revid=rev["revid"],
            redirected=page["title"] != title,
            is_disambiguation="disambiguation" in page.get("pageprops", {}),
            lead=article.lead if article else "",
            body="\n".join(s.text for s in article.sections[1:]) if article else "",
            wikitext=rev["slots"]["main"]["content"],
        )
    return info


def locate(page: dict, needle: str) -> str:
    if not page["exists"]:
        return "missing_article"
    n = _norm(needle)
    if n in _norm(page["lead"]):
        return "lead"
    if n in _norm(page["body"]):
        return "body"
    if n in _norm(page["wikitext"]):
        return "markup"
    return "missing"


def monthly_views(title: str) -> int | None:
    return _disk_cached("views", title, lambda: _monthly_views(title))


def _monthly_views(title: str) -> int | None:
    today = dt.date.today().replace(day=1)
    last_month_end = today - dt.timedelta(days=1)
    start = last_month_end.replace(day=1)
    url = PAGEVIEWS_URL.format(title=quote(title.replace(" ", "_"), safe=""),
                               start=start.strftime("%Y%m%d00"), end=last_month_end.strftime("%Y%m%d00"))
    for attempt in range(5):
        wikipedia._pace()
        resp = wikipedia._session.get(url, timeout=20)
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After", "")
            time.sleep(float(retry_after) if retry_after.isdigit() else 2 ** (attempt + 2))
            continue
        if resp.status_code != 200:
            return None
        return sum(i["views"] for i in resp.json().get("items", []))
    return None


def popularity(views: int | None) -> str | None:
    if views is None:
        return None
    return "head" if views >= HEAD_VIEWS else "tail" if views < TAIL_VIEWS else "mid"


def alias_status(alias: str, canonical: str) -> dict:
    """Does the alias redirect to the canonical article, and where does search rank it?"""
    bare = re.sub(r"^the ", "", alias, flags=re.I)
    page = fetch_page(bare[0].upper() + bare[1:])
    canonical_resolved = fetch_page(canonical).get("title", canonical)
    if not page["exists"]:
        redirect = "none"
    elif page["is_disambiguation"]:
        redirect = "disambiguation"
    elif page["title"] == canonical_resolved:
        redirect = "redirect" if page["redirected"] else "is_title"
    else:
        redirect = f"other:{page['title']}"
    results = _disk_cached("search", bare, lambda: [r.title for r in wikipedia.search(bare, limit=10)])
    rank = next((i for i, t in enumerate(results, 1) if t == canonical_resolved), None)
    return {"alias_redirect": redirect, "alias_search_rank": rank}


def verify_item(item: dict) -> dict:
    sources = []
    for src in item.get("gold_sources", []):
        page = fetch_page(src["title"])
        sources.append({
            "title": src["title"],
            "resolved_title": page.get("title"),
            "revid": page.get("revid"),
            "location": locate(page, src["contains"]),
        })
    out = {"sources": sources, "checked": dt.date.today().isoformat()}
    if sources:
        out["answer_location"] = sources[-1]["location"]
        primary = sources[0]["resolved_title"]
        views = monthly_views(primary) if primary else None
        out["primary_monthly_views"] = views
        out["popularity"] = popularity(views)
    if item.get("alias") and item.get("canonical_title"):
        out.update(alias_status(item["alias"], item["canonical_title"]))
    out["problems"] = [
        f"{s['title']}: {s['location']}" for s in sources if s["location"] in ("missing", "missing_article")
    ] + [f"{s['title']} redirects to {s['resolved_title']}" for s in sources
         if s["resolved_title"] and s["resolved_title"] != s["title"]]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="Write a `verified` block into each item.")
    parser.add_argument("--evals", nargs="*", help="Only these eval files (e.g. general scope).")
    parser.add_argument("--ids", nargs="*", help="Only these item ids.")
    args = parser.parse_args()

    by_file = load_items(args.evals, by_file=True)
    n_problems = 0
    for path, items in by_file.items():
        for item in items:
            if args.ids and item["id"] not in args.ids:
                continue
            v = verify_item(item)
            item["verified"] = v
            flag = "PROBLEM" if v["problems"] else "ok"
            n_problems += bool(v["problems"])
            extra = ""
            if "alias_redirect" in v:
                extra = f" alias={v['alias_redirect']} rank={v['alias_search_rank']}"
            print(f"{flag:7} {item['id']:9} loc={v.get('answer_location')} pop={v.get('popularity')}{extra}"
                  + (f"  -> {'; '.join(v['problems'])}" if v["problems"] else ""), flush=True)
        if args.write:
            with open(path, "w") as f:
                for item in items:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\n{n_problems} item(s) with problems.", file=sys.stderr)


if __name__ == "__main__":
    main()
