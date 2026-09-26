"""Thin client over the MediaWiki Action API (en.wikipedia.org).

These operations back the agent's tools:
- search(): ranked search results, each with a short intro and a disambiguation flag
- get_article(): full plain-text article, split into sections
- get_infobox(): the article's infobox as key-value pairs (from wikitext; extracts omit it)

Responses are cached on disk (SQLite) so repeated runs, e.g. eval reruns, make few
requests and every run reads the same article text. See CACHE_MODE.
"""

from __future__ import annotations

import html
import datetime
import json
import logging
import os
import re
import sqlite3
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from functools import lru_cache
from typing import Optional
from urllib.parse import quote

import requests

API_URL = "https://en.wikipedia.org/w/api.php"
WIKI_BASE = "https://en.wikipedia.org/wiki/"
# Wikimedia asks clients to identify themselves with contact info and throttles generic
# agents harder; set WIKI_USER_AGENT to e.g. "wiki-search/0.1 (you@example.com)".
USER_AGENT = os.environ.get("WIKI_USER_AGENT", "wiki-search/0.1 (educational Q&A tool)")

# Sections that rarely hold answerable content; hidden from tables of contents.
BOILERPLATE_SECTIONS = {
    "see also", "references", "external links", "notes", "further reading",
    "sources", "bibliography", "citations", "footnotes", "notes and references",
}

# Minimum spacing between requests across all threads. Wikimedia throttles bursts
# (HTTP 429), which costs far more time in backoff than steady pacing does.
MIN_REQUEST_INTERVAL_S = float(os.environ.get("WIKI_MIN_REQUEST_INTERVAL", "0.2"))

# Disk cache of API responses. "on": read and write; "refresh": ignore cached entries but
# overwrite them with fresh responses; "off": no cache.
CACHE_MODE = os.environ.get("WIKI_CACHE", "on")
CACHE_PATH = Path(os.environ.get("WIKI_CACHE_PATH", Path(__file__).resolve().parent.parent / ".cache" / "wikipedia.sqlite"))

log = logging.getLogger(__name__)
stats: Counter = Counter()  # cache_hits, requests, retries; reset with stats.clear()
_session = requests.Session()
_session.headers["User-Agent"] = USER_AGENT
_pace_lock = threading.Lock()
_last_request = 0.0
_db_lock = threading.Lock()
_db: Optional[sqlite3.Connection] = None


def _cache() -> sqlite3.Connection:
    global _db
    if _db is None:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _db = sqlite3.connect(str(CACHE_PATH), check_same_thread=False)
        _db.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, body TEXT, fetched_at REAL)")
    return _db


def _pace() -> None:
    global _last_request
    with _pace_lock:
        wait = _last_request + MIN_REQUEST_INTERVAL_S - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()


def _get(params: dict, retries: int = 4) -> dict:
    params = {"action": "query", "format": "json", "formatversion": 2, **params}
    key = json.dumps(params, sort_keys=True)
    if CACHE_MODE == "on":
        with _db_lock:
            row = _cache().execute("SELECT body FROM responses WHERE key = ?", (key,)).fetchone()
        if row:
            stats["cache_hits"] += 1
            return json.loads(row[0])

    for attempt in range(retries + 1):
        _pace()
        stats["requests"] += 1
        resp = _session.get(API_URL, params=params, timeout=20)
        if resp.status_code not in (429, 500, 502, 503, 504) or attempt == retries:
            break
        # Honor Retry-After, else back off exponentially.
        retry_after = resp.headers.get("Retry-After", "")
        delay = float(retry_after) if retry_after.isdigit() else 2 ** attempt
        stats["retries"] += 1
        log.warning("Wikipedia HTTP %s; retrying in %.0fs", resp.status_code, delay)
        time.sleep(delay)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"Wikipedia API error: {data['error'].get('info', data['error'])}")

    if CACHE_MODE in ("on", "refresh"):
        with _db_lock:
            _cache().execute("INSERT OR REPLACE INTO responses VALUES (?, ?, ?)", (key, resp.text, time.time()))
            _cache().commit()
    return data


def page_url(title: str, section: Optional[str] = None) -> str:
    url = WIKI_BASE + quote(title.replace(" ", "_"), safe="()_,'")
    if section:
        url += "#" + quote(section.replace(" ", "_"), safe="()_,'")
    return url


@dataclass
class SearchResult:
    rank: int  # Wikipedia's relevance order; 0 = disambiguation page found by direct lookup
    title: str
    url: str
    intro: str
    is_disambiguation: bool


# Only short queries get the direct "(disambiguation)" lookup: those are the bare-name
# searches it exists for, and it costs a second request.
DISAMBIGUATION_LOOKUP_MAX_WORDS = 3


def search(query: str, limit: int = 10, intro_chars: int = 600) -> list[SearchResult]:
    """Full-text search in Wikipedia's relevance order, with each page's intro and
    disambiguation flag, in one request (generator=search feeds the results to prop=).

    Note: the API returns intros for at most 20 pages per request.
    """
    data = _get({
        "generator": "search", "gsrsearch": query, "gsrlimit": limit,
        "prop": "extracts|pageprops", "exintro": 1, "explaintext": 1, "exlimit": "max",
        "ppprop": "disambiguation",
    })
    pages = sorted(data.get("query", {}).get("pages", []), key=lambda p: p["index"])
    results = [
        SearchResult(
            rank=p["index"],
            title=p["title"],
            url=page_url(p["title"]),
            intro=_shorten(p.get("extract") or "", intro_chars),
            is_disambiguation="disambiguation" in p.get("pageprops", {}),
        )
        for p in pages
    ]
    if results and len(query.split()) <= DISAMBIGUATION_LOOKUP_MAX_WORDS:
        extra = _lookup_disambiguation(query, {r.title for r in results}, intro_chars)
        if extra:
            results.insert(0, extra)
    return results


def _lookup_disambiguation(query: str, exclude: set[str], intro_chars: int) -> Optional[SearchResult]:
    """Terms like "Paris" have a primary article plus a separate "(disambiguation)" page
    that search may rank low or not at all; fetch it directly."""
    page = _get({
        "titles": f"{query.strip()} (disambiguation)", "redirects": 1,
        "prop": "extracts|pageprops", "exintro": 1, "explaintext": 1,
        "ppprop": "disambiguation",
    })["query"]["pages"][0]
    if page.get("missing") or page["title"] in exclude or "disambiguation" not in page.get("pageprops", {}):
        return None
    return SearchResult(0, page["title"], page_url(page["title"]),
                        _shorten(page.get("extract") or "", intro_chars), True)


def _shorten(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " …"


@dataclass
class Section:
    heading: str
    level: int  # 1 = lead, 2 = "== H ==", 3 = "=== H ===", ...
    text: str


@dataclass
class Article:
    title: str
    url: str
    is_disambiguation: bool
    revid: int  # revision the text came from, for pinning eval evidence
    sections: list[Section] = field(default_factory=list)

    @property
    def lead(self) -> str:
        return self.sections[0].text if self.sections else ""

    def toc(self) -> list[str]:
        """Content section headings (indented by depth), boilerplate removed."""
        return [
            "  " * (s.level - 2) + s.heading
            for s in self.sections[1:]
            if s.heading.lower() not in BOILERPLATE_SECTIONS
        ]

    def find_section(self, name: str) -> Optional[tuple[Section, str]]:
        """Case-insensitive exact match, then substring match.

        Returns the section's heading and its text including all subsections.
        """
        wanted = name.strip().strip("=# ").lower()
        idx = next((i for i, s in enumerate(self.sections[1:], 1) if s.heading.lower() == wanted), None)
        if idx is None:
            idx = next((i for i, s in enumerate(self.sections[1:], 1) if wanted in s.heading.lower()), None)
        if idx is None:
            return None
        head = self.sections[idx]
        parts = [head.text]
        for sub in self.sections[idx + 1:]:
            if sub.level <= head.level:
                break
            if sub.text:
                parts.append(f"{'=' * sub.level} {sub.heading} {'=' * sub.level}\n{sub.text}")
        return head, "\n\n".join(p for p in parts if p)


_HEADING_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.MULTILINE)


@lru_cache(maxsize=256)
def get_article(title: str) -> Optional[Article]:
    """Fetch an article as plain text and split it on its section headings."""
    pages = _get({
        "titles": title, "redirects": 1, "prop": "extracts|pageprops|revisions",
        "explaintext": 1, "exsectionformat": "wiki", "ppprop": "disambiguation", "rvprop": "ids",
    })["query"]["pages"]
    page = pages[0]
    if page.get("missing") or "extract" not in page:
        return None

    text = page["extract"]
    sections = []
    matches = list(_HEADING_RE.finditer(text))
    lead_end = matches[0].start() if matches else len(text)
    sections.append(Section("(lead)", 1, text[:lead_end].strip()))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append(Section(m.group(2).strip(), len(m.group(1)), text[m.end():end].strip()))

    return Article(
        title=page["title"],
        url=page_url(page["title"]),
        is_disambiguation="disambiguation" in page.get("pageprops", {}),
        revid=page.get("revisions", [{}])[0].get("revid", 0),
        sections=sections,
    )


# ---------------------------------------------------------------------- infoboxes
# Plain-text extracts omit infoboxes, so facts that live only there (runtimes, elevations,
# capacities) need the wikitext. The parser below is deliberately rough: it handles links,
# refs, comments, <br> and common templates (convert, dates, lists), but values built from
# unusual templates or parser functions may come out garbled or missing. Fields an infobox
# pulls from Wikidata at render time (e.g. mountain elevations with fetchwikidata=ALL) are
# absent from the wikitext and so missing here.

_INFOBOX_SKIP_KEYS = re.compile(r"^(image|logo|map|signature|alt|caption|pushpin|embed|fetchwikidata|onlysourced)|_(size|alt|ref|footnotes)$|^module\d*$")
_LIST_TEMPLATES = {"plainlist", "plain list", "ubl", "unbulleted list", "hlist", "flatlist", "flat list", "bulleted list", "enum", "collapsible list"}
_DROP_TEMPLATES = {"wikidata", "notetag", "efn", "efn-ua", "efn-lr", "sfn", "sfnp", "refn", "citation needed", "cn", "rp", "better source needed", "dead link", "clarify"}


def _split_params(body: str) -> list[str]:
    """Split a template body on top-level pipes (not those inside nested {{ }} or [[ ]])."""
    parts, depth, start, i = [], 0, 0, 0
    while i < len(body):
        pair = body[i:i + 2]
        if pair in ("{{", "[["):
            depth += 1
            i += 2
        elif pair in ("}}", "]]"):
            depth -= 1
            i += 2
        else:
            if body[i] == "|" and depth == 0:
                parts.append(body[start:i])
                start = i + 1
            i += 1
    parts.append(body[start:])
    return parts


def _render_template(body: str) -> str:
    """Reduce one innermost template to its first values, e.g. {{convert|194|min}} -> "194 min"."""
    name, *params = _split_params(body)
    name = name.strip().lower().replace("_", " ")
    positional = [p.strip() for p in params if "=" not in p.split("[[")[0]]
    if name in _DROP_TEMPLATES or name.startswith("cite "):
        return ""
    if name in _LIST_TEMPLATES:
        items = [line.strip("*# ") for p in positional for line in p.split("\n")]
        return ", ".join(i for i in items if i)
    if "date" in name:  # {{film date|1997|11|1|Tokyo|1997|12|19|US}} -> "1997-11-01 (Tokyo), 1997-12-19 (US)"
        out, run = [], []
        for p in positional:
            if p.isdigit():
                run.append(p.zfill(2))
                continue
            if run:
                out.append("-".join(run[:3]))
                run = []
            if out and p:
                out[-1] += f" ({p})"
        return ", ".join(out + (["-".join(run[:3])] if run else []))
    if name in ("convert", "cvt"):  # {{convert|105|x|68|m|ft}} -> "105 x 68 m"
        out = []
        for p in positional:
            out.append(p)
            if not re.fullmatch(r"[\d.,]+|x|by|to|and|-|–", p):
                break  # the first unit
        return " ".join(out)
    return " ".join(positional)


def _strip_notes(text: str) -> str:
    """Remove comments and <ref>s, which may contain pipes, so do this before splitting params."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    return re.sub(r"<ref[^>/]*/>|<ref[^>]*>.*?</ref>", "", text, flags=re.S | re.I)


def _clean_wikitext(text: str) -> str:
    text = re.sub(r"<br\s*/?>", ", ", text, flags=re.I)
    for _ in range(10):  # innermost templates first, until none are left
        text, n = re.subn(r"\{\{([^{}]*)\}\}", lambda m: _render_template(m.group(1)), text)
        if not n:
            break
    text = re.sub(r"\[\[(?:File|Image):[^\[\]]*(?:\[\[[^\]]*\]\][^\[\]]*)*\]\]", "", text, flags=re.I)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\s+([^\]]*)\]", r"\1", text)
    text = re.sub(r"'{2,}", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text).replace("\u00a0", " ")
    text = re.sub(r"^\s*[*#:]+\s*", "", text, flags=re.M)  # list bullets
    text = re.sub(r"\s*\n\s*", ", ", text.strip())
    return re.sub(r"\s{2,}", " ", re.sub(r"(?:,\s*){2,}", ", ", text)).strip(" ,")


def parse_infobox(wikitext: str) -> list[tuple[str, str]]:
    """(key, value) pairs of the first {{Infobox ...}} template in `wikitext`, cleaned to plain text."""
    start = re.search(r"\{\{\s*Infobox", wikitext, re.I)
    if not start:
        return []
    depth, i = 0, start.start()
    while i < len(wikitext):
        if wikitext.startswith("{{", i):
            depth += 1
            i += 2
        elif wikitext.startswith("}}", i):
            depth -= 1
            i += 2
            if depth == 0:
                break
        else:
            i += 1
    fields = []
    for param in _split_params(_strip_notes(wikitext[start.end() - len("Infobox"):i - 2]))[1:]:
        key, sep, value = param.partition("=")
        key = key.strip().replace("_", " ")
        if not sep or not key or _INFOBOX_SKIP_KEYS.search(key.replace(" ", "_").lower()):
            continue
        value = _clean_wikitext(value)
        if value:
            fields.append((key, value))
    return fields


@lru_cache(maxsize=256)
def get_infobox(title: str) -> list[tuple[str, str]]:
    """The article's infobox as (key, value) pairs; empty if it has none. Reads only section 0."""
    page = _get({
        "titles": title, "redirects": 1, "prop": "revisions",
        "rvprop": "content", "rvslots": "main", "rvsection": 0,
    })["query"]["pages"][0]
    if page.get("missing") or not page.get("revisions"):
        return []
    return parse_infobox(page["revisions"][0]["slots"]["main"].get("content", ""))


@lru_cache(maxsize=1024)
def revision_timestamp(revid: int) -> Optional[datetime.datetime]:
    """When a specific revision was saved (UTC). Revisions never change, so the cached answer
    stays correct; used to check how recently a cited page had been updated."""
    if not revid:
        return None
    pages = _get({"revids": revid, "prop": "revisions", "rvprop": "timestamp"})["query"].get("pages", [])
    revisions = pages[0].get("revisions", []) if pages else []
    if not revisions:
        return None
    return datetime.datetime.strptime(revisions[0]["timestamp"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)
