"""Query faithfulness: do the agent's searches keep the question's level of ambiguity?

A search that refers to the ambiguous term (by the term itself or by a word that picks one
of its readings) *narrows* when it adds, relative to the question, words that pick a strict
subset of the readings ("Inter Milan stadium" for "Inter"; "pulmonary embolism" for "PE").
Rewording, dropping words and adding words that fit every reading ("stadium", "2026") are
not narrowing; a search with the term and no such words is *open*. Two measures per item:

- committed: every turn-1 search on the term narrows, and all of them pick the same single
  reading (no open search, no other reading ever searched). The headline
  measure: the agent committed to a reading through search alone.
- first_narrowed: the first search on the term narrows, so the disambiguation check can't
  fire on it. Searching each reading in turn ("NY Rangers ...", "Texas Rangers ...") is
  first_narrowed but not committed.

    .venv/bin/python -m evals.query_faithfulness --runs suite_v3_sonnet46 [more runs...]

On traps a narrowed first search pre-empts the disambiguation check; on controls it is
usually justified, because the question itself names the reading (reported for context).
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from .dataset import load_items

RUNS = Path(__file__).parent / "runs"

# pair_id -> (term regex, {reading: [discriminator regexes]}). Regexes match lowercased text
# with word boundaries added; a discriminator counts only if the question doesn't contain it.
READINGS: dict[str, tuple[str, dict[str, list[str]]]] = {
    "amb-01": (r"inter", {
        "Inter Milan": ["milan", "internazionale", "san siro", "meazza", "ital(y|ian)", "serie a"],
        "Inter Miami": ["miami", "messi", "mls", "nu stadium", "chase stadium", "florida"],
        "SC Internacional": ["internacional", "porto alegre", "brazil", "beira-rio"]}),
    "amb-02": (r"daddy[ -]?long[ -]?legs", {
        "harvestman": ["harvest(man|men)", "opiliones", "arachnids?"],
        "cellar spider": ["cellar spiders?", "pholcid(ae)?", "spiders?", "arachnids?"],
        "crane fly": ["crane ?fl(y|ies)", "tipulidae", "insects?"]}),
    "amb-03": (r"rangers?", {
        "Rangers F.C.": ["glasgow", "scot(tish|land)", "f\\.?c\\.?", "football", "premiership", "ibrox"],
        "Texas Rangers": ["texas", "world series", "baseball", "mlb"],
        "New York Rangers": ["new york", "stanley cup", "nhl", "hockey"]}),
    "amb-04": (r"chris(topher)? evans", {
        "actor": ["actor", "captain america", "marvel", "american", "boston"],
        "broadcaster": ["presenter", "broadcaster", "top gear", "radio", "dj", "english", "british"]}),
    "amb-05": (r"michael collins", {
        "Irish leader": ["irish", "ireland", "revolutionary", "cork", "sinn f[eé]in", "ira"],
        "astronaut": ["astronaut", "apollo", "nasa", "columbia"]}),
    "amb-06": (r"globe", {
        "original Globe": ["1599", "original", "elizabethan", "burbage"],
        "Shakespeare's Globe": ["shakespeare'?s globe", "reconstruct(ion|ed)", "wanamaker", "1997", "modern"]}),
    "amb-07": (r"ronaldo", {
        "Cristiano Ronaldo": ["cristiano", "portug(al|uese)", "cr7"],
        "Ronaldo Nazário": ["naz[aá]rio", "brazil(ian)?", "r9"]}),
    "amb-08": (r"john paul", {
        "John Paul I": ["john paul i", "luciani", "33 days"],
        "John Paul II": ["john paul ii", "wojty[lł]a", "polish"]}),
    "amb-09": (r"pe", {
        "pre-eclampsia": ["pre-?eclampsia", "eclampsia", "hypertension", "blood pressure", "proteinuria"],
        "pulmonary embolism": ["pulmonary", "embolism", "d-?dimer", "ctpa", "v/q"]}),
    "amb-10": (r"ada", {
        "American Dental Association": ["dental", "dentists?"],
        "American Diabetes Association": ["diabetes"],
        "Americans with Disabilities Act": ["disabilit(y|ies)", "act"]}),
    "amb-11": (r"go", {
        "Go (programming language)": ["programming", "language", "golang", "google", "compiler", "pike", "thompson", "griesemer"],
        "Go (game)": ["board ?game", "game", "baduk", "weiqi", "igo"]}),
    "amb-12": (r"rust", {
        "Rust (programming language)": ["programming", "language", "mozilla", "rustc"],
        "Rust (video game)": ["video ?game", "game", "facepunch", "early access"],
        "Rust (film)": ["film", "movie", "baldwin"]}),
    "amb-13": (r"c[oó]rdoba", {
        "Córdoba, Spain": ["españa", "spain", "andaluc[ií]a", "andalusia", "guadalquivir"],
        "Córdoba, Argentina": ["argentina"]}),
    "amb-14": (r"saint-denis|saint denis", {
        "Saint-Denis (Seine-Saint-Denis)": ["seine-saint-denis", "paris", "[iî]le-de-france", "basilique", "basilica"],
        "Saint-Denis, Réunion": ["r[ée]union"]}),
    "amb-15": (r"trinity college", {
        "Dublin": ["dublin", "ireland", "liffey"],
        "Cambridge": ["cambridge", "cam"],
        "Oxford": ["oxford", "thames", "cherwell"],
        "Hartford": ["hartford", "connecticut"]}),
    "amb-16": (r"santiago", {
        "Santiago de Compostela": ["compostela", "galicia", "spain", "camino"],
        "Santiago, Chile": ["chile", "metropolitan cathedral"],
        "Santiago de Cuba": ["cuba"]}),
    "amb-17": (r"miami", {
        "Miami Hurricanes": ["hurricanes", "university of miami", "florida", "coral gables", "acc"],
        "Miami RedHawks": ["redhawks", "ohio", "oxford", "miami university", "mac", "mid-american"],
        "Dolphins / Heat": ["dolphins", "nfl", "heat", "nba"]}),
    "amb-18": (r"panthers?", {
        "Florida Panthers": ["florida", "nhl", "hockey", "sunrise", "amerant"],
        "Carolina Panthers": ["carolina", "nfl", "charlotte", "football", "bank of america"]}),
    "amb-19": (r"the office|office", {
        "UK Office": ["uk", "british", "gervais", "brent", "slough", "bbc"],
        "US Office": ["us", "american", "carell", "michael scott", "dunder mifflin", "scranton", "nbc"]}),
    "amb-20": (r"macedonia", {
        "North Macedonia": ["north", "republic", "skopje", "country"],
        "Greek Macedonia": ["greece", "greek", "thessaloniki", "region"]}),
    "amb-21": (r"kansas city", {
        "Kansas City, Missouri": ["missouri"],
        "Kansas City, Kansas": ["kansas city,? kansas", "kck", "wyandotte"]}),
    "amb-22": (r"masters?", {
        "Masters Tournament (golf)": ["golf", "augusta", "tournament", "mcilroy"],
        "Masters (snooker)": ["snooker", "alexandra palace", "kyren wilson"]}),
    "amb-23": (r"super league", {
        "Super League (rugby league)": ["rugby", "grand final", "hull", "old trafford"],
        "Chinese Super League": ["chin(a|ese)", "csl", "football"],
        "Indian Super League": ["india(n)?", "isl", "football"],
        "Super League Greece": ["gree(ce|k)", "football"],
        "Swiss Super League": ["swiss", "switzerland", "football"],
        "Women's Super League": ["wom[ae]n'?s?", "wsl", "football"]}),
    "amb-24": (r"echr", {
        "European Convention on Human Rights": ["convention"],
        "European Court of Human Rights": ["court"]}),
    "amb-25": (r"congo", {
        "DR Congo": ["democratic", "drc", "kinshasa", "tshisekedi", "goma"],
        "Republic of the Congo": ["(?<!democratic )republic of (the )?congo", "brazzaville", "sassou"]}),
}


def _has(pattern: str, text: str) -> bool:
    return re.search(rf"(?<![\w-]){pattern}(?![\w-])", text) is not None


def classify(item: dict, response: dict) -> dict:
    """Classify the first turn-1 search that refers to the item's ambiguous term."""
    term, readings = READINGS[item["pair_id"]]
    question = item["question"].lower()
    searches = [e["query"] for e in (response.get("turn1") or {}).get("trace", []) if e.get("type") == "search"]
    all_disc = [d for ds in readings.values() for d in ds]
    on_term = [q for q in searches if _has(term, q.lower()) or any(_has(d, q.lower()) for d in all_disc)]
    out = {"id": item["id"], "role": item["role"], "question": item["question"],
           "first_search": on_term[0] if on_term else None, "searches_on_term": on_term,
           "n_searches": len(searches), "added": [], "picked": [], "picked_all": []}
    if not on_term:
        out.update(label="no_search_for_term", committed=False)
        return out

    def picks(query: str) -> tuple[list[str], list[str]]:
        added = sorted({d for d in all_disc if _has(d, query.lower()) and not _has(d, question)})
        return added, sorted(r for r, ds in readings.items() if any(d in added for d in ds))

    added, picked = picks(on_term[0])
    per_search = [picks(q)[1] for q in on_term]
    union = sorted({r for p in per_search for r in p})
    out.update(added=added, picked=picked, picked_all=union,
               label="narrowed" if picked and len(picked) < len(readings) else "faithful",
               committed=all(p for p in per_search) and len(union) == 1)
    return out


def analyze(run: str) -> list[dict]:
    items = {i["id"]: i for i in load_items(["ambiguity"])}
    rows = []
    for line in (RUNS / run / "responses.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["id"] in items and not r.get("error"):
            rows.append(classify(items[r["id"]], r))
    judgments = RUNS / run / "judgments.jsonl"
    if judgments.exists():
        verdicts = {j["id"]: j["verdict"] for j in map(json.loads, judgments.read_text().splitlines())}
        for row in rows:
            row["verdict"] = verdicts.get(row["id"])
    return sorted(rows, key=lambda r: r["id"])


def summarize(run: str, rows: list[dict]) -> str:
    lines = [f"## {run}", ""]
    for role in ("trap", "control"):
        rs = [r for r in rows if r["role"] == role]
        c = Counter(r["label"] for r in rs)
        lines.append(f"- {role}s: committed {sum(r['committed'] for r in rs)}/{len(rs)}, first search narrowed "
                     f"{c['narrowed']}/{len(rs)}, no search for term {c['no_search_for_term']}; "
                     f"mean searches {sum(r['n_searches'] for r in rs) / len(rs):.1f}")
        if role == "trap" and all(r.get("verdict") for r in rs):
            for name, sub in (("committed", [r for r in rs if r["committed"]]), ("not committed", [r for r in rs if not r["committed"]])):
                if sub:
                    lines.append(f"  - trap pass rate when {name}: {sum(r['verdict'] == 'pass' for r in sub)}/{len(sub)}")
    lines += ["", "| id | committed | first narrowed | searches on term -> readings picked | verdict |", "|---|---|---|---|---|"]
    for r in rows:
        if r["committed"] or r["label"] != "faithful" or r.get("verdict") not in (None, "pass"):
            lines.append(f"| {r['id']} | {'yes' if r['committed'] else ''} | {'yes' if r['label'] == 'narrowed' else r['label'].replace('faithful', '')} | "
                         f"{' / '.join(r['searches_on_term'])} -> {'; '.join(r['picked_all'])} | {r.get('verdict', '')} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", help="also write per-item rows to this JSONL file")
    args = ap.parse_args()
    for run in args.runs:
        rows = analyze(run)
        print(summarize(run, rows), "\n")
        if args.out:
            with open(args.out, "a") as f:
                for r in rows:
                    f.write(json.dumps({"run": run, **r}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
