"""The LAS canon schema, mirrored for the finalist papers.

The site's canon (`lib/canon-schema.ts` in largeagentsystems.org) is the human
side of this project: 42 papers, hand-tagged. This module keeps two things
aligned with it.

1. `data/ground-truth/` holds a frozen copy of that canon. It is the evaluation
   set for the retrieval filter (see the leave-one-out test) and the source of
   the anchor list. Frozen on purpose -- an evaluation set that tracks upstream
   silently invalidates every earlier measurement.

2. Every finalist paper is tagged against the same six dimensions and appended
   to `data/canon/finalists.csv`, which has the canon's exact column order. That
   file is a *proposal*, produced by a model from an abstract. Nothing here
   writes to Airtable or to the site's canon; a human still decides what is
   admitted. Keeping the schema identical is what makes that decision a copy
   rather than a re-tagging job.

Any change to the choice lists in `lib/canon-schema.ts` has to be mirrored here
by hand -- there is no import path between a TypeScript file in one repo and a
Python file in another.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# --- mirrored from lib/canon-schema.ts (largeagentsystems.org) ---------------

ITEM_TYPES = [
    "bookSection", "conferencePaper", "dataset", "journalArticle",
    "preprint", "report", "webpage",
]
SYSTEM_TYPES = ["production economy", "social network", "labour market", "financial system"]
PARTICIPANT_MIXES = ["pure-AI", "mixed human+AI"]
OBSERVABILITY_LEVELS = ["aggregates observable", "interactions observable", "agents observable"]
FOCUS_AREAS = ["Monitoring", "Steering", "Simulation", "Redesign", "Design"]
THREAT_MODELS = [
    "Gradual Disempowerment", "Systemic Instability", "Inequality",
    "Collective Superintelligence", "Partially Observable Systems",
    "Power Concentration", "Outdated Models",
]
CLAIM_TYPES = [
    "theoretical/conceptual framework", "empirical study", "survey/taxonomy",
    "proposed method/system", "position/opinion", "threat model articulation",
    "policy/regulatory analysis", "dataset/tool", "live deployment",
]

# Column order is the canon CSV's, exactly.
CANON_COLUMNS = [
    "title", "itemType", "creators", "date", "url", "tags", "summary",
    "system_type", "participant_mix", "observability", "focus_area",
    "threat_model", "claim_type", "tag_confidence", "institutions",
]

DIMENSION_CHOICES: dict[str, list[str]] = {
    "system_type": SYSTEM_TYPES,
    "participant_mix": PARTICIPANT_MIXES,
    "observability": OBSERVABILITY_LEVELS,
    "focus_area": FOCUS_AREAS,
    "threat_model": THREAT_MODELS,
    "claim_type": CLAIM_TYPES,
}

# Extra columns this repo adds beyond the canon's. They are appended after the
# canon columns so the first 15 columns can be lifted straight into the canon.
EXTRA_COLUMNS = ["arxiv_id", "first_seen", "similarity", "nearest_anchor_id",
                 "significance", "novelty", "from_random"]

FINALIST_COLUMNS = CANON_COLUMNS + EXTRA_COLUMNS


def clean_multi(values, allowed: list[str]) -> list[str]:
    """Keep only values on the closed list, in the list's own order.

    A model asked for closed-list values will occasionally return a near-miss
    ("social networks", "Simulation of markets"). Dropping those is right: the
    canon's value is that its lists are closed, and a silently-invented value
    would be worse than a blank cell, which is a legitimate state in the canon
    (several hand-tagged rows have empty dimensions).
    """
    if not values:
        return []
    if isinstance(values, str):
        values = [values]
    chosen = {str(v).strip() for v in values}
    return [a for a in allowed if a in chosen]


def to_canon_row(
    *,
    paper,
    tags: dict,
    summary: str,
    similarity: float,
    nearest_anchor_id: str,
    significance,
    novelty,
    from_random: bool,
    first_seen: str,
) -> dict:
    """One finalist, in canon column order plus this repo's extra columns."""
    date = (paper.published or "")[:4]
    return {
        "title": paper.title,
        # arXiv is a preprint server; a paper that is also a conference paper is
        # re-tagged by hand on admission rather than guessed at here.
        "itemType": "preprint",
        "creators": "; ".join(paper.authors),
        "date": date,
        "url": paper.url,
        "tags": "; ".join(tags.get("tags", []) or []),
        "summary": summary,
        "system_type": "; ".join(clean_multi(tags.get("system_type"), SYSTEM_TYPES)),
        "participant_mix": "; ".join(clean_multi(tags.get("participant_mix"), PARTICIPANT_MIXES)),
        "observability": "; ".join(clean_multi(tags.get("observability"), OBSERVABILITY_LEVELS)),
        "focus_area": "; ".join(clean_multi(tags.get("focus_area"), FOCUS_AREAS)),
        "threat_model": "; ".join(clean_multi(tags.get("threat_model"), THREAT_MODELS)),
        "claim_type": "; ".join(clean_multi(tags.get("claim_type"), CLAIM_TYPES)),
        # Always summary-only: this pipeline reads abstracts, never full text.
        "tag_confidence": "summary-only",
        # The canon's institutions column is open-ended, but an abstract does not
        # carry affiliations -- left blank rather than guessed.
        "institutions": "",
        "arxiv_id": paper.arxiv_id,
        "first_seen": first_seen,
        "similarity": f"{similarity:.4f}",
        "nearest_anchor_id": nearest_anchor_id,
        "significance": significance if significance is not None else "",
        "novelty": novelty if novelty is not None else "",
        "from_random": "yes" if from_random else "",
    }


def append_finalists(path: Path, rows: list[dict]) -> int:
    """Append rows, skipping URLs already present. Returns the number written."""
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)

    existing: set[str] = set()
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            existing = {r.get("url", "") for r in csv.DictReader(f)}

    fresh = [r for r in rows if r["url"] not in existing]
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FINALIST_COLUMNS)
        if write_header:
            w.writeheader()
        for r in fresh:
            w.writerow(r)

    log.info("finalists.csv: %d new row(s), %d skipped as already present",
             len(fresh), len(rows) - len(fresh))
    return len(fresh)


GROUND_TRUTH_CSV = Path(__file__).resolve().parent.parent / "data" / "ground-truth" / "las-canon-frozen.csv"


def known_tags(path: Path | None = None) -> list[str]:
    """The free-text `tags` vocabulary the human canon actually uses.

    Passed to the tagging call as examples. Not a closed list -- the canon's own
    tags column is open -- but without them a model invents a fresh vocabulary
    per paper and the column stops being groupable.
    """
    path = path or GROUND_TRUTH_CSV
    if not path.exists():
        return []
    tags: set[str] = set()
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for t in (row.get("tags") or "").split(";"):
                t = t.strip()
                if t:
                    tags.add(t)
    return sorted(tags)
