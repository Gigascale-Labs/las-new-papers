"""The LAS canon schema, mirrored here.

Two things use it.

`data/ground-truth/` holds a frozen copy of the site's canon. It is the test
set for the filter and the source of the anchor list. It is frozen because a
test set that follows the canon makes older results incomparable.

Every shortlisted paper is appended to `data/canon/candidates.csv`, in the
canon's column order. That is one file, not two: the ten papers that were
emailed carry their dimension tags and an `emailed` mark, and the rest carry
their similarity and scores with the dimensions left blank. Blank dimensions
are normal in the canon.

The file is a proposal. Nothing here writes to Airtable, and a human decides
what is admitted. The shared column order makes admitting a paper a copy rather
than a re-tagging job.

The choice lists below are copied by hand from `lib/canon-schema.ts` in the
site repository. A change there needs a change here.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from .guard import neutralize_cell

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

# Extra columns this repo adds beyond the canon's. They come after the canon
# columns, so the first 15 columns lift straight into the canon.
EXTRA_COLUMNS = ["arxiv_id", "first_seen", "similarity", "similarity_rank",
                 "nearest_anchor_id", "from_random", "significance", "novelty",
                 "emailed"]

CANDIDATE_COLUMNS = CANON_COLUMNS + EXTRA_COLUMNS


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
    similarity_rank: int,
    nearest_anchor_id: str,
    significance,
    novelty,
    from_random: bool,
    first_seen: str,
    emailed: bool = False,
) -> dict:
    """One shortlisted paper, in canon column order plus the extra columns.

    `tags` is empty for a paper that was shortlisted but not emailed. It never
    got the question-extraction call, so it has no dimension values, and the
    dimension columns stay blank.
    """
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
        "similarity_rank": similarity_rank,
        "nearest_anchor_id": nearest_anchor_id,
        "from_random": "yes" if from_random else "",
        "significance": significance if significance is not None else "",
        "novelty": novelty if novelty is not None else "",
        "emailed": "yes" if emailed else "",
    }


def append_candidates(path: Path, rows: list[dict]) -> int:
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
        w = csv.DictWriter(f, fieldnames=CANDIDATE_COLUMNS)
        if write_header:
            w.writeheader()
        for r in fresh:
            # Model-written text from an untrusted abstract lands in a file
            # people open in Excel. A cell starting = + - @ executes there.
            w.writerow({k: neutralize_cell(v) for k, v in r.items()})

    log.info("candidates.csv: %d new row(s), %d already present",
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
