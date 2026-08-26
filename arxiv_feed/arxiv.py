"""arXiv API client: one day of new papers, and metadata for anchor IDs.

Uses the public Atom API. No key needed. arXiv asks for three seconds between
requests, and `_MIN_INTERVAL` enforces that.
"""

from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import requests

log = logging.getLogger(__name__)

API = "https://export.arxiv.org/api/query"
_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

_MIN_INTERVAL = 3.0          # arXiv's requested politeness delay, seconds
_PAGE_SIZE = 100
_MAX_PAGES = 20              # 2,000 papers/day ceiling; far above the ~600 expected

# If arXiv does not answer: wait 60 seconds, try three times in total.
_ATTEMPTS = 3
_RETRY_WAIT = 60.0

_last_request = 0.0


class ArxivError(Exception):
    """arXiv did not answer after every attempt was used."""


@dataclass
class Paper:
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    published: str
    updated: str
    categories: list[str] = field(default_factory=list)

    @property
    def url(self) -> str:
        return f"https://arxiv.org/abs/{self.arxiv_id}"

    @property
    def embed_text(self) -> str:
        """Title and abstract, the pair SPECTER-family models are trained on.

        The separator matters: SPECTER2 is trained with title and abstract joined
        by the tokenizer's [SEP], which `sentence-transformers` inserts for a text
        pair. Plain concatenation is close enough and keeps one code path -- see
        README ("Embedding model").
        """
        return f"{self.title}\n\n{self.abstract}"

    def to_dict(self) -> dict:
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "published": self.published,
            "updated": self.updated,
            "categories": self.categories,
            "url": self.url,
        }


def strip_version(arxiv_id: str) -> str:
    """`2502.14143v2` -> `2502.14143`. Versions must not create duplicate entries."""
    return re.sub(r"v\d+$", "", arxiv_id.strip())


# New-style (2502.14143) and pre-2007 style (math/0309136). Every link on the
# page and every URL in the archive is built from this ID, so it is validated
# at the point it enters the system rather than trusted because arXiv sent it.
_VALID_ID = re.compile(r"^(\d{4}\.\d{4,5}|[a-z-]+(\.[A-Z]{2})?/\d{7})$")


def is_valid_id(arxiv_id: str) -> bool:
    return bool(_VALID_ID.match(arxiv_id or ""))


def _get(params: dict) -> str:
    """One API call, with the retry rule and the politeness delay."""
    global _last_request
    last_error: Exception | None = None

    for attempt in range(1, _ATTEMPTS + 1):
        wait = _MIN_INTERVAL - (time.monotonic() - _last_request)
        if wait > 0:
            time.sleep(wait)
        try:
            resp = requests.get(API, params=params, timeout=60)
            _last_request = time.monotonic()
            resp.raise_for_status()
            return resp.text
        except Exception as exc:                      # network, HTTP, timeout alike
            _last_request = time.monotonic()
            last_error = exc
            log.warning("arXiv attempt %d/%d failed: %s", attempt, _ATTEMPTS, exc)
            if attempt < _ATTEMPTS:
                time.sleep(_RETRY_WAIT)

    raise ArxivError(f"arXiv did not answer after {_ATTEMPTS} attempts: {last_error}")


def _parse(xml_text: str) -> list[Paper]:
    root = ET.fromstring(xml_text)
    papers: list[Paper] = []

    for entry in root.findall("atom:entry", _NS):
        raw_id = (entry.findtext("atom:id", "", _NS) or "").rsplit("/abs/", 1)[-1]
        arxiv_id = strip_version(raw_id)
        if not is_valid_id(arxiv_id):
            log.warning("skipping entry with unusable arXiv id: %r", raw_id[:80])
            continue
        title = " ".join((entry.findtext("atom:title", "", _NS) or "").split())
        abstract = " ".join((entry.findtext("atom:summary", "", _NS) or "").split())
        authors = [
            " ".join((a.findtext("atom:name", "", _NS) or "").split())
            for a in entry.findall("atom:author", _NS)
        ]
        cats = [
            c.attrib.get("term", "")
            for c in entry.findall("atom:category", _NS)
            if c.attrib.get("term")
        ]
        papers.append(
            Paper(
                arxiv_id=arxiv_id,
                title=title,
                abstract=abstract,
                authors=[a for a in authors if a],
                published=entry.findtext("atom:published", "", _NS) or "",
                updated=entry.findtext("atom:updated", "", _NS) or "",
                categories=cats,
            )
        )
    return papers


def default_day() -> str:
    """Yesterday, UTC.

    Not today: a run at 06:00 UTC would otherwise see only the few hours of
    "today" that exist yet, and would silently miss the rest of the day forever.
    """
    return (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")


def fetch_new_papers(
    categories: list[str], day: str | None = None, search_queries: list[str] | None = None
) -> list[Paper]:
    """Every paper submitted to `categories`, or matching `search_queries`, on
    `day` (UTC, YYYY-MM-DD).

    `search_queries` is matched against title, abstract and comments (arXiv's
    `all:` field), OR'd in alongside the categories -- so a paper outside every
    configured category is still fetched if its text uses one of the phrases.

    Deduplicated: a paper cross-listed to three of the configured lists, or
    matching two search queries, is one paper, not two or three.
    """
    day = day or default_day()
    stamp = day.replace("-", "")
    cat_clause = " OR ".join(f"cat:{c}" for c in categories)
    scope = f"({cat_clause})"
    if search_queries:
        kw_clause = " OR ".join(f'all:"{q}"' for q in search_queries)
        scope = f"({scope} OR ({kw_clause}))"
    query = f"{scope} AND submittedDate:[{stamp}0000 TO {stamp}2359]"

    seen: dict[str, Paper] = {}
    for page in range(_MAX_PAGES):
        xml_text = _get(
            {
                "search_query": query,
                "start": page * _PAGE_SIZE,
                "max_results": _PAGE_SIZE,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
        )
        batch = _parse(xml_text)
        for p in batch:
            seen.setdefault(p.arxiv_id, p)
        log.info("arXiv page %d: %d entries (%d unique so far)", page + 1, len(batch), len(seen))
        if len(batch) < _PAGE_SIZE:
            break
    else:
        log.warning("hit the %d-page ceiling for %s; some papers were not read",
                    _MAX_PAGES, day)

    return list(seen.values())


def fetch_by_ids(ids: list[str], chunk: int = 50) -> dict[str, Paper]:
    """Metadata for specific arXiv IDs, keyed by version-stripped ID.

    IDs that arXiv does not return (withdrawn, mistyped) are simply absent from
    the result -- the caller decides whether that is fatal.
    """
    out: dict[str, Paper] = {}
    wanted = [strip_version(i) for i in ids]
    for i in range(0, len(wanted), chunk):
        batch = wanted[i : i + chunk]
        xml_text = _get({"id_list": ",".join(batch), "max_results": len(batch)})
        for p in _parse(xml_text):
            out[p.arxiv_id] = p
    return out
