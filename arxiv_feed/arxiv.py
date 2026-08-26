"""arXiv API client: one day of new papers, and metadata for anchor IDs.

Uses the public Atom API. No key needed. arXiv asks for three seconds between
requests, and `_MIN_INTERVAL` enforces that.

That spacing is not the only limit in practice. This runs on GitHub Actions,
whose runner IPs are shared across many unrelated workflows; arXiv rate
limited two separate runs within the same hour (2026-08-26), each time on the
very first request of a fresh query, well after the last request and with
_MIN_INTERVAL respected throughout. A 429 there is evidence of load on the
shared IP, not of this client misbehaving, so it gets its own longer,
exponential backoff -- see `_retry_wait`.
"""

from __future__ import annotations

import logging
import random
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import requests

log = logging.getLogger(__name__)

API = "https://export.arxiv.org/api/query"
_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

# Identifies this tool to arXiv, as their API etiquette asks -- no contact
# address: that would send a personal detail to a third-party service for no
# functional benefit here.
_USER_AGENT = "las-new-papers/1.0 (+https://github.com/Gigascale-Labs/las-new-papers)"

_MIN_INTERVAL = 3.0          # arXiv's requested politeness delay, seconds
_PAGE_SIZE = 100
_MAX_PAGES = 20              # 2,000 papers/day ceiling; far above the ~600 expected

# If arXiv does not answer: five attempts total, so a rate limit's backoff
# (see _retry_wait) has room to actually grow before giving up.
_ATTEMPTS = 5
_RETRY_WAIT = 60.0                # a non-429 error: same fixed wait as before
_RATE_LIMIT_BASE_WAIT = 30.0      # a 429 with no Retry-After: 30s, 60s, 120s, 240s
_RATE_LIMIT_MAX_WAIT = 300.0      # ...capped at 5 minutes, honouring Retry-After up to here too

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


def _retry_wait(exc: Exception, attempt: int) -> float:
    """How long to sleep before the next attempt.

    A 429 gets the server's own Retry-After if it sent one; otherwise an
    exponential backoff that grows well past the default fixed wait, since a
    rate limit on a shared IP can take minutes to clear, not seconds. Every
    other error (a timeout, a dropped connection) keeps the original,
    shorter fixed wait -- there is no evidence those need more.
    """
    response = getattr(exc, "response", None)
    if response is not None and response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return min(float(retry_after), _RATE_LIMIT_MAX_WAIT)
            except ValueError:
                pass
        base = _RATE_LIMIT_BASE_WAIT * (2 ** (attempt - 1))
        return min(base * random.uniform(0.85, 1.15), _RATE_LIMIT_MAX_WAIT)
    return _RETRY_WAIT


def _get(params: dict) -> str:
    """One API call, with the retry rule and the politeness delay."""
    global _last_request
    last_error: Exception | None = None

    for attempt in range(1, _ATTEMPTS + 1):
        wait = _MIN_INTERVAL - (time.monotonic() - _last_request)
        if wait > 0:
            time.sleep(wait)
        try:
            resp = requests.get(API, params=params, timeout=60,
                                headers={"User-Agent": _USER_AGENT})
            _last_request = time.monotonic()
            resp.raise_for_status()
            return resp.text
        except Exception as exc:                      # network, HTTP, timeout alike
            _last_request = time.monotonic()
            last_error = exc
            log.warning("arXiv attempt %d/%d failed: %s", attempt, _ATTEMPTS, exc)
            if attempt < _ATTEMPTS:
                time.sleep(_retry_wait(exc, attempt))

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
