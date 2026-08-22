"""An Atom feed, built fresh from the day files already on disk.

No password. No account. A feed reader polls a URL; nothing sends mail. This
exists because the spec's one channel (email) needs an SMTP login, and some
readers of this feed do not want to hold one, 2FA app password or not.

The feed and the email describe a run the same way: both call
emailer.render_body_html() for the content, so a paper reads the same in
either place.

Rebuilt in full on every run, from the last MAX_ENTRIES day files. Not
appended to. A rebuild cannot drift from what is on disk, and there is no
feed-specific state to lose.
"""

from __future__ import annotations

import json
import logging
import re
import xml.sax.saxutils as saxutils
from pathlib import Path

from . import emailer

log = logging.getLogger(__name__)

MAX_ENTRIES = 60          # about two months of daily runs

_DAY_FILE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.json$")


def _entry_count(result: dict) -> tuple[int, int]:
    papers = result.get("papers", [])
    n_q = sum(len(p.get("open_questions", [])) for p in papers)
    return len(papers), n_q


def _entry_xml(result: dict, site_url: str) -> str:
    date = result["date"]
    n_papers, n_q = _entry_count(result)
    title = f"arXiv open questions — {date} — {n_papers} papers, {n_q} questions"
    entry_url = f"{site_url}#{date}"
    body_html = emailer.render_body_html(result)
    # Atom content is HTML text inside an XML text node: escape once for XML
    # on top of the HTML escaping render_body_html already applied for HTML.
    # That double layer is what stops a paper's own text from closing the
    # <content> element early.
    content = saxutils.escape(body_html)

    return (
        "<entry>"
        f"<title>{saxutils.escape(title)}</title>"
        f'<id>{saxutils.escape(entry_url)}</id>'
        f'<link rel="alternate" href="{saxutils.escape(entry_url)}"/>'
        f"<updated>{saxutils.escape(result['generated_at'])}</updated>"
        f'<content type="html">{content}</content>'
        "</entry>"
    )


def build_feed(results: list[dict], site_url: str, self_url: str, now: str) -> str:
    """An Atom 1.0 document from the given day results, newest first.

    `results` should already be sorted newest first and capped by the caller
    (see rebuild()); this function does not re-sort or re-cap, so tests can
    hand it an exact list.

    `now` is an RFC 3339 timestamp, used as the feed-level <updated> only when
    `results` is empty (no day file exists yet). The caller supplies it rather
    than this module calling a clock, so a build is reproducible from its
    arguments alone.
    """
    updated = results[0]["generated_at"] if results else now
    entries = "".join(_entry_xml(r, site_url) for r in results)

    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<feed xmlns="http://www.w3.org/2005/Atom">'
        "<title>arXiv open questions</title>"
        f'<id>{saxutils.escape(site_url)}</id>'
        f'<link rel="self" href="{saxutils.escape(self_url)}"/>'
        f'<link rel="alternate" href="{saxutils.escape(site_url)}"/>'
        f"<updated>{saxutils.escape(updated)}</updated>"
        f"{entries}"
        "</feed>"
    )


def _read_day_files(data_dir: Path, max_entries: int) -> list[dict]:
    """The last `max_entries` day files in data_dir, newest first.

    Reads by filename, not by mtime: a backfilled or re-run day should sort by
    the date in its name, not by when it happened to be written.
    """
    days = []
    for path in data_dir.glob("*.json"):
        m = _DAY_FILE.match(path.name)
        if m:
            days.append((m.group(1), path))
    days.sort(key=lambda t: t[0], reverse=True)

    results = []
    for _, path in days[:max_entries]:
        try:
            results.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError) as exc:
            # One unreadable day must not block the feed for every other day.
            log.warning("feed: skipping unreadable day file %s (%s)", path, exc)
    return results


def rebuild(data_dir: Path, feed_path: Path, site_url: str, self_url: str, now: str,
            max_entries: int = MAX_ENTRIES) -> int:
    """Regenerate feed_path from every day file under data_dir.

    Returns the number of entries written.
    """
    results = _read_day_files(data_dir, max_entries)
    xml_text = build_feed(results, site_url, self_url, now)
    feed_path.parent.mkdir(parents=True, exist_ok=True)
    feed_path.write_text(xml_text, encoding="utf-8")
    log.info("feed: wrote %d entries to %s", len(results), feed_path)
    return len(results)
