#!/usr/bin/env python3
"""Standalone arXiv scraper: one day of new papers, archived to disk.

    python scrapers/arxiv_scraper.py                    # yesterday, UTC
    python scrapers/arxiv_scraper.py --date 2026-08-19
    python scrapers/arxiv_scraper.py --categories cs.MA cs.AI --no-store

The daily feed calls the same function, so the archive is a by-product of the
run. Every paper the filter saw is kept, not just the ten that were emailed.
You can re-rank an old day with different anchors, or a different embedding
model, without fetching anything again.

Text is sanitised before it is stored, so anything reading the archive later
reads text a human would recognise. Papers with injection-like patterns are
marked, not dropped.

Size: about 120KB gzipped per day. `--no-store` skips writing.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from arxiv_feed import arxiv, guard                        # noqa: E402
from arxiv_feed.config import load_config                  # noqa: E402

log = logging.getLogger("arxiv_scraper")

RAW_DIR = REPO_ROOT / "data" / "raw"


def archive_path(day: str) -> Path:
    return RAW_DIR / f"{day}.jsonl.gz"


def scrape_day(categories: list[str], day: str | None = None,
               store: bool = True) -> list[arxiv.Paper]:
    """Fetch one UTC day of new papers and (optionally) archive them."""
    day = day or arxiv.default_day()
    papers = arxiv.fetch_new_papers(categories, day)
    log.info("%s: %d papers across %s", day, len(papers), ", ".join(categories))
    if store:
        write_archive(day, papers)
    return papers


def write_archive(day: str, papers: list[arxiv.Paper]) -> Path:
    """One gzipped JSON object per line, sorted by ID so re-runs are stable."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = archive_path(day)

    with gzip.open(path, "wt", encoding="utf-8") as f:
        for p in sorted(papers, key=lambda x: x.arxiv_id):
            title = guard.sanitize(p.title, guard.MAX_TITLE_CHARS)
            abstract = guard.sanitize(p.abstract, guard.MAX_ABSTRACT_CHARS)
            record = {
                **p.to_dict(),
                "title": title,
                "abstract": abstract,
                "sanitized": (title != p.title or abstract != p.abstract),
                "suspicious_markers": guard.suspicious_markers(f"{title}\n{abstract}"),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    log.info("archived %d papers to %s (%.0f KB)", len(papers), path,
             path.stat().st_size / 1024)
    return path


def read_archive(day: str) -> list[dict]:
    """Read a stored day back. Empty list if that day was never scraped."""
    path = archive_path(day)
    if not path.exists():
        return []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    ap.add_argument("--date", help="UTC day, YYYY-MM-DD (default: yesterday)")
    ap.add_argument("--categories", nargs="+",
                    help="override the categories in the config file")
    ap.add_argument("--no-store", action="store_true", help="fetch but do not write")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)-7s %(name)s: %(message)s")

    categories = args.categories or load_config(args.config).categories
    day = args.date or arxiv.default_day()
    papers = scrape_day(categories, day, store=not args.no_store)

    primary = Counter(p.categories[0] for p in papers if p.categories)
    flagged = sum(1 for p in papers
                  if guard.suspicious_markers(f"{p.title}\n{p.abstract}"))

    print(f"\n{day}: {len(papers)} papers")
    for cat, n in primary.most_common():
        print(f"  {cat:<10} {n}")
    if flagged:
        print(f"  {flagged} paper(s) contain injection-like patterns (advisory)")
    if not args.no_store:
        print(f"  archived to {archive_path(day).relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
