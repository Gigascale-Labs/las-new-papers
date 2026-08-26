#!/usr/bin/env python3
"""arXiv open questions feed -- a page and an RSS feed, once a day.

    python main.py                 # yesterday's papers
    python main.py --dry-run       # same work, without marking papers seen
    python main.py --date 2026-08-19
    python main.py --rebuild-anchors

On the default (no --date) path, a recent day with no good retrieval on
record -- missing, unreadable, or fetched == 0 -- is re-run alongside
today's. arXiv does not announce Friday or Saturday submissions until the
following Monday, and an arXiv rate limit can crash a run before it writes
anything at all; both leave a day worth retrying once things clear up. See
arxiv_feed.run.backfill_days.
"""

from __future__ import annotations

import argparse
import logging
import sys

from arxiv_feed import arxiv
from arxiv_feed.config import ConfigError, load_config
from arxiv_feed.run import backfill_days, run


def _report(result: dict) -> bool:
    """Print one day's summary line. Returns whether the feed delivered."""
    n_q = sum(len(p["open_questions"]) for p in result["papers"])
    n_feed = result.get("feed", {}).get("entries", 0)
    c = result["counts"]
    print(f"{result['date']}: {c['fetched']} fetched, "
          f"{c.get('screened', 0)} screened, {c.get('relevant', 0)} relevant, "
          f"{c['kept']} kept, {n_q} questions, feed {n_feed} entries")
    for problem in result["problems"]:
        print(f"  problem: {problem}")

    # A run that wrote its JSON but rebuilt an empty feed is still a
    # failure. CI must show it, even though the data was saved.
    return bool(n_feed)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--date", help="UTC day to read, YYYY-MM-DD (default: yesterday)")
    ap.add_argument("--dry-run", action="store_true",
                    help="do everything except mark papers as seen")
    ap.add_argument("--rebuild-anchors", action="store_true",
                    help="re-embed the anchors even if the cache still matches")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    day = args.date or arxiv.default_day()
    # An explicit --date is a precise, manual request for that one day only;
    # backfill only runs the way the daily cron does, unprompted.
    days = [day] if args.date else backfill_days(cfg, day) + [day]

    delivered_ok = True
    for i, d in enumerate(days):
        result = run(cfg, day=d, dry_run=args.dry_run,
                     rebuild_anchors=args.rebuild_anchors and i == 0)
        delivered_ok = _report(result)

    # Only the last day (the primary, requested one) decides the exit code:
    # a backfilled day that is still empty is not a fault in today's run.
    if not args.dry_run and not delivered_ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
