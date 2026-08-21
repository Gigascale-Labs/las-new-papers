#!/usr/bin/env python3
"""arXiv open questions feed -- one email a day.

    python main.py                 # yesterday's papers, send the email
    python main.py --dry-run       # same work, write the JSON, send nothing
    python main.py --date 2026-08-19
    python main.py --rebuild-anchors
"""

from __future__ import annotations

import argparse
import logging
import sys

from arxiv_feed.config import ConfigError, load_config
from arxiv_feed.run import run


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--date", help="UTC day to read, YYYY-MM-DD (default: yesterday)")
    ap.add_argument("--dry-run", action="store_true",
                    help="do everything except send the email")
    ap.add_argument("--rebuild-anchors", action="store_true",
                    help="re-embed the anchors even if the cache still matches")
    ap.add_argument("--seed", type=int,
                    help="seed for the random explore slice (for reproducible runs)")
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

    result = run(cfg, day=args.date, dry_run=args.dry_run,
                 seed=args.seed, rebuild_anchors=args.rebuild_anchors)

    n_q = sum(len(p["open_questions"]) for p in result["papers"])
    print(f"{result['date']}: {result['counts']['fetched']} fetched, "
          f"{result['counts']['kept']} kept, {n_q} questions, "
          f"email {'skipped (dry run)' if args.dry_run else ('sent' if result['email']['sent'] else 'NOT SENT')}")
    for problem in result["problems"]:
        print(f"  problem: {problem}")

    # A run that wrote its JSON but could not send is still a failure. CI must
    # show it, even though the data was saved.
    if not args.dry_run and not result["email"]["sent"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
