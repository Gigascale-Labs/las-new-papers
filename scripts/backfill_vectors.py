#!/usr/bin/env python3
"""Writes the kept-paper vector file for days that ran before it existed.

    python scripts/backfill_vectors.py --dry-run
    python scripts/backfill_vectors.py
    python scripts/backfill_vectors.py --day 2026-08-26 --force

arxiv_feed/vectors.py writes data/embeddings/YYYY-MM-DD.json from the run that
produced the day. Days published before that change have no such file, and
re-running them is not an option: seen.json holds their papers, so a re-run
screens nothing and rebuilds the day file from a merge.

Nothing needs re-fetching or re-judging. The day file already carries the title
and abstract of every kept paper, which is exactly Paper.embed_text. Encoding
those with the model config.yaml names reproduces the vector the original run
computed and discarded.

Measured on the first backfill: 52 papers over 7 days. Rebuilding the anchors
and recomputing each vector's nearest anchor reproduced the published
similarity for 49 of them (median absolute error 3e-5, mean 3.8e-4, max
9.9e-3). The 3 misses each name a nearest_anchor_id absent from the current
anchor set, so the anchor list moved rather than the vector.

Two limits:

- **One model per file.** The file records the model that wrote it. This skips
  a day whose file names a different model unless --force is given. Mixing two
  embedding spaces in one projection fails silently rather than loudly.
- **Kept papers only**, matching vectors.py. Nothing displays the screened pool
  and data/raw/ already holds it.

Existing files stay untouched without --force. The write is additive either
way, so a backfill cannot subtract a vector a run published.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arxiv_feed import vectors as vectors_mod          # noqa: E402
from arxiv_feed.config import DATA_DIR, load_config     # noqa: E402
from arxiv_feed.embed import Embedder                  # noqa: E402

log = logging.getLogger("backfill_vectors")


def day_files(day: str | None) -> list[Path]:
    """Returns day archives, oldest first. Skips data/latest.json, which is a copy."""
    if day:
        path = DATA_DIR / f"{day}.json"
        return [path] if path.exists() else []
    return sorted(p for p in DATA_DIR.glob("*.json") if p.stem[:2].isdigit())


def kept_texts(path: Path) -> tuple[str, list[tuple[str, str]]]:
    """Returns (date, [(arxiv_id, embed_text)]) for one day file's kept papers.

    Drops a paper with no abstract. Paper.embed_text is title plus abstract, so
    encoding a title alone would put a vector built from different input into
    the same file.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for paper in payload.get("papers") or []:
        arxiv_id = paper.get("arxiv_id")
        abstract = paper.get("abstract")
        if not arxiv_id or not abstract:
            continue
        rows.append((arxiv_id, f"{paper.get('title', '')}\n\n{abstract}"))
    return payload.get("date", path.stem), rows


def existing_model(path: Path) -> str | None:
    """Returns the model name an embeddings file records, or None."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("model")
    except (json.JSONDecodeError, OSError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day", help="One UTC day (YYYY-MM-DD). Default: every day on file.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--force", action="store_true",
                    help="Rewrite days that already have a file, including one "
                         "written by a different model.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be written and encode nothing.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    paths = day_files(args.day)
    if not paths:
        log.error("no day files found%s", f" for {args.day}" if args.day else "")
        return 1

    todo: list[tuple[Path, str, list[tuple[str, str]]]] = []
    for path in paths:
        date, rows = kept_texts(path)
        out = cfg.embeddings_path(date)
        was = existing_model(out)
        if was is not None and not args.force:
            log.debug("%s: already written by %s; skipping", date, was)
            continue
        if was is not None and was != cfg.embed_model:
            log.warning("%s: written by %s, this run uses %s; --force rewrites it",
                        date, was, cfg.embed_model)
        if not rows:
            log.debug("%s: no kept paper carries an abstract; skipping", date)
            continue
        todo.append((out, date, rows))

    papers = sum(len(rows) for _, _, rows in todo)
    log.info("%d day(s), %d paper(s) to encode with %s", len(todo), papers, cfg.embed_model)
    if args.dry_run or not todo:
        for _, date, rows in todo:
            log.info("would write %s: %d paper(s)", date, len(rows))
        return 0

    embedder = Embedder(cfg.embed_model, device=cfg.embed_device)
    written = 0
    for out, date, rows in todo:
        matrix = embedder.encode([text for _, text in rows])
        count = vectors_mod.write(
            out, date, cfg.embed_model,
            {arxiv_id: matrix[i] for i, (arxiv_id, _) in enumerate(rows)},
        )
        written += count
        log.info("%s: %d vector(s)", date, count)

    log.info("backfilled %d vector(s) across %d day(s)", written, len(todo))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
