#!/usr/bin/env python3
"""Make the question call again for sent papers whose extraction failed.

    python scripts/repair_questions.py --day 2026-09-03 --dry-run
    python scripts/repair_questions.py --day 2026-09-03

A sent paper is marked seen, so a rerun of its day never reaches the question
call for it again. On 2026-09-03 the OpenRouter key ran out partway through
the run, and 2609.03553 was sent with no questions and no canon tags. This
makes that one call again, for exactly those papers.

A paper qualifies only if its day file holds no questions for it AND a
"question extraction failed" problem names it. A paper the model gave no
questions for, on a call that succeeded, is left alone.

The paper's title, abstract and authors come from its own entry in the day
file. That is the text the daily run sent to the same call.

What is touched, once a call succeeds:

    data/YYYY-MM-DD.json   the paper's open_questions and canon; its problem
                           line is removed
    data/latest.json       recopied from the newest day file
    data/canon/candidates.csv   the paper's tag and summary cells only
    data/feed.xml          rebuilt from the day files, as run.py does
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Run as `python scripts/repair_questions.py`, sys.path[0] is scripts/, so the
# package this imports is not on the path yet.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arxiv_feed import canon, feed as feed_mod, questions              # noqa: E402
from arxiv_feed.arxiv import Paper                                     # noqa: E402
from arxiv_feed.config import DATA_DIR, Config, ConfigError, load_config  # noqa: E402
from arxiv_feed.guard import neutralize_cell                           # noqa: E402
from arxiv_feed.llm import ModelClient, ModelError                     # noqa: E402
from scripts.restyle_descriptions import refresh_latest                # noqa: E402

log = logging.getLogger("repair_questions")

# The problem line run.py writes when a question call fails twice.
FAILED_MARK = ": question extraction failed"

# The candidates.csv cells the question call fills. Every other cell on the
# row came from the screen and the judge, which did not fail, and stays.
TAG_COLUMNS = ["tags", "summary", "system_type", "participant_mix", "observability",
               "focus_area", "threat_model", "claim_type"]


def targets(day_data: dict) -> list[dict]:
    """Sent papers with no questions whose extraction failure is on record."""
    failed_ids = {p.split(FAILED_MARK, 1)[0] for p in day_data.get("problems", [])
                  if FAILED_MARK in p}
    return [p for p in day_data.get("papers", [])
            if not p.get("open_questions") and p.get("arxiv_id") in failed_ids]


def _paper(entry: dict) -> Paper:
    names = {f.name for f in fields(Paper)}
    return Paper(**{k: v for k, v in entry.items() if k in names})


def update_candidates_csv(path: Path, entries: list[dict], day: str) -> tuple[int, list[str]]:
    """Fill the tag and summary cells of each repaired paper's row.

    Returns (rows written, failures). Matched on url, the file's dedupe key.
    """
    wanted = {e["url"]: e for e in entries if e.get("url")}
    if not wanted or not path.exists():
        return 0, []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = list(reader)
    if header != canon.CANDIDATE_COLUMNS:
        return 0, [f"{path.name} has a stale header and was left alone"]

    hit = 0
    for row in rows:
        entry = wanted.get(row.get("url", ""))
        if entry is None:
            continue
        tags = entry.get("canon") or {}
        new = canon.to_canon_row(
            paper=_paper(entry), tags=tags,
            summary=tags.get("summary", "") or entry.get("one_sentence", ""),
            similarity=float(entry.get("similarity") or 0.0),
            similarity_rank=entry.get("similarity_rank"),
            nearest_anchor_id=entry.get("nearest_anchor_id", ""),
            significance=entry.get("significance"), novelty=entry.get("novelty"),
            screen_relevant=True, first_seen=day, emailed=True,
        )
        for col in TAG_COLUMNS:
            row[col] = neutralize_cell(new[col])
        hit += 1

    if hit:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            w.writerows(rows)
        tmp.replace(path)
    return hit, []


def repair_day(client, cfg: Config, data_dir: Path, day: str, *,
               dry_run: bool = False, tag_vocab: list[str] | None = None
               ) -> tuple[list[str], list[str]]:
    """Returns (arxiv_ids repaired, failures). Writes nothing on a dry run."""
    path = data_dir / f"{day}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return [], [f"{path.name}: unreadable ({exc})"]

    todo = targets(data)
    if not todo:
        print(f"{day}: no sent paper is missing its questions")
        return [], []

    vocab = canon.known_tags() if tag_vocab is None else tag_vocab
    problems: list[str] = []
    done: list[dict] = []
    for entry in todo:
        aid = entry["arxiv_id"]
        try:
            got = questions.extract(client, cfg.profile, _paper(entry), vocab)
        except ModelError as exc:
            problems.append(f"{aid}: question extraction failed again ({exc})")
            continue
        print(f"\n{day} {aid}: {len(got['open_questions'])} question(s)")
        for q in got["open_questions"]:
            print(f"  - {q['question']}")
        if not dry_run:
            entry["open_questions"] = got["open_questions"]
            entry["canon"] = got["canon"]
        done.append(entry)

    repaired = [e["arxiv_id"] for e in done]
    if dry_run or not done:
        return repaired, problems

    data["problems"] = [p for p in data.get("problems", [])
                        if not any(p.startswith(aid + FAILED_MARK) for aid in repaired)]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("%s: %d paper(s) repaired", path.name, len(done))

    _, latest_problems = refresh_latest(data_dir)
    problems.extend(latest_problems)
    rows, csv_problems = update_candidates_csv(cfg.candidates_csv, done, day)
    problems.extend(csv_problems)
    log.info("candidates.csv: %d row(s) filled", rows)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    feed_mod.rebuild(data_dir, cfg.feed_path, cfg.feed_site_url, cfg.feed_url, now,
                     max_entries=cfg.feed_max_entries)
    return repaired, problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    ap.add_argument("--day", required=True, help="the day file to repair, YYYY-MM-DD")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the new questions and write nothing")
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

    key = Config.openrouter_key()
    if not key:
        print("OPENROUTER_API_KEY is not set. Nothing was written.", file=sys.stderr)
        return 2

    # The same model and effort the daily run uses for this call.
    client = ModelClient(cfg.model, effort=cfg.effort, api_key=key)
    repaired, problems = repair_day(client, cfg, DATA_DIR, args.day, dry_run=args.dry_run)

    print(f"\n{args.day}: {len(repaired)} paper(s) "
          + ("would be repaired (dry run, nothing written)" if args.dry_run else "repaired"))
    for problem in problems:
        print(f"  problem: {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
