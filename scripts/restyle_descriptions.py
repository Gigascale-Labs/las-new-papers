#!/usr/bin/env python3
"""Rewrite the summaries already in the archive to the current house style.

    python scripts/restyle_descriptions.py --dry-run
    python scripts/restyle_descriptions.py --day 2026-08-25
    python scripts/restyle_descriptions.py --limit 5 --model anthropic/claude-sonnet-5

The style block in arxiv_feed/style.py changed after these summaries were
written, so every `one_sentence` in `data/*.json` predates the spec it is
supposed to follow. This re-runs them through it.

A restyle, not a re-read. The model gets the published summary and the paper's
title, and nothing else. It never sees the abstract: given the abstract it
would re-derive the summary rather than rewrite it, and the new text could
then make claims the old one never made, with no way to tell which. The
prompt forbids adding a claim, and this script cannot check that -- so the
input is kept to the text being rewritten.

What is touched, once the rewrites are in:

    data/YYYY-MM-DD.json   the `papers` list, `one_sentence` only
    data/latest.json       recopied from the newest day file
    data/canon/candidates.csv   the `summary` cell, on rows whose url matches
                                AND whose cell still holds the old text
    data/feed.xml          rebuilt from the day files, as run.py does

The extra condition on the CSV is not caution for its own sake. A paper that
reached the question call has the canon summary in that cell, which is longer
text from a different call; overwriting it with a restyled `one_sentence`
would destroy it. The `screened` rows inside a day file are left alone too:
they are the record of what the judge wrote on the day, and nothing renders
them.

One paper failing never stops the rest. Failures are collected and reported,
and a paper the model returned nothing for keeps its original text.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Run as `python scripts/restyle_descriptions.py`, sys.path[0] is scripts/,
# so the package this imports is not on the path yet.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arxiv_feed import canon, feed as feed_mod                        # noqa: E402
from arxiv_feed.config import (DATA_DIR, Config, ConfigError,         # noqa: E402
                               load_config)
from arxiv_feed.guard import (DATA_NOT_INSTRUCTIONS, MAX_TITLE_CHARS,  # noqa: E402
                              fence, neutralize_cell, sanitize)
from arxiv_feed.llm import ModelClient, ModelError                    # noqa: E402
from arxiv_feed.style import PLAIN_ENGLISH                            # noqa: E402

log = logging.getLogger("restyle")

BATCH_SIZE = 8            # papers per call; the archive is 36 papers, not 360
MAX_SUMMARY_CHARS = 2000  # a summary is three sentences; anything longer is not one

_DAY_FILE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.json$")

SYSTEM = """\
You rewrite published one-paragraph summaries of arXiv papers into a fixed
house style.

This is a restyle, not a re-read. For each paper you get the summary text that
is already published and the paper's title. You do not have the paper and you
do not have its abstract.

Keep every claim the given summary makes. Add none. Do not add a number, a
sample size, a method, a dataset, a result, a limitation or a qualifier that is
not already in the text you are given. Do not resolve an ambiguity by deciding
what the paper probably meant. If the style below asks for something the given
text does not contain -- a number, what the paper measured rather than assumed,
what it did not check -- leave that out and rewrite the rest. A restyle that
obeys every rule but one is right. An invented fact is wrong, and it will be
published as though the paper said it.

Use the title only to keep the subject straight, for example to work out what
"it" refers to. The title is not a source of new claims either.

Do not make the summary longer than the one you are given. At most three
sentences, each under 20 words.

Never join two sentences into one. Splitting a long sentence is right;
merging two short ones is wrong, whatever it does to the word count.
Measured on the archive this rewrites: 22 sentences already run over 20
words, the longest 43. Those are the ones to split. If the text you are
given reads "...tiers.Finds improved..." that is a missing space after a
full stop, not one sentence -- restore the space and keep both sentences.

Describe the work. Do not assess it. The summary must not say whether the paper
is relevant, novel, significant, rigorous or implementable, and must not
explain why the paper was selected.

If a summary already obeys the style, return it unchanged. Rewording for the
sake of returning something different is a loss, not a gain.

Return one summary for every paper you are given, keyed by its exact arxiv_id.
The arxiv_id is the value stated on the line directly above each fenced
document, in the form "arxiv_id: <value>". It is not the fence's nonce
attribute; that nonce is a security marker, not a paper identifier, and must
never appear in your output.

""" + PLAIN_ENGLISH + "\n" + DATA_NOT_INSTRUCTIONS

SCHEMA = {
    "type": "object",
    "properties": {
        "summaries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string"},
                    "one_sentence": {"type": "string"},
                },
                "required": ["arxiv_id", "one_sentence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["summaries"],
    "additionalProperties": False,
}


@dataclass
class Item:
    """One paper's summary, and where it lives."""

    path: Path
    day: str
    arxiv_id: str
    url: str
    title: str
    old: str
    new: str = ""


@dataclass
class Report:
    """What a run did. Printed by main(); asserted on by the tests."""

    found: int = 0
    rewritten: int = 0
    unchanged: int = 0
    days_written: list[str] = field(default_factory=list)
    csv_rows: int = 0
    csv_other_text: int = 0
    feed_entries: int = 0
    problems: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# read
# --------------------------------------------------------------------------

def collect(data_dir: Path, day: str | None = None,
            limit: int | None = None) -> tuple[list[Item], list[str]]:
    """Every paper summary in the archive, oldest day first.

    An unreadable day file is reported and skipped, not raised: the other five
    days are still worth rewriting.
    """
    problems: list[str] = []
    paths = []
    for path in data_dir.glob("*.json"):
        m = _DAY_FILE.match(path.name)
        if m and (day is None or m.group(1) == day):
            paths.append((m.group(1), path))
    paths.sort(key=lambda t: t[0])

    if day and not paths:
        problems.append(f"no day file for {day} in {data_dir}")

    items: list[Item] = []
    for d, path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            problems.append(f"{path.name}: unreadable, skipped ({exc})")
            continue
        for paper in data.get("papers", []):
            text = str(paper.get("one_sentence", "") or "").strip()
            aid = str(paper.get("arxiv_id", "") or "").strip()
            # No id, no rewrite: the id is the key the model answers by, and a
            # blank one would match a blank one in a bad answer.
            if not text or not aid:
                continue
            items.append(
                Item(
                    path=path,
                    day=d,
                    arxiv_id=aid,
                    url=str(paper.get("url", "") or ""),
                    title=str(paper.get("title", "") or ""),
                    old=text,
                )
            )

    if limit is not None:
        items = items[:limit]
    return items, problems


# --------------------------------------------------------------------------
# rewrite
# --------------------------------------------------------------------------

def _render(items: list[Item]) -> str:
    """One fenced block per paper. The id sits outside, as in judge.py."""
    blocks = []
    for it in items:
        title = sanitize(it.title, MAX_TITLE_CHARS)
        # Repaired before the model sees it, so a run-on reads as the two
        # sentences it is, not as one the model may try to smooth into a
        # single long one.
        summary = repair_spacing(sanitize(it.old, MAX_SUMMARY_CHARS))
        body, _ = fence(f"title: {title}\nsummary: {summary}")
        blocks.append(f"arxiv_id: {it.arxiv_id}\n{body}")
    return "\n\n".join(blocks)


def _batches(items: list[Item], size: int) -> list[list[Item]]:
    """Batches of up to `size`, never two items with one arxiv_id in a batch.

    The model keys its answer by arxiv_id, so two rows sharing an id in one
    batch could not be told apart. The same paper in two day files is rare but
    possible; it goes in two batches.
    """
    size = max(1, size)
    batches: list[list[Item]] = []
    current: list[Item] = []
    seen: set[str] = set()
    for it in items:
        if len(current) >= size or it.arxiv_id in seen:
            batches.append(current)
            current, seen = [], set()
        current.append(it)
        seen.add(it.arxiv_id)
    if current:
        batches.append(current)
    return batches


_RUN_ON = re.compile(r"([.!?])([A-Z])")


def repair_spacing(text: str) -> str:
    """Restore the space a full stop lost: "tiers.Finds" -> "tiers. Finds".

    Measured on the archive at the time of writing: 9 of 36 summaries (25%)
    run two sentences together this way. It is a formatting defect from the
    call that wrote them, not a style choice, and it has one correct repair --
    so it is done here, deterministically, rather than spent on a model call
    that might merge the two sentences instead of separating them.

    Applied to what the model returns as well as to what it is given: a model
    handed clean input can still hand back a run-on.
    """
    return _RUN_ON.sub(r"\1 \2", text)


def restyle_batch(client, items: list[Item], label: str) -> tuple[dict[str, str], list[str]]:
    """Rewrite one batch. Returns (new text by arxiv_id, failures).

    A row for an id that was not in the batch is dropped: the model invented
    it, and writing it would put one paper's summary on another. A paper the
    model skipped is reported, never defaulted -- the caller leaves its text
    alone.
    """
    if not items:
        return {}, []

    user = f"PAPERS ({len(items)})\n\n{_render(items)}"
    try:
        data = client.structured(
            system=SYSTEM,
            user=user,
            schema=SCHEMA,
            max_tokens=8000,
            label=label,
        )
    except ModelError as exc:
        msg = (f"{label} failed ({len(items)} paper(s) keep their original "
               f"text): {exc}")
        log.error(msg)
        return {}, [msg]

    wanted = {it.arxiv_id for it in items}
    out: dict[str, str] = {}
    for row in data.get("summaries", []) or []:
        aid = str(row.get("arxiv_id", "")).strip()
        text = repair_spacing(str(row.get("one_sentence", "") or "").strip())
        if aid in wanted and text:
            out[aid] = text

    problems = []
    missing = sorted(wanted - set(out))
    if missing:
        msg = (f"{label} returned no summary for {len(missing)} paper(s), "
               f"which keep their original text: {', '.join(missing)}")
        log.warning(msg)
        problems.append(msg)
    return out, problems


def restyle(client, items: list[Item], batch_size: int = BATCH_SIZE) -> list[str]:
    """Fill in `new` on every item the model rewrote. Returns failures.

    A batch that fails costs its own papers, not the run.
    """
    problems: list[str] = []
    batches = _batches(items, batch_size)
    log.info("restyling %d summar(ies) in %d batch(es) of up to %d",
             len(items), len(batches), batch_size)

    for i, batch in enumerate(batches, start=1):
        label = f"restyle {i}/{len(batches)}"
        rewritten, batch_problems = restyle_batch(client, batch, label)
        problems.extend(batch_problems)
        for it in batch:
            it.new = rewritten.get(it.arxiv_id, "")
    return problems


# --------------------------------------------------------------------------
# write
# --------------------------------------------------------------------------

def changed(items: list[Item]) -> list[Item]:
    """Items with new text that differs from the old. Nothing else is written."""
    return [it for it in items if it.new and it.new != it.old]


def write_days(items: list[Item]) -> tuple[list[Path], list[str]]:
    """Write `one_sentence` back into each day file. Returns (paths, failures).

    A day file whose papers all failed is never opened for writing. Read fresh
    and matched on arxiv_id rather than on the index collect() saw, so a file
    edited in between is not overwritten from stale state.
    """
    problems: list[str] = []
    written: list[Path] = []

    by_path: dict[Path, list[Item]] = {}
    for it in changed(items):
        by_path.setdefault(it.path, []).append(it)

    for path, group in sorted(by_path.items(), key=lambda kv: kv[0].name):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            problems.append(f"{path.name}: unreadable at write time, left alone ({exc})")
            continue

        new_text = {it.arxiv_id: it.new for it in group}
        hits = 0
        for paper in data.get("papers", []):
            aid = str(paper.get("arxiv_id", "")).strip()
            if aid in new_text:
                paper["one_sentence"] = new_text[aid]
                hits += 1
        if not hits:
            problems.append(f"{path.name}: no matching paper left to write, left alone")
            continue

        try:
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        except OSError as exc:
            problems.append(f"{path.name}: could not be written ({exc})")
            continue
        written.append(path)
        log.info("%s: %d summar(ies) rewritten", path.name, hits)

    return written, problems


def refresh_latest(data_dir: Path) -> tuple[Path | None, list[str]]:
    """Recopy latest.json from the newest day file, byte for byte.

    A copy, not a symlink, for the same reason run.write_output makes one: the
    site reads this over raw.githubusercontent.com, which serves a symlink's
    link text.
    """
    days = [(m.group(1), p) for p in data_dir.glob("*.json")
            if (m := _DAY_FILE.match(p.name))]
    if not days:
        return None, ["no day file to copy into latest.json"]
    days.sort(key=lambda t: t[0])
    newest = days[-1][1]
    try:
        (data_dir / "latest.json").write_bytes(newest.read_bytes())
    except OSError as exc:
        return None, [f"latest.json could not be written ({exc})"]
    log.info("latest.json copied from %s", newest.name)
    return newest, []


def update_candidates_csv(path: Path, items: list[Item]) -> tuple[int, int, list[str]]:
    """Restyle the `summary` cell of matching candidate rows.

    Returns (rows rewritten, rows matched on url but holding other text,
    failures). The second number is normal, not a fault, and is reported so
    that a run which changes no CSV row says so rather than looking silent.

    Two conditions, both required. The url must match -- it is this file's
    dedupe key, see canon.append_candidates. And the cell must still hold the
    exact text being replaced: on a paper that reached the question call, that
    cell holds the canon summary, which is longer text from another call and
    is not this script's to overwrite.

    The whole file is rewritten in the canon's column order, from its own
    header. A header that is not the expected one means the file predates a
    schema change; that is a migration for a person, so this refuses rather
    than shifting every value one column left.
    """
    wanted = {it.url: it for it in items if it.url}
    if not wanted or not path.exists():
        return 0, 0, []

    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            rows = list(reader)
    except OSError as exc:
        return 0, 0, [f"{path.name}: unreadable, left alone ({exc})"]

    if header != canon.CANDIDATE_COLUMNS:
        return 0, 0, [
            f"{path.name} has a stale header and was left alone. Missing: "
            f"{sorted(set(canon.CANDIDATE_COLUMNS) - set(header))}; unexpected: "
            f"{sorted(set(header) - set(canon.CANDIDATE_COLUMNS))}."
        ]

    hit = other = 0
    for row in rows:
        it = wanted.get(row.get("url", ""))
        if it is None:
            continue
        current = (row.get("summary") or "").strip()
        # neutralize_cell may have put a leading apostrophe on the stored copy.
        if current not in (it.old, neutralize_cell(it.old)):
            other += 1
            continue
        row["summary"] = neutralize_cell(it.new)
        hit += 1

    if not hit:
        log.info("candidates.csv: unchanged; %d matching row(s) hold other text", other)
        return 0, other, []

    try:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            w.writerows(rows)
    except OSError as exc:
        return 0, other, [f"{path.name}: could not be written ({exc})"]

    log.info("candidates.csv: %d summar(ies) rewritten, %d matching row(s) hold "
             "other text", hit, other)
    return hit, other, []


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------

def _print_pairs(items: list[Item]) -> None:
    for it in items:
        print(f"\n{it.day} {it.arxiv_id}")
        print(f"  old: {it.old}")
        print(f"  new: {it.new}")


def restyle_archive(client, cfg: Config, data_dir: Path, *, day: str | None = None,
                    limit: int | None = None, dry_run: bool = False,
                    batch_size: int = BATCH_SIZE) -> Report:
    """Collect, rewrite, and -- unless dry_run -- write every file that follows."""
    report = Report()
    items, problems = collect(data_dir, day=day, limit=limit)
    report.problems.extend(problems)
    report.found = len(items)
    if not items:
        return report

    report.problems.extend(restyle(client, items, batch_size=batch_size))
    edited = changed(items)
    report.rewritten = len(edited)
    report.unchanged = sum(1 for it in items if it.new and it.new == it.old)
    _print_pairs(edited)

    if dry_run:
        print(f"\ndry run: {len(edited)} summar(ies) would change, nothing written")
        return report
    if not edited:
        return report

    written, problems = write_days(items)
    report.days_written = [p.name for p in written]
    report.problems.extend(problems)

    # Only the papers actually on disk now may propagate to the other files.
    on_disk = set(written)
    landed = [it for it in edited if it.path in on_disk]
    if not landed:
        return report

    _, problems = refresh_latest(data_dir)
    report.problems.extend(problems)

    rows, other, problems = update_candidates_csv(cfg.candidates_csv, landed)
    report.csv_rows = rows
    report.csv_other_text = other
    report.problems.extend(problems)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        report.feed_entries = feed_mod.rebuild(
            data_dir, cfg.feed_path, cfg.feed_site_url, cfg.feed_url, now,
            max_entries=cfg.feed_max_entries,
        )
    except OSError as exc:
        report.problems.append(f"feed.xml could not be rebuilt ({exc})")

    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    ap.add_argument("--dry-run", action="store_true",
                    help="print old and new for every summary, write nothing")
    ap.add_argument("--limit", type=int, help="restyle at most N summaries")
    ap.add_argument("--day", help="restyle one day file only, YYYY-MM-DD")
    ap.add_argument("--model",
                    help="OpenRouter model id (default: judge_model from config.yaml, "
                         "the model that wrote these summaries)")
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
        # Checked here rather than at the call: every rewrite needs the model,
        # including --dry-run, so there is nothing this run could do without
        # it. Nothing has been written at this point and nothing will be.
        print("OPENROUTER_API_KEY is not set, so no summary can be rewritten. "
              "Export it and run again. Nothing was written.", file=sys.stderr)
        return 2

    model = args.model or cfg.judge_model
    client = ModelClient(model, effort=cfg.effort, api_key=key)
    print(f"restyling with {model}"
          + (f", day {args.day}" if args.day else "")
          + (f", limit {args.limit}" if args.limit else "")
          + (" (dry run)" if args.dry_run else ""))

    report = restyle_archive(client, cfg, DATA_DIR, day=args.day, limit=args.limit,
                             dry_run=args.dry_run)

    print(f"\n{report.found} summar(ies) read, {report.rewritten} rewritten, "
          f"{report.unchanged} returned unchanged")
    if not args.dry_run:
        print(f"day files written: {', '.join(report.days_written) or 'none'}; "
              f"candidates.csv rows: {report.csv_rows} rewritten, "
              f"{report.csv_other_text} left holding other text; "
              f"feed entries: {report.feed_entries}")
    for problem in report.problems:
        print(f"  problem: {problem}")

    # Non-zero on any failure so a CI run shows it, but only after every file
    # that could be written has been written.
    return 1 if report.problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
