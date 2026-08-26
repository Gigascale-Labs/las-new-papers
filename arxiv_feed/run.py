"""The daily run, in order.

fetch -> drop seen -> embed -> pre-sort to the cap -> guard -> screen (cheap
model, every paper) -> judge (strong model, what passed) -> keep the top 10 ->
questions -> save.

Two model tiers, not one. The cheap screen reads every paper that survives the
pre-sort and answers "is this relevant at all"; the strong judge reads only
what passed and answers "is it good, and is it new". Similarity used to decide
both, and could answer neither -- see docs/ranking-report.md.

One rule shapes the error handling: one bad paper never stops the run. Every
failure below is collected and reported, never raised.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from . import anchors as anchors_mod
from . import arxiv, canon, feed as feed_mod, guard, judge, questions, screen
from .config import DATA_DIR, Config, anchor_count_warning
from .embed import Embedder
from .llm import ModelClient, ModelError
from .seen import SeenStore
from .select import preselect
from scrapers.arxiv_scraper import scrape_day

log = logging.getLogger(__name__)


def backfill_days(cfg: Config, target_day: str, window: int = 5) -> list[str]:
    """Days before `target_day` with no good retrieval on record, oldest first.

    Two known causes make a day come back with nothing:

    - arXiv does not announce Friday or Saturday submissions until the
      following Sunday or Monday (info.arxiv.org/help/availability.html), so
      a run that queries "yesterday" the next morning always gets 0 papers
      for either day -- not because nothing was submitted, but because the
      query landed before arXiv's own backlog was processed.
    - arXiv occasionally rate-limits the query (HTTP 429) past every retry
      _get() already makes; that run crashes before it ever writes a file
      for the day, rather than writing one with `fetched: 0`.

    By the time `target_day` comes around, both are usually resolved: the
    backlog is announced, and a fresh request is a fresh chance to not be
    rate-limited. So a day counts as needing a retry if its file is missing,
    unreadable, or reports `fetched == 0` -- anything short of a file that
    reports a real, nonzero fetch. `unseen == 0` on a nonzero `fetched` is
    left alone: that is a real, already-understood outcome (every paper that
    day had already been sent), not a retrieval miss.

    The default 5-day window is deliberately wider than the two-day weekend
    gap: a single stretch of bad luck (a rate limit landing right after a
    weekend, say) can leave several consecutive days unresolved, and this
    must still reach all of them once things clear up.
    """
    target = date.fromisoformat(target_day)
    candidates = []
    for i in range(window, 0, -1):
        day = (target - timedelta(days=i)).isoformat()
        path = cfg.output_path(day)
        fetched = None
        if path.exists():
            try:
                fetched = json.loads(path.read_text(encoding="utf-8")).get("counts", {}).get("fetched")
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("backfill: %s is unreadable (%s); treating %s as unresolved",
                           path, exc, day)
        if not fetched:
            candidates.append(day)
    return candidates


def run(cfg: Config, day: str | None = None, dry_run: bool = False,
        rebuild_anchors: bool = False) -> dict:
    """Execute one day's run and return the result record that is written to disk."""
    day = day or arxiv.default_day()
    problems: list[str] = []

    warning = anchor_count_warning(cfg)
    if warning:
        log.warning(warning)
        problems.append(warning)

    # 1. anchors (cached unless the list or the model changed)
    embedder = Embedder(cfg.embed_model, device=cfg.embed_device)
    store = anchors_mod.load_or_build(cfg, embedder, force=rebuild_anchors)

    # 2. the day's papers, minus anything already sent.
    #
    # Via the standalone scraper, so the full day is archived to data/raw/ as a
    # by-product: every paper the filter saw, not just the ten that were sent.
    # A past day can then be re-ranked with different anchors without re-fetching.
    papers = scrape_day(cfg.categories, day, search_queries=cfg.search_queries)
    log.info("fetched %d papers for %s", len(papers), day)
    seen = SeenStore(cfg.seen_path)
    unseen = seen.filter_unseen(papers)
    log.info("%d unseen (%d already sent)", len(unseen), len(papers) - len(unseen))

    counts = {
        "fetched": len(papers),
        "unseen": len(unseen),
        "screened": 0,
        "relevant": 0,
        "kept": 0,
        "anchors": len(store.ids),
    }
    result = {
        "date": day,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "categories": cfg.categories,
            "search_queries": cfg.search_queries,
            "screen_n": cfg.screen_n,
            "screen_batch_size": cfg.screen_batch_size,
            "top_n": cfg.top_n,
            "embed_model": cfg.embed_model,
            "screen_model": cfg.screen_model,
            "screen_effort": cfg.screen_effort,
            "judge_model": cfg.judge_model,
            "model": cfg.model,
            "effort": cfg.effort,
            "anchor_count": len(store.ids),
        },
        "counts": counts,
        "papers": [],
        "screened": [],
        "problems": problems,
    }

    if not unseen:
        problems.append(f"no unseen papers found for {day}")
        # Still publish: a day with nothing unseen is a real outcome, and
        # latest.json is the web UI's only signal of the most recent run. Left
        # unwritten, the UI freezes on the last day that had papers -- exactly
        # arXiv's own Friday-to-Sunday gap, every week.
        write_output(cfg, result, day)
        n_entries = feed_mod.rebuild(
            DATA_DIR, cfg.feed_path, cfg.feed_site_url, cfg.feed_url, result["generated_at"],
            max_entries=cfg.feed_max_entries,
        )
        result["feed"] = {"entries": n_entries,
                          "path": str(cfg.feed_path.relative_to(DATA_DIR.parent))}
        return result

    # 3. embed + 4. pre-sort to the screening cap
    #
    # Not a filter any more: on a day under the cap nothing is dropped here at
    # all. It exists so a 400-paper day still costs a bounded number of tokens.
    vectors = embedder.encode([p.embed_text for p in unseen])
    candidates = preselect(unseen, vectors, store, screen_n=cfg.screen_n)
    counts["screened"] = len(candidates)

    # 4b. guard the papers before anything reaches a model.
    #
    # Layer 1 of guard.py (fencing, sanitising) applies to every paper
    # regardless of this step; this is the optional Lakera layer above it.
    guard_info = {
        "lakera_enabled": cfg.guard_enabled,
        "lakera_available": False,
        "on_error": cfg.guard_on_error,
        "screened": 0,
        "blocked": [],
    }
    if cfg.guard_enabled:
        client_guard = guard.LakeraGuard(
            Config.lakera_key(),
            endpoint=cfg.guard_endpoint,
            project_id=cfg.guard_project_id,
            timeout=cfg.guard_timeout,
            on_error=cfg.guard_on_error,
        )
        guard_info["lakera_available"] = client_guard.available
        if client_guard.available:
            safe, blocked = guard.screen_papers([c.paper for c in candidates], client_guard)
            safe_ids = {p.arxiv_id for p in safe}
            candidates = [c for c in candidates if c.paper.arxiv_id in safe_ids]
            guard_info["screened"] = len(safe) + len(blocked)
            guard_info["blocked"] = blocked
            for b in blocked:
                problems.append(
                    f"{b['arxiv_id']}: withheld from the model calls -- {b['reason']}"
                    + (f" ({', '.join(b['detectors'])})" if b["detectors"] else "")
                )
        else:
            msg = ("Lakera screening did not run: LAKERA_GUARD_API_KEY is not set. "
                   "Prompt fencing and sanitising still applied.")
            log.warning(msg)
            problems.append(msg)
    result["guard"] = guard_info

    # 5. call 1: the cheap model screens every candidate for relevance
    key = Config.openrouter_key()
    # Low effort by design: measured on 2026-08-20, reasoning was 9,824 of the
    # screen's 16,631 output tokens -- a third of its cost -- to answer a yes/no
    # question. The judge still reasons; it is the call that needs to.
    screen_client = ModelClient(cfg.screen_model, effort=cfg.screen_effort,
                                api_key=key)
    verdicts, screen_problems = screen.screen_candidates(
        screen_client, cfg.profile, candidates, batch_size=cfg.screen_batch_size,
    )
    problems.extend(screen_problems)
    passed = screen.relevant(candidates, verdicts)
    counts["relevant"] = len(passed)
    log.info("screen kept %d of %d paper(s)", len(passed), len(candidates))

    if not passed and verdicts:
        problems.append(
            f"the screen found nothing relevant in {len(verdicts)} paper(s); "
            f"a thin day is a legitimate outcome, not necessarily a fault"
        )

    # 6. call 2: the strong model judges what passed
    judge_client = ModelClient(cfg.judge_model, effort=cfg.effort, api_key=key)
    judgements, judge_problems = judge.judge_candidates(
        judge_client, cfg.profile, passed,
    )
    problems.extend(judge_problems)

    if not judgements:
        # Without judgements there is no defensible top 10. Send what the
        # screen passed rather than nothing: a yes/no verdict is still real,
        # and it is a better fallback than the similarity order ever was.
        if passed:
            problems.append(
                "no judgements returned; falling back to the screen's verdicts"
            )
        kept = passed[: cfg.top_n]
    else:
        kept = judge.rank(passed, judgements, cfg.top_n)
    counts["kept"] = len(kept)

    # 7. one question-extraction call per kept paper
    client = ModelClient(cfg.model, effort=cfg.effort, api_key=key)
    tag_vocab = canon.known_tags()
    kept_ids = {c.paper.arxiv_id for c in kept}
    for c in kept:
        entry = c.to_dict()
        j = judgements.get(c.paper.arxiv_id, {})
        v = verdicts.get(c.paper.arxiv_id, {})
        entry["on_topic"] = j.get("on_topic")
        entry["topic_reason"] = j.get("topic_reason", "")
        entry["significance"] = j.get("significance")
        entry["novelty"] = j.get("novelty")
        entry["one_sentence"] = j.get("one_sentence", "")
        entry["screen_reason"] = v.get("reason", "")
        entry["open_questions"] = []
        entry["canon"] = {}
        # Advisory only -- see guard.py on why keyword hits never block here.
        entry["suspicious_markers"] = guard.suspicious_markers(
            f"{c.paper.title}\n{c.paper.abstract}"
        )

        try:
            extracted = questions.extract(client, cfg.profile, c.paper, tag_vocab)
            entry["open_questions"] = extracted["open_questions"]
            entry["canon"] = extracted["canon"]
        except ModelError as exc:
            # Keep the other papers. Mark the failure.
            msg = f"{c.paper.arxiv_id}: question extraction failed ({exc})"
            problems.append(msg)
            log.error(msg)

        result["papers"].append(entry)

    # 8. record every paper that was screened, not just the ten that were sent.
    #
    # The screen's yes/no and its reason are kept for all of them, and the
    # judge's scores for the subset that passed. That pair is the record of why
    # a paper was or was not sent, and it is the pool the canon grows from. The
    # abstract is not repeated here; it is in data/raw/ for the same day.
    kept_entries = {e["arxiv_id"]: e for e in result["papers"]}
    candidate_rows = []
    for c in candidates:
        aid = c.paper.arxiv_id
        j = judgements.get(aid, {})
        v = verdicts.get(aid, {})
        tags = (kept_entries.get(aid) or {}).get("canon") or {}
        emailed = aid in kept_ids

        brief = c.to_dict()
        brief.pop("abstract", None)
        brief.pop("categories", None)
        result["screened"].append(
            {
                **brief,
                "relevant": v.get("relevant"),
                "screen_reason": v.get("reason", ""),
                "on_topic": j.get("on_topic"),
                "topic_reason": j.get("topic_reason", ""),
                "significance": j.get("significance"),
                "novelty": j.get("novelty"),
                "one_sentence": j.get("one_sentence", ""),
                "kept": emailed,
            }
        )

        candidate_rows.append(
            canon.to_canon_row(
                paper=c.paper,
                tags=tags,
                summary=tags.get("summary", "") or j.get("one_sentence", ""),
                similarity=c.similarity,
                similarity_rank=c.rank,
                nearest_anchor_id=c.nearest_anchor_id,
                significance=j.get("significance"),
                novelty=j.get("novelty"),
                screen_relevant=v.get("relevant"),
                first_seen=day,
                emailed=emailed,
            )
        )

    # 9. write the archive. A failed delivery must not cost the data.
    write_output(cfg, result, day)
    canon.append_candidates(cfg.candidates_csv, candidate_rows)

    # 10. rebuild the feed from every day file on disk, including today's. No
    # password, no account: this is the channel that needs neither. Rebuilt on
    # a dry run too, so --dry-run lets you inspect data/feed.xml locally.
    n_entries = feed_mod.rebuild(
        DATA_DIR, cfg.feed_path, cfg.feed_site_url, cfg.feed_url, result["generated_at"],
        max_entries=cfg.feed_max_entries,
    )
    result["feed"] = {"entries": n_entries, "path": str(cfg.feed_path.relative_to(DATA_DIR.parent))}

    if dry_run:
        log.info("dry run: %d paper(s) written, %d feed entries, nothing marked seen",
                 len(result["papers"]), n_entries)
        write_output(cfg, result, day)
        return result

    # By this point every paper is already in data/feed.xml, so it must not
    # be shown again.
    seen.mark([p["arxiv_id"] for p in result["papers"]], day)
    seen.save()

    write_output(cfg, result, day)          # rewrite now that "feed" is set
    return result


def _union_by_id(*groups) -> list[dict]:
    """Rows from every group, first occurrence of each arxiv_id winning."""
    out: dict[str, dict] = {}
    for group in groups:
        for row in group or []:
            aid = row.get("arxiv_id")
            if aid and aid not in out:
                out[aid] = row
    return list(out.values())


def merge_into_existing(result: dict, path: Path) -> dict:
    """Fold a day file already on disk into this run's result, in place.

    Re-running a day is routine: a manual run and the scheduled run can both
    land on it, or a crashed run gets retried. But the first run marked its
    papers seen, so the second run's screen never sees them again -- and
    writing only the second run's papers drops the first run's from the page
    for good, while seen.json guarantees they can never come back. That is
    how 2026-08-20 went from ten papers to two across five runs.

    So the day file is additive. Papers and screening rows are unioned on
    arxiv_id, keeping what is already published. Counts take the larger of
    the two, except `kept`, which is recomputed from the union so it always
    describes the file rather than the last run.
    """
    if not path.exists():
        return result
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        # A day file that cannot be read is one this run is about to replace
        # anyway. Say so rather than failing the run.
        log.warning("cannot merge %s (%s); writing this run's papers alone", path, exc)
        return result

    # A file for a different date means output_path changed shape; never
    # merge across days.
    if old.get("date") != result.get("date"):
        return result

    kept_before = len(result.get("papers", []))
    result["papers"] = _union_by_id(old.get("papers"), result.get("papers"))
    result["screened"] = _union_by_id(old.get("screened"), result.get("screened"))

    counts = dict(result.get("counts") or {})
    for key, was in (old.get("counts") or {}).items():
        now = counts.get(key)
        if isinstance(was, int) and isinstance(now, int):
            counts[key] = max(now, was)
    result["counts"] = counts

    recovered = len(result["papers"]) - kept_before
    if recovered:
        log.info("%s: kept %d paper(s) already on file, %d in this run",
                 result["date"], recovered, kept_before)
    return result


def coherent_counts(result: dict) -> dict:
    """Keep the day's funnel monotone: fetched >= unseen >= screened >= relevant >= kept.

    A single run reports these consistently. Merging two runs does not: the
    second run screened only what the first had not already sent, so its
    `relevant` counts a different, smaller population than the file now
    holds. On 2026-08-20 that read "2 relevant, 11 kept", which cannot
    happen -- every paper in the file was judged relevant by whichever run
    kept it, so `relevant` can never sit below `kept`.

    Each level is raised to the largest number the file itself can support:
    what the run reported, what its own records show, and what the level
    below it already proves. Nothing is invented and nothing shrinks.
    """
    counts = dict(result.get("counts") or {})
    papers = result.get("papers") or []
    screened = result.get("screened") or []

    counts["kept"] = len(papers)
    marked_relevant = sum(1 for row in screened if row.get("relevant") is True)
    counts["relevant"] = max(counts.get("relevant") or 0, marked_relevant,
                             counts["kept"])
    counts["screened"] = max(counts.get("screened") or 0, len(screened),
                             counts["relevant"])
    counts["unseen"] = max(counts.get("unseen") or 0, counts["screened"])
    counts["fetched"] = max(counts.get("fetched") or 0, counts["unseen"])

    result["counts"] = counts
    return counts


def write_output(cfg: Config, result: dict, day: str) -> None:
    """`data/YYYY-MM-DD.json`, plus `data/latest.json` as a stable read URL.

    latest.json is a copy, not a symlink: this repo is read over
    raw.githubusercontent.com by largeagentsystems.org, and raw serves the link
    text for a symlink, not the file it points at.

    Additive: see merge_into_existing. A second run of a day adds to it, and
    can never subtract from it.
    """
    path = cfg.output_path(day)
    path.parent.mkdir(parents=True, exist_ok=True)
    merge_into_existing(result, path)
    coherent_counts(result)
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    path.write_text(payload, encoding="utf-8")
    (path.parent / "latest.json").write_text(payload, encoding="utf-8")
    log.info("wrote %s", path)
