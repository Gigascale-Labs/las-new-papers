"""The daily run, in order.

fetch -> drop seen -> embed -> pre-sort to the cap -> guard -> screen (cheap
model, every paper) -> judge (strong model, what passed) -> keep the top 10 ->
questions -> email -> save.

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
from datetime import datetime, timezone

from . import anchors as anchors_mod
from . import arxiv, canon, emailer, feed as feed_mod, guard, judge, questions, screen
from .config import DATA_DIR, Config, anchor_count_warning
from .embed import Embedder
from .llm import ModelClient, ModelError
from .seen import SeenStore
from .select import preselect
from scrapers.arxiv_scraper import scrape_day

log = logging.getLogger(__name__)


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
    embedder = Embedder(cfg.embed_model)
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
        # No address here: data/*.json is committed to a public repository by
        # the daily workflow.
        "email": {"sent": False, "error": None, "dry_run": dry_run},
    }

    if not unseen:
        problems.append(f"no unseen papers found for {day}")
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
        DATA_DIR, cfg.feed_path, cfg.feed_url, cfg.feed_url, result["generated_at"],
        max_entries=cfg.feed_max_entries,
    )
    result["feed"] = {"entries": n_entries, "path": str(cfg.feed_path.relative_to(DATA_DIR.parent))}

    if dry_run:
        log.info("dry run: no email sent, %d paper(s) written, %d feed entries",
                 len(result["papers"]), n_entries)
        write_output(cfg, result, day)
        return result

    # 11. email, if a recipient is configured. Optional: the feed above has
    # already delivered the day, so a missing or failing email is not fatal to
    # anything but the email itself.
    if cfg.email_to():
        try:
            emailer.send(result, cfg)
            result["email"]["sent"] = True
        except emailer.EmailError as exc:
            result["email"]["error"] = str(exc)
            problems.append(f"email not sent: {exc}")
            log.error("email not sent: %s", exc)
    else:
        log.info("no FEED_EMAIL_TO configured; delivered via the feed only")

    # Seen-marking follows the feed, not the email: by this point every paper
    # is already in data/feed.xml, so it must not be shown again regardless of
    # whether the optional email succeeded.
    seen.mark([p["arxiv_id"] for p in result["papers"]], day)
    seen.save()

    write_output(cfg, result, day)          # rewrite with the delivery outcome
    return result


def write_output(cfg: Config, result: dict, day: str) -> None:
    """`data/YYYY-MM-DD.json`, plus `data/latest.json` as a stable read URL.

    latest.json is a copy, not a symlink: this repo is read over
    raw.githubusercontent.com by largeagentsystems.org, and raw serves the link
    text for a symlink, not the file it points at.
    """
    path = cfg.output_path(day)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    path.write_text(payload, encoding="utf-8")
    (path.parent / "latest.json").write_text(payload, encoding="utf-8")
    log.info("wrote %s", path)
