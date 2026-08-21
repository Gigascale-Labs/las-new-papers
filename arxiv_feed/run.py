"""The daily run, in the order the spec lays it out.

fetch -> drop seen -> embed -> similarity filter -> score (1 call) -> keep top N
-> questions (1 call per paper) -> email -> save.

The one rule that shapes the error handling throughout: never stop the whole run
because one paper failed. Every failure below is collected into `problems` and
reported in the email rather than raised.
"""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime, timezone

from . import anchors as anchors_mod
from . import arxiv, canon, emailer, guard, questions, score
from .config import Config, anchor_count_warning
from .embed import Embedder
from .llm import ModelClient, ModelError
from .seen import SeenStore
from .select import shortlist
from scrapers.arxiv_scraper import scrape_day

log = logging.getLogger(__name__)


def run(cfg: Config, day: str | None = None, dry_run: bool = False,
        seed: int | None = None, rebuild_anchors: bool = False) -> dict:
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
    papers = scrape_day(cfg.categories, day)
    log.info("fetched %d papers for %s", len(papers), day)
    seen = SeenStore(cfg.seen_path)
    unseen = seen.filter_unseen(papers)
    log.info("%d unseen (%d already sent)", len(unseen), len(papers) - len(unseen))

    counts = {
        "fetched": len(papers),
        "unseen": len(unseen),
        "shortlisted": 0,
        "kept": 0,
        "anchors": len(store.ids),
    }
    result = {
        "date": day,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "categories": cfg.categories,
            "shortlist_n": cfg.shortlist_n,
            "explore_n": cfg.explore_n,
            "top_n": cfg.top_n,
            "embed_model": cfg.embed_model,
            "model": cfg.model,
            "effort": cfg.effort,
            "anchor_count": len(store.ids),
        },
        "counts": counts,
        "papers": [],
        "problems": problems,
        # No address here: data/*.json is committed to a public repository by
        # the daily workflow.
        "email": {"sent": False, "error": None, "dry_run": dry_run},
    }

    if not unseen:
        problems.append(f"no unseen papers found for {day}")
        return result

    # 3. embed + 4. similarity filter
    vectors = embedder.encode([p.embed_text for p in unseen])
    candidates = shortlist(
        unseen, vectors, store,
        shortlist_n=cfg.shortlist_n,
        explore_n=cfg.explore_n,
        rng=random.Random(seed),
    )
    counts["shortlisted"] = len(candidates)

    # 4b. screen the shortlist before anything reaches a model.
    #
    # The shortlist, not the whole day: 45 screening calls instead of ~500, and
    # a paper that never reaches a model needs no screening. Layer 1 of guard.py
    # (fencing, sanitising) applies to every paper regardless of this step.
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

    # 5. one scoring call for the whole shortlist
    client = ModelClient(cfg.model, effort=cfg.effort)
    scores, score_problems = score.score_candidates(client, cfg.profile, candidates)
    problems.extend(score_problems)

    if not scores:
        # Without scores there is no defensible top 10. Send what the filter
        # found rather than nothing: the similarity ranking is still real.
        problems.append("no scores returned; falling back to the similarity ranking")
        kept = [c for c in candidates if not c.from_random][: cfg.top_n]
    else:
        kept = score.rank(candidates, scores, cfg.top_n)
    counts["kept"] = len(kept)

    # 6. one question-extraction call per kept paper
    tag_vocab = canon.known_tags()
    finalist_rows = []
    for c in kept:
        entry = c.to_dict()
        s = scores.get(c.paper.arxiv_id, {})
        entry["significance"] = s.get("significance")
        entry["novelty"] = s.get("novelty")
        entry["one_sentence"] = s.get("one_sentence", "")
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
            # Spec section 9: keep the other papers, mark the failure.
            msg = f"{c.paper.arxiv_id}: question extraction failed ({exc})"
            problems.append(msg)
            log.error(msg)

        result["papers"].append(entry)

        if entry["canon"]:
            finalist_rows.append(
                canon.to_canon_row(
                    paper=c.paper,
                    tags=entry["canon"],
                    summary=entry["canon"].get("summary", "") or entry["one_sentence"],
                    similarity=c.similarity,
                    nearest_anchor_id=c.nearest_anchor_id,
                    significance=entry["significance"],
                    novelty=entry["novelty"],
                    from_random=c.from_random,
                    first_seen=day,
                )
            )

    # 7. write the archive before sending: an email that fails must not cost the data
    write_output(cfg, result, day)
    canon.append_finalists(cfg.finalists_csv, finalist_rows)

    # 8. send
    if dry_run:
        log.info("dry run: no email sent, %d paper(s) written", len(result["papers"]))
        return result

    try:
        emailer.send(result, cfg)
        result["email"]["sent"] = True
        seen.mark([p["arxiv_id"] for p in result["papers"]], day)
        seen.save()
    except emailer.EmailError as exc:
        # The JSON is already on disk. Log loudly, keep the file, do not mark the
        # papers as sent -- they were not.
        result["email"]["error"] = str(exc)
        problems.append(f"email not sent: {exc}")
        log.error("email not sent: %s", exc)

    write_output(cfg, result, day)          # rewrite with the email outcome
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
