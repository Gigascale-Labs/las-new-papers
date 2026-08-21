#!/usr/bin/env python3
"""Leave-one-out: does the filter still find a paper you already care about?

Spec section 10, test 5. For each anchor tested:

  1. take that anchor out of the anchor set;
  2. fetch every paper submitted to the configured categories on the day that
     anchor was submitted;
  3. rank them against the *remaining* anchors;
  4. the held-out paper must come back in the top `shortlist_n`.

At least 8 of 10 must pass. A failure means the anchor set does not cover that
part of the field, or the embedding model cannot see the resemblance -- the
spec's two remedies, in that order.

This test makes real arXiv calls and runs the real embedding model. It makes no
model API calls and costs nothing. Day pools are cached under data/.loo_cache/
so a re-run is fast.

    python -m tests.leave_one_out                 # 10 anchors, seeded sample
    python -m tests.leave_one_out --anchors all
    python -m tests.leave_one_out --anchors 3 --seed 1
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from arxiv_feed import anchors as anchors_mod
from arxiv_feed import arxiv
from arxiv_feed.config import load_config
from arxiv_feed.embed import Embedder

log = logging.getLogger("leave_one_out")

CACHE = Path(__file__).resolve().parent.parent / "data" / ".loo_cache"


def day_pool(categories: list[str], day: str) -> list[arxiv.Paper]:
    """Every paper submitted on `day`, cached on disk between runs."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = CACHE / f"{day}_{'-'.join(sorted(categories))}.json"
    if key.exists():
        raw = json.loads(key.read_text(encoding="utf-8"))
        return [
            arxiv.Paper(
                arxiv_id=r["arxiv_id"], title=r["title"], abstract=r["abstract"],
                authors=r["authors"], published=r["published"], updated=r["updated"],
                categories=r.get("categories", []),
            )
            for r in raw
        ]
    papers = arxiv.fetch_new_papers(categories, day)
    key.write_text(json.dumps([p.to_dict() for p in papers]), encoding="utf-8")
    return papers


def run(anchors_to_test: list[str], config_path: str) -> dict:
    cfg = load_config(config_path)
    embedder = Embedder(cfg.embed_model)
    full_store = anchors_mod.load_or_build(cfg, embedder)
    meta = arxiv.fetch_by_ids(anchors_to_test)

    results = []
    for aid in anchors_to_test:
        held = meta.get(aid)
        if held is None:
            results.append({"anchor_id": aid, "status": "skipped",
                            "note": "arXiv returned no metadata for this ID"})
            log.warning("%s: no metadata, skipping", aid)
            continue
        if aid not in full_store.ids:
            results.append({"anchor_id": aid, "status": "skipped",
                            "note": "not in the built anchor store"})
            continue

        day = (held.published or "")[:10]
        pool = day_pool(cfg.categories, day)

        # The held-out paper may not be in its own day's pool: its primary list
        # may not be one of the configured categories. Adding it keeps the test
        # about ranking rather than about category configuration -- and the note
        # records that it was added, so the result stays honest.
        added = False
        if all(p.arxiv_id != aid for p in pool):
            pool = pool + [held]
            added = True

        store = full_store.drop(aid)
        vectors = embedder.encode([p.embed_text for p in pool])
        best, idx = store.best_match(vectors)
        order = np.argsort(-best)
        position = int(np.where(np.array([pool[i].arxiv_id for i in order]) == aid)[0][0]) + 1

        passed = position <= cfg.shortlist_n
        anchor_i = int(idx[[i for i, p in enumerate(pool) if p.arxiv_id == aid][0]])
        results.append({
            "anchor_id": aid,
            "title": held.title,
            "day": day,
            "pool_size": len(pool),
            "rank": position,
            "shortlist_n": cfg.shortlist_n,
            "similarity": round(float(best[[i for i, p in enumerate(pool)
                                            if p.arxiv_id == aid][0]]), 4),
            "nearest_remaining_anchor": store.ids[anchor_i],
            "nearest_remaining_anchor_title": store.titles[anchor_i],
            "injected_into_pool": added,
            "status": "pass" if passed else "fail",
        })
        log.info("%s rank %d/%d on %s -> %s", aid, position, len(pool), day,
                 results[-1]["status"])

    tested = [r for r in results if r["status"] in ("pass", "fail")]
    passes = sum(1 for r in tested if r["status"] == "pass")
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embed_model": cfg.embed_model,
        "shortlist_n": cfg.shortlist_n,
        "anchors_in_store": len(full_store.ids),
        "tested": len(tested),
        "passed": passes,
        "threshold": "at least 8 of 10 (80%)",
        "verdict": "pass" if tested and passes / len(tested) >= 0.8 else "fail",
        "results": results,
    }
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--anchors", default="10", help="how many to test, or 'all'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", default="data/eval/leave-one-out.json")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")

    cfg = load_config(args.config)
    if args.anchors == "all":
        chosen = list(cfg.anchors)
    else:
        n = min(int(args.anchors), len(cfg.anchors))
        chosen = sorted(random.Random(args.seed).sample(cfg.anchors, n))

    report = run(chosen, args.config)

    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print(f"{'anchor':<12} {'rank':>6} {'pool':>6}  {'status':<7} title")
    for r in report["results"]:
        if r["status"] == "skipped":
            print(f"{r['anchor_id']:<12} {'-':>6} {'-':>6}  skipped  {r['note']}")
        else:
            print(f"{r['anchor_id']:<12} {r['rank']:>6} {r['pool_size']:>6}  "
                  f"{r['status']:<7} {r['title'][:52]}")
    print()
    print(f"{report['passed']}/{report['tested']} passed "
          f"(threshold {report['threshold']}) -> {report['verdict'].upper()}")
    print(f"report written to {path}")
    return 0 if report["verdict"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
