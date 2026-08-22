"""The similarity pre-sort: which papers reach the screening model.

This is no longer the filter. It is a cap.

The screening call reads up to `screen_n` papers a day for about ten cents.
Similarity no longer decides what gets judged. It decides only what gets
dropped on a day too large to screen whole. Most days sit under the cap. On
those days nothing is dropped.

It orders. It does not judge. The screening and judging calls decide relevance,
significance and novelty.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .anchors import AnchorStore
from .arxiv import Paper

log = logging.getLogger(__name__)


@dataclass
class Candidate:
    paper: Paper
    similarity: float
    nearest_anchor_id: str
    nearest_anchor_title: str
    rank: int                  # 1-based rank by similarity across all papers that day

    def to_dict(self) -> dict:
        return {
            **self.paper.to_dict(),
            "similarity": round(self.similarity, 4),
            "nearest_anchor_id": self.nearest_anchor_id,
            "nearest_anchor_title": self.nearest_anchor_title,
            "similarity_rank": self.rank,
        }


def preselect(
    papers: list[Paper],
    vectors: np.ndarray,
    store: AnchorStore,
    screen_n: int = 200,
) -> list[Candidate]:
    """Rank by similarity to the nearest anchor and keep at most `screen_n`.

    Returns every paper when the day is smaller than the cap, which is the
    common case. The similarity and rank travel with each candidate because the
    archive records them, not because anything downstream filters on them.
    """
    if not papers:
        return []

    best_sim, best_idx = store.best_match(vectors)
    order = np.argsort(-best_sim)          # descending similarity

    kept = []
    for rank, pos in enumerate(order[:screen_n], start=1):
        pos = int(pos)
        anchor_i = int(best_idx[pos])
        kept.append(
            Candidate(
                paper=papers[pos],
                similarity=float(best_sim[pos]),
                nearest_anchor_id=store.ids[anchor_i],
                nearest_anchor_title=store.titles[anchor_i],
                rank=rank,
            )
        )

    dropped = len(papers) - len(kept)
    if dropped:
        log.info(
            "pre-sort: %d of %d papers kept for screening (%.3f-%.3f); "
            "%d dropped below the cap of %d",
            len(kept), len(papers), kept[-1].similarity, kept[0].similarity,
            dropped, screen_n,
        )
    else:
        log.info("pre-sort: all %d papers kept for screening (%.3f-%.3f); "
                 "day is under the cap of %d",
                 len(kept), kept[-1].similarity, kept[0].similarity, screen_n)
    return kept
