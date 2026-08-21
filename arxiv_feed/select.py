"""The similarity filter: 40 nearest + 5 from the tail.

This is the step that makes the whole thing affordable -- ~500 papers in, 45
out, for the cost of a numpy dot product. It is a *filter*, not a judgement:
significance and novelty are decided later, by the model.
"""

from __future__ import annotations

import logging
import random
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
    from_random: bool = False  # picked by the explore slice, not by similarity

    def to_dict(self) -> dict:
        return {
            **self.paper.to_dict(),
            "similarity": round(self.similarity, 4),
            "nearest_anchor_id": self.nearest_anchor_id,
            "nearest_anchor_title": self.nearest_anchor_title,
            "similarity_rank": self.rank,
            "from_random": self.from_random,
        }


def shortlist(
    papers: list[Paper],
    vectors: np.ndarray,
    store: AnchorStore,
    shortlist_n: int = 40,
    explore_n: int = 5,
    explore_pool: int = 100,
    rng: random.Random | None = None,
) -> list[Candidate]:
    """Rank by similarity, keep the top `shortlist_n`, add `explore_n` at random.

    The random papers come from the `explore_pool` ranks immediately below the
    cut (41-140 by default). They exist because similarity can only find more of
    what the anchors already describe: a genuinely new subfield has no anchor to
    be close to, and would never surface without this slice. They are marked in
    the output so a good paper from the tail is visible as evidence the slice is
    earning its place.
    """
    if not papers:
        return []
    rng = rng or random.Random()

    best_sim, best_idx = store.best_match(vectors)
    order = np.argsort(-best_sim)          # descending similarity

    def make(pos: int, rank: int, from_random: bool) -> Candidate:
        anchor_i = int(best_idx[pos])
        return Candidate(
            paper=papers[pos],
            similarity=float(best_sim[pos]),
            nearest_anchor_id=store.ids[anchor_i],
            nearest_anchor_title=store.titles[anchor_i],
            rank=rank,
            from_random=from_random,
        )

    top = [make(int(p), i + 1, False) for i, p in enumerate(order[:shortlist_n])]

    tail = order[shortlist_n : shortlist_n + explore_pool]
    picks = rng.sample(list(range(len(tail))), min(explore_n, len(tail)))
    explore = [make(int(tail[i]), shortlist_n + i + 1, True) for i in sorted(picks)]

    log.info(
        "shortlisted %d by similarity (%.3f-%.3f) + %d random from ranks %d-%d",
        len(top),
        top[-1].similarity if top else float("nan"),
        top[0].similarity if top else float("nan"),
        len(explore),
        shortlist_n + 1,
        shortlist_n + len(tail),
    )
    return top + explore
