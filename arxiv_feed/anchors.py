"""Anchor vectors, cached on disk.

Anchors change rarely, and building them costs one arXiv call plus a few
seconds of CPU, so the result is saved. The cache records which anchor IDs and
which model produced it. If either changes, the store rebuilds itself.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import arxiv
from .config import Config
from .embed import Embedder

log = logging.getLogger(__name__)


class AnchorError(Exception):
    """No usable anchor vectors could be produced."""


@dataclass
class AnchorStore:
    ids: list[str]
    titles: list[str]
    vectors: np.ndarray            # (n_anchors, dim), unit rows
    model_name: str
    built_at: str

    def similarities(self, paper_vectors: np.ndarray) -> np.ndarray:
        """(n_papers, n_anchors) cosine similarity matrix.

        Both sides are unit vectors, so a dot product *is* the cosine.
        """
        if paper_vectors.size == 0:
            return np.zeros((0, len(self.ids)), dtype=np.float32)
        return paper_vectors @ self.vectors.T

    def best_match(self, paper_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Highest similarity to any *single* anchor, and which anchor that was.

        Deliberately max, not mean: the anchor set spans simulation, market
        design, governance and safety, and the mean of those directions points at
        no real paper. A new market-design paper should score on the market-design
        anchors alone.
        """
        sims = self.similarities(paper_vectors)
        if sims.size == 0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=int)
        return sims.max(axis=1), sims.argmax(axis=1)

    def drop(self, arxiv_id: str) -> "AnchorStore":
        """This store minus one anchor. Used by the leave-one-out test."""
        keep = [i for i, a in enumerate(self.ids) if a != arxiv_id]
        return AnchorStore(
            ids=[self.ids[i] for i in keep],
            titles=[self.titles[i] for i in keep],
            vectors=self.vectors[keep],
            model_name=self.model_name,
            built_at=self.built_at,
        )


def _meta_matches(meta: dict, cfg: Config) -> bool:
    return (
        meta.get("model") == cfg.embed_model
        and list(meta.get("ids", [])) == list(cfg.anchors)
    )


def build(cfg: Config, embedder: Embedder) -> AnchorStore:
    """Fetch anchor metadata from arXiv, embed it, and write the cache."""
    log.info("building anchor vectors for %d anchors", len(cfg.anchors))
    meta_by_id = arxiv.fetch_by_ids(cfg.anchors)

    missing = [a for a in cfg.anchors if a not in meta_by_id]
    if missing:
        # Not fatal: a withdrawn or mistyped anchor should cost you that anchor,
        # not the day's email.
        log.warning("arXiv returned nothing for %d anchor(s): %s",
                    len(missing), ", ".join(missing))

    ids = [a for a in cfg.anchors if a in meta_by_id]
    if not ids:
        raise AnchorError("arXiv returned no metadata for any configured anchor")

    papers = [meta_by_id[a] for a in ids]
    vectors = embedder.encode([p.embed_text for p in papers])

    cfg.anchors_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(cfg.anchors_npy, vectors)
    cfg.anchors_meta.write_text(
        json.dumps(
            {
                "model": cfg.embed_model,
                # The configured list, not the resolved one: if a previously
                # missing anchor comes back, the mismatch triggers a rebuild.
                "ids": cfg.anchors,
                "resolved_ids": ids,
                "titles": [p.title for p in papers],
                "missing": missing,
                "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("anchor vectors written to %s", cfg.anchors_npy)

    return AnchorStore(
        ids=ids,
        titles=[p.title for p in papers],
        vectors=vectors,
        model_name=cfg.embed_model,
        built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def load_or_build(cfg: Config, embedder: Embedder, force: bool = False) -> AnchorStore:
    """Return the cached store when it still matches the config, else rebuild."""
    npy: Path = cfg.anchors_npy
    meta_path: Path = cfg.anchors_meta

    if not force and npy.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if _meta_matches(meta, cfg):
                vectors = np.load(npy)
                resolved = meta.get("resolved_ids") or meta["ids"]
                if vectors.shape[0] == len(resolved):
                    log.info("using cached anchor vectors (%d anchors)", len(resolved))
                    return AnchorStore(
                        ids=list(resolved),
                        titles=list(meta.get("titles", resolved)),
                        vectors=vectors,
                        model_name=meta["model"],
                        built_at=meta.get("built_at", ""),
                    )
                log.warning("anchor cache is inconsistent with its metadata; rebuilding")
            else:
                log.info("anchor list or embedding model changed; rebuilding vectors")
        except Exception as exc:
            log.warning("anchor cache unreadable (%s); rebuilding", exc)

    return build(cfg, embedder)
