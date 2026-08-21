"""Test doubles: papers, vectors, and a scripted model client.

These let the pipeline logic be tested with no network, no API key, and no
model download.
"""

from __future__ import annotations

import numpy as np

from arxiv_feed.anchors import AnchorStore
from arxiv_feed.arxiv import Paper
from arxiv_feed.llm import ModelError


def paper(i: int, title: str | None = None) -> Paper:
    return Paper(
        arxiv_id=f"2608.{i:05d}",
        title=title or f"Test paper {i}",
        abstract=f"Abstract for test paper {i}.",
        authors=[f"Author {i}A", f"Author {i}B"],
        published="2026-08-20T00:00:00Z",
        updated="2026-08-20T00:00:00Z",
        categories=["cs.MA"],
    )


def unit(vec) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    return v / np.linalg.norm(v)


def store(n_anchors: int = 3, dim: int = 4) -> AnchorStore:
    vecs = np.eye(n_anchors, dim, dtype=np.float32)
    return AnchorStore(
        ids=[f"anchor{i}" for i in range(n_anchors)],
        titles=[f"Anchor paper {i}" for i in range(n_anchors)],
        vectors=vecs,
        model_name="stub",
        built_at="2026-08-21T00:00:00+00:00",
    )


class ScriptedClient:
    """Returns queued responses; raises what it is told to raise.

    `structured` matches the real ModelClient's signature so it can be dropped
    straight in.
    """

    def __init__(self, responses: list):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def structured(self, *, system, user, schema, max_tokens=16000, label="call"):
        self.calls.append({"label": label, "user": user, "schema": schema})
        if not self.responses:
            raise ModelError(f"{label}: no scripted response left")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
