"""Writes the kept papers' embedding vectors to data/embeddings/YYYY-MM-DD.json.

Every run encodes the day's papers to order them (embed.py, then
select.preselect) and then discards the vectors at exit. They are the only
semantic coordinate this project holds. largeagentsystems.org projects them to
two dimensions to draw the reading list as a map.

Three decisions, with the numbers behind them:

- **A file of its own, not a field on the day file.** A 768-float array per
  paper adds about 6.5KB per paper to data/YYYY-MM-DD.json, which every reader
  of the reading list would then download. Measured: 52 papers over 7 days take
  337KB.
- **Kept papers only.** The screened pool records what the filter decided and
  nothing displays it. data/raw/ already archives every paper of the day.
- **Four decimal places.** Embedder.encode returns L2-normalised float32, so
  every component falls in [-1, 1] and 4 dp resolves it to 5e-5. That is a
  third the size of the full float repr. Measured downstream: reproducing each
  vector's nearest anchor from these rounded values matched the published
  similarity for 49 of 52 papers, median absolute error 3e-5, mean 3.8e-4, max
  9.9e-3. All 3 misses name an anchor no longer in the anchor set.

The write is additive, for the reason run.merge_into_existing is: a second run
of a day never sees the first run's papers again, because seen.json holds them.
Writing only this run's vectors would drop the first run's for good.

Not checked: whether a model change mid-archive is handled anywhere. This file
records the model name; nothing reads it back and refuses a mismatch.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

#: Decimal places per component. See the module docstring for the error it costs.
DECIMALS = 4


def quantise(vector) -> list[float]:
    """Returns one vector as a JSON-safe list, rounded to DECIMALS places."""
    return [round(float(x), DECIMALS) for x in vector]


def merge_into_existing(payload: dict, path: Path) -> dict:
    """Folds the vectors already on disk into this run's payload, in place.

    This run's vector wins an id present in both. Absent a model change the two
    are equal, because the same model read the same text; on a model change the
    newer vector is the one to keep.

    Two files this skips rather than merging: one that does not parse, and one
    carrying a different date. Both mean the file on disk is not the file this
    payload extends. Neither fails the run.
    """
    if not path.exists():
        return payload
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("cannot merge %s (%s); writing this run's vectors alone", path, exc)
        return payload

    if not isinstance(old, dict) or old.get("date") != payload.get("date"):
        return payload

    old_vectors = old.get("vectors")
    if not isinstance(old_vectors, dict):
        return payload

    payload["vectors"] = {**old_vectors, **payload["vectors"]}
    return payload


def write(path: Path, day: str, model: str, vectors_by_id: dict) -> int:
    """Writes data/embeddings/<day>.json and returns how many vectors it holds.

    `vectors_by_id` maps arxiv_id to a vector for this run's kept papers. This
    skips an id whose vector is None: that paper came back from an earlier run
    of the same day, this run never encoded it, and the merge below already
    carries its vector.
    """
    vectors = {
        arxiv_id: quantise(vector)
        for arxiv_id, vector in vectors_by_id.items()
        if vector is not None
    }
    payload = {"date": day, "model": model, "vectors": vectors}
    merge_into_existing(payload, path)
    # After the merge, not before. A run that encoded nothing new still
    # describes the file it writes, which may hold an earlier run's vectors.
    merged = payload["vectors"]
    payload["dim"] = len(next(iter(merged.values()))) if merged else 0
    payload["count"] = len(merged)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %d vector(s) to %s", payload["count"], path)
    return payload["count"]
