"""Call 1: score the 45 shortlisted papers.

One call for all of them, not one per paper. A score only means something in
comparison, so the model sees the whole shortlist at once.

Similarity cannot do this job. It says a paper is about the right subject. It
says nothing about whether the paper is good, or whether anyone has done it
before.
"""

from __future__ import annotations

import logging

from .guard import (DATA_NOT_INSTRUCTIONS, MAX_ABSTRACT_CHARS, MAX_TITLE_CHARS,
                    fence, sanitize)
from .llm import ModelClient, ModelError
from .select import Candidate

log = logging.getLogger(__name__)

SYSTEM = """\
You score new arXiv papers for one researcher, whose profile is given below.

You judge two things, each 1-5:

significance -- how much this paper matters to that researcher's stated
interests if its claims hold. 5 means it would change how they approach a
problem they care about. 1 means it is outside their interests or too minor to
act on. Judge significance *for this researcher*, not in general: a celebrated
result in an unrelated field is a 1 here.

novelty -- how new the contribution is against the published literature you
know. 5 means you know of no prior work doing this. 3 means a real but
incremental advance on established work. 1 means a routine application of a
standard method, or a result already well established.

Score the whole set relative to each other. A typical day should use the middle
of both scales; reserve 5s for papers that genuinely earn them.

Also write one sentence per paper saying what it does -- plain, factual, no
adjectives, no "this paper". If the abstract does not say, say less rather than
guessing.

Some papers were drawn at random rather than by topical similarity, and will be
unrelated to the profile. Score them honestly on the same scales: an unrelated
paper is simply low significance for this researcher.

Return a score for every paper you are given, keyed by its exact arxiv_id.

""" + DATA_NOT_INSTRUCTIONS

SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string"},
                    "significance": {"type": "integer", "minimum": 1, "maximum": 5},
                    "novelty": {"type": "integer", "minimum": 1, "maximum": 5},
                    "one_sentence": {"type": "string"},
                },
                "required": ["arxiv_id", "significance", "novelty", "one_sentence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["scores"],
    "additionalProperties": False,
}


def _render(candidates: list[Candidate]) -> str:
    """One fenced block per paper.

    The arxiv_id sits outside the fence: it is the key the model must return,
    it is validated against the shortlist afterwards, and keeping it out means
    no abstract can appear to relabel another paper's id.
    """
    blocks = []
    for c in candidates:
        title = sanitize(c.paper.title, MAX_TITLE_CHARS)
        abstract = sanitize(c.paper.abstract, MAX_ABSTRACT_CHARS)
        body, _ = fence(f"title: {title}\nabstract: {abstract}")
        blocks.append(f"arxiv_id: {c.paper.arxiv_id}\n{body}")
    return "\n\n".join(blocks)


def score_candidates(
    client: ModelClient, profile: str, candidates: list[Candidate]
) -> tuple[dict[str, dict], list[str]]:
    """Score every candidate. Returns (scores by arxiv_id, list of failures).

    A candidate the model omits is reported rather than defaulted: a paper with
    an invented score would compete for a top-10 slot it never earned.
    """
    if not candidates:
        return {}, []

    user = (
        f"RESEARCHER PROFILE\n{profile}\n\n"
        f"PAPERS ({len(candidates)})\n\n{_render(candidates)}"
    )

    try:
        data = client.structured(
            system=SYSTEM,
            user=user,
            schema=SCHEMA,
            max_tokens=16000,
            label="call 1 (scoring)",
        )
    except ModelError as exc:
        log.error("scoring call failed: %s", exc)
        return {}, [f"scoring call failed: {exc}"]

    scores: dict[str, dict] = {}
    wanted = {c.paper.arxiv_id for c in candidates}
    raw_scores = data.get("scores", [])
    returned_ids = [str(row.get("arxiv_id", "")).strip() for row in raw_scores]
    log.info(
        "call 1 (scoring): got %d score row(s) for %d wanted paper(s); "
        "sample returned ids=%s, sample wanted ids=%s",
        len(raw_scores), len(wanted),
        returned_ids[:5], sorted(wanted)[:5],
    )
    for row in raw_scores:
        aid = str(row.get("arxiv_id", "")).strip()
        if aid in wanted:
            scores[aid] = {
                "significance": int(row["significance"]),
                "novelty": int(row["novelty"]),
                "one_sentence": str(row.get("one_sentence", "")).strip(),
            }

    problems = []
    missing = sorted(wanted - set(scores))
    if missing:
        problems.append(
            f"scoring call returned no score for {len(missing)} paper(s): {', '.join(missing)}"
        )
        log.warning(problems[-1])

    return scores, problems


def rank(candidates: list[Candidate], scores: dict[str, dict], top_n: int) -> list[Candidate]:
    """The `top_n` best-scoring candidates.

    Significance and novelty are summed with significance breaking ties, so a
    highly novel paper the researcher does not care about loses to a merely
    solid one that they do. Similarity is the last tie-break, which keeps the
    ordering deterministic for a given day.
    """
    scored = [c for c in candidates if c.paper.arxiv_id in scores]
    scored.sort(
        key=lambda c: (
            scores[c.paper.arxiv_id]["significance"] + scores[c.paper.arxiv_id]["novelty"],
            scores[c.paper.arxiv_id]["significance"],
            c.similarity,
        ),
        reverse=True,
    )
    return scored[:top_n]
