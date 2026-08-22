"""Call 2: of the papers that passed the screen, which are worth sending?

One call for all of them, not one per paper. A judgement only means something
in comparison, so the model sees every paper that passed the screen at once.

The screen answered "is this the right kind of paper". This answers "is this
one good, and is it new" -- the two questions neither similarity nor a cheap
binary pass can answer.
"""

from __future__ import annotations

import logging

from .guard import (DATA_NOT_INSTRUCTIONS, MAX_ABSTRACT_CHARS, MAX_TITLE_CHARS,
                    fence, sanitize)
from .llm import ModelClient, ModelError
from .select import Candidate
from .style import PLAIN_ENGLISH

log = logging.getLogger(__name__)

SYSTEM = """\
You judge new arXiv papers for one researcher, whose profile is given below.
Every paper here already passed a relevance screen, so the question is no
longer whether it is on topic. The question is whether it is worth their time.

You judge two things, each 1-5:

significance -- how much this paper matters to that researcher's stated
interests if its claims hold. 5 means it would change how they approach a
problem they care about. 1 means it is too minor to act on. Judge significance
*for this researcher*, not in general.

novelty -- how new the contribution is against the published literature you
know. 5 means you know of no prior work doing this. 3 means a real but
incremental advance on established work. 1 means a routine application of a
standard method, or a result already well established.

Score the whole set relative to each other. A typical day should use the middle
of both scales; reserve 5s for papers that genuinely earn them. These papers
passed a screen, not a quality bar -- a day where every paper is a 2 is a
legitimate day, and inflating the scores to fill the digest serves nobody.

Also write a summary per paper saying what the paper does. At most three
sentences, each under 20 words. The field is named one_sentence for historical
reasons; more than one is correct when one idea per sentence needs more.

Describe the work. Do not assess it. The summary must not say whether the paper
is relevant, novel, significant, rigorous, empirical or implementable, and must
not explain why it was selected. That is what significance and novelty are for,
and neither is shown to the reader. A summary that argues for the paper is
wrong even when the argument is correct.

Return a judgement for every paper you are given, keyed by its exact arxiv_id.
The arxiv_id is the value stated on the line directly above each fenced
document, in the form "arxiv_id: <value>". It is not the fence's nonce
attribute; that nonce is a security marker, not a paper identifier, and must
never appear in your output.

""" + PLAIN_ENGLISH + "\n" + DATA_NOT_INSTRUCTIONS

SCHEMA = {
    "type": "object",
    "properties": {
        "judgements": {
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
    "required": ["judgements"],
    "additionalProperties": False,
}


def _render(candidates: list[Candidate]) -> str:
    """One fenced block per paper. See screen.py on why the id sits outside."""
    blocks = []
    for c in candidates:
        title = sanitize(c.paper.title, MAX_TITLE_CHARS)
        abstract = sanitize(c.paper.abstract, MAX_ABSTRACT_CHARS)
        body, _ = fence(f"title: {title}\nabstract: {abstract}")
        blocks.append(f"arxiv_id: {c.paper.arxiv_id}\n{body}")
    return "\n\n".join(blocks)


def judge_candidates(
    client: ModelClient, profile: str, candidates: list[Candidate]
) -> tuple[dict[str, dict], list[str]]:
    """Judge every candidate. Returns (judgements by arxiv_id, list of failures).

    A candidate the model omits is reported rather than defaulted: a paper with
    an invented score would compete for a slot it never earned.
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
            label="call 2 (judge)",
        )
    except ModelError as exc:
        log.error("judging call failed: %s", exc)
        return {}, [f"judging call failed: {exc}"]

    judgements: dict[str, dict] = {}
    wanted = {c.paper.arxiv_id for c in candidates}
    rows = data.get("judgements", [])
    log.info(
        "call 2 (judge): got %d judgement row(s) for %d wanted paper(s); "
        "sample returned ids=%s, sample wanted ids=%s",
        len(rows), len(wanted),
        [str(r.get("arxiv_id", "")).strip() for r in rows][:5], sorted(wanted)[:5],
    )
    for row in rows:
        aid = str(row.get("arxiv_id", "")).strip()
        if aid in wanted:
            judgements[aid] = {
                "significance": int(row["significance"]),
                "novelty": int(row["novelty"]),
                "one_sentence": str(row.get("one_sentence", "")).strip(),
            }

    problems = []
    missing = sorted(wanted - set(judgements))
    if missing:
        problems.append(
            f"judging call returned no judgement for {len(missing)} paper(s): "
            f"{', '.join(missing)}"
        )
        log.warning(problems[-1])

    return judgements, problems


def rank(candidates: list[Candidate], judgements: dict[str, dict],
         top_n: int) -> list[Candidate]:
    """The `top_n` best-judged candidates.

    Significance and novelty are summed with significance breaking ties, so a
    highly novel paper the researcher does not care about loses to a merely
    solid one that they do. Similarity is the last tie-break, which keeps the
    ordering deterministic for a given day.
    """
    judged = [c for c in candidates if c.paper.arxiv_id in judgements]
    judged.sort(
        key=lambda c: (
            judgements[c.paper.arxiv_id]["significance"]
            + judgements[c.paper.arxiv_id]["novelty"],
            judgements[c.paper.arxiv_id]["significance"],
            c.similarity,
        ),
        reverse=True,
    )
    return judged[:top_n]
