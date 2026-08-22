"""Call 1: is this paper relevant at all?

A cheap model reads every paper that survived the pre-sort and answers one
binary question per paper. This is the step that replaced the similarity
filter, because similarity could not answer it.

Papers go in batches, not one per call. A relevance judgement is comparative.
The question "is this the kind of thing the profile asks for" is easier to
answer with two dozen same-day papers in view. Batching also costs a fraction
of one call per paper.

A batch that fails costs its own papers, not the day. The remaining batches
still run and the failure is reported.
"""

from __future__ import annotations

import logging

from .guard import (DATA_NOT_INSTRUCTIONS, MAX_ABSTRACT_CHARS, MAX_TITLE_CHARS,
                    fence, sanitize)
from .llm import ModelClient, ModelError
from .select import Candidate

log = logging.getLogger(__name__)

SYSTEM = """\
You screen new arXiv papers for one researcher, whose profile is given below.

For each paper, answer one question: could this paper plausibly matter to that
researcher, given the profile?

Say yes when the paper is about the systems the profile describes -- many
agents, populations, economies, markets, organisations, societies -- or about
measuring, steering or failing at that scale.

Governance and policy work counts as steering. Say yes to regulation, legal
analysis, standards and institutional design when the subject is agent
populations or their effects at scale. A paper arguing how AI agents should be
regulated is in scope. A paper on compliance tooling for one deployed model is
not: the test is the scale of the subject, not whether the paper proposes
rules.

Say no when the paper is about making one model or one agent better: a training
method, a decoding trick, a retrieval or context technique, a single-agent
benchmark, a jailbreak or attack method, a domain application with no
population dynamics. These are the bulk of any day's listing. Most papers are a
no, and a day where almost everything is a no is a normal day, not a failed
screen. Do not stretch to find yeses.


Judge the papers against the profile, and against each other: you are seeing
one batch from a single day, so use the batch to calibrate what a strong yes
looks like relative to a weak one.

Give a reason of at most 15 words. State what the paper is about and why it
does or does not fit -- no hedging, no restating the question.

Return a verdict for every paper you are given, keyed by its exact arxiv_id.
The arxiv_id is the value stated on the line directly above each fenced
document, in the form "arxiv_id: <value>". It is not the fence's nonce
attribute; that nonce is a security marker, not a paper identifier, and must
never appear in your output.

""" + DATA_NOT_INSTRUCTIONS

SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string"},
                    "relevant": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["arxiv_id", "relevant", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}


def _render(candidates: list[Candidate]) -> str:
    """One fenced block per paper.

    The arxiv_id sits outside the fence. It is the key the model must return.
    The caller validates it against the batch. Keeping it outside means no
    abstract can appear to relabel another paper's id.
    """
    blocks = []
    for c in candidates:
        title = sanitize(c.paper.title, MAX_TITLE_CHARS)
        abstract = sanitize(c.paper.abstract, MAX_ABSTRACT_CHARS)
        body, _ = fence(f"title: {title}\nabstract: {abstract}")
        blocks.append(f"arxiv_id: {c.paper.arxiv_id}\n{body}")
    return "\n\n".join(blocks)


def _batches(candidates: list[Candidate], size: int) -> list[list[Candidate]]:
    size = max(1, size)
    return [candidates[i : i + size] for i in range(0, len(candidates), size)]


def screen_candidates(
    client: ModelClient,
    profile: str,
    candidates: list[Candidate],
    batch_size: int = 25,
) -> tuple[dict[str, dict], list[str]]:
    """Screen every candidate. Returns (verdicts by arxiv_id, list of failures).

    A paper the model omits is reported rather than defaulted. Defaulting to
    relevant would push unjudged papers into the expensive call; defaulting to
    irrelevant would drop a paper silently. Neither is honest, so the caller is
    told and decides.
    """
    if not candidates:
        return {}, []

    verdicts: dict[str, dict] = {}
    problems: list[str] = []
    batches = _batches(candidates, batch_size)
    log.info("screening %d paper(s) in %d batch(es) of up to %d",
             len(candidates), len(batches), batch_size)

    for i, batch in enumerate(batches, start=1):
        label = f"call 1.{i} (screen)"
        user = (
            f"RESEARCHER PROFILE\n{profile}\n\n"
            f"PAPERS ({len(batch)})\n\n{_render(batch)}"
        )
        try:
            data = client.structured(
                system=SYSTEM,
                user=user,
                schema=SCHEMA,
                max_tokens=8000,
                label=label,
            )
        except ModelError as exc:
            msg = (f"screening batch {i}/{len(batches)} failed ({len(batch)} paper(s) "
                   f"unscreened): {exc}")
            log.error(msg)
            problems.append(msg)
            continue

        wanted = {c.paper.arxiv_id for c in batch}
        rows = data.get("verdicts", [])
        for row in rows:
            aid = str(row.get("arxiv_id", "")).strip()
            if aid in wanted:
                verdicts[aid] = {
                    "relevant": bool(row["relevant"]),
                    "reason": str(row.get("reason", "")).strip(),
                }

        missing = sorted(wanted - set(verdicts))
        if missing:
            msg = (f"screening batch {i}/{len(batches)} returned no verdict for "
                   f"{len(missing)} paper(s): {', '.join(missing)}")
            log.warning(msg)
            problems.append(msg)

        n_yes = sum(1 for c in batch
                    if verdicts.get(c.paper.arxiv_id, {}).get("relevant"))
        log.info("%s: %d/%d relevant", label, n_yes, len(batch))

    total_yes = sum(1 for v in verdicts.values() if v["relevant"])
    log.info("screen: %d of %d screened paper(s) relevant",
             total_yes, len(verdicts))
    return verdicts, problems


def relevant(
    candidates: list[Candidate], verdicts: dict[str, dict]
) -> list[Candidate]:
    """The candidates the screen said yes to, in similarity order."""
    return [c for c in candidates
            if verdicts.get(c.paper.arxiv_id, {}).get("relevant")]
