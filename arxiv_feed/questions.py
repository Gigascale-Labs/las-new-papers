"""Call 2: the open questions in one paper, and its canon tags.

One call per kept paper. Both jobs read the same abstract.

Abstracts only, never the PDF, so every tag is `summary-only` confidence.
"""

from __future__ import annotations

import logging

from . import canon
from .arxiv import Paper
from .guard import (DATA_NOT_INSTRUCTIONS, MAX_ABSTRACT_CHARS, MAX_TITLE_CHARS,
                    fence, sanitize)
from .llm import ModelClient, ModelError

log = logging.getLogger(__name__)

SYSTEM = """\
You read one paper's abstract and report the open questions it leaves.

An open question is something the paper does not answer and that someone could
work on next. Take them from what the paper itself says is unresolved, limited,
or future work, and from what its claims plainly leave open. Do not invent
questions the abstract gives you no basis for, and do not pad: two sharp
questions beat six vague ones. If the abstract genuinely supports none, return
an empty list.

Write each question so it can be read on its own, without the paper's title
next to it.

Label each question for the researcher whose profile is given:

approachable -- they could make real progress on it within a few weeks using
public data, public APIs, or a simulation they could write themselves, with the
skills their profile claims.

not approachable -- it needs resources or access the profile does not have
(frontier-scale compute, a large team, proprietary platform data, human
subjects), or skills they say they lack.

The reason is one short clause, concrete about which resource or skill decides
it. "Needs proprietary trading data" is useful; "may be difficult" is not.

You also tag the paper for a research canon. The dimension lists are closed:
use only the given values, and leave a dimension empty when the abstract does
not clearly support any value. An empty dimension is a normal, correct answer --
several hand-tagged papers in this canon have them. Guessing is worse than
leaving it blank.

The summary is one to three sentences describing what the paper does, factual
and free of adjectives, written the way a catalogue entry is written.

""" + DATA_NOT_INSTRUCTIONS


def _schema(tag_vocab: list[str]) -> dict:
    dim = lambda values: {                                   # noqa: E731
        "type": "array",
        "items": {"type": "string", "enum": values},
    }
    vocab_hint = (
        f" Reuse an existing tag where one fits: {', '.join(tag_vocab)}."
        if tag_vocab else ""
    )
    return {
        "type": "object",
        "properties": {
            "open_questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "label": {"type": "string", "enum": ["approachable", "not approachable"]},
                        "reason": {"type": "string"},
                    },
                    "required": ["question", "label", "reason"],
                    "additionalProperties": False,
                },
            },
            "canon": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "One to three topical tags." + vocab_hint,
                    },
                    "system_type": dim(canon.SYSTEM_TYPES),
                    "participant_mix": dim(canon.PARTICIPANT_MIXES),
                    "observability": dim(canon.OBSERVABILITY_LEVELS),
                    "focus_area": dim(canon.FOCUS_AREAS),
                    "threat_model": dim(canon.THREAT_MODELS),
                    "claim_type": dim(canon.CLAIM_TYPES),
                },
                "required": [
                    "summary", "tags", "system_type", "participant_mix",
                    "observability", "focus_area", "threat_model", "claim_type",
                ],
                "additionalProperties": False,
            },
        },
        "required": ["open_questions", "canon"],
        "additionalProperties": False,
    }


def extract(client: ModelClient, profile: str, paper: Paper,
            tag_vocab: list[str] | None = None) -> dict:
    """Questions and canon tags for one paper.

    Raises ModelError if the call failed twice -- the caller drops that paper and
    marks it in the email. One bad paper never stops the run.
    """
    tag_vocab = canon.known_tags() if tag_vocab is None else tag_vocab
    body, _ = fence(
        f"title: {sanitize(paper.title, MAX_TITLE_CHARS)}\n"
        f"authors: {sanitize(', '.join(paper.authors), MAX_TITLE_CHARS)}\n"
        f"abstract: {sanitize(paper.abstract, MAX_ABSTRACT_CHARS)}"
    )
    user = (
        f"RESEARCHER PROFILE\n{profile}\n\n"
        f"PAPER (arxiv_id: {paper.arxiv_id})\n{body}"
    )
    data = client.structured(
        system=SYSTEM,
        user=user,
        schema=_schema(tag_vocab),
        max_tokens=8000,
        label=f"call 2 ({paper.arxiv_id})",
    )

    questions = []
    for q in data.get("open_questions", []):
        text = str(q.get("question", "")).strip()
        if not text:
            continue
        label = str(q.get("label", "")).strip()
        questions.append(
            {
                "question": text,
                # Anything not exactly "approachable" is treated as not
                # approachable: the cost of a wrongly encouraging label is a
                # wasted week.
                "label": "approachable" if label == "approachable" else "not approachable",
                "reason": str(q.get("reason", "")).strip(),
            }
        )

    return {"open_questions": questions, "canon": data.get("canon", {}) or {}}


__all__ = ["extract", "ModelError"]
