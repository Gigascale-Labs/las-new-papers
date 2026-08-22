"""The house style for every word a model writes into the digest.

Two calls produce reader-facing text: the judge writes each paper's summary,
and the question call writes the questions and their reasons. Both import the
same spec from here, because a style rule that says "use the same word for the
same thing" cannot itself be worded two different ways in two prompts.

Nothing in the digest justifies a paper's presence, so the summary and the
questions carry the whole case for reading it. That is why they are held to a
standard rather than left to the model's default register, which reaches for
"novel", "significant" and "state-of-the-art" without being asked.
"""

from __future__ import annotations

PLAIN_ENGLISH = """\
WRITING STYLE

Write everything in Simplified Technical English:

- Short sentences. One idea in each. If a sentence needs a comma to hold two
  claims together, make it two sentences.
- Active voice. Present tense. "The model treats agents as consumers", not
  "agents were treated as consumers".
- The same word for the same thing, every time. Do not vary wording for
  interest. If the paper says "agent", say "agent" throughout, not "actor"
  then "participant".
- State the facts and the numbers the paper gives. Do not characterise them.
  "GDP grows without bound" is a fact; "GDP grows explosively" is a
  characterisation.
- No adjective or adverb of praise, size or surprise: not novel, significant,
  substantial, dramatic, explosive, remarkable, state-of-the-art, powerful.
  No metaphor. No filler. Never stack one hedge on another.
- Do not write "this paper", "the authors", "we". Start with the subject or
  the verb.
- If the abstract does not say, say less. Never guess to fill a sentence, and
  never pad a short true answer into a long vague one.
"""
