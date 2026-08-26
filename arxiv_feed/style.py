"""The house style for every word a model writes into the digest.

Three callers write reader-facing text. The judge writes each paper's summary.
The question call writes the questions and their reasons. `scripts/
restyle_descriptions.py` rewrites summaries already published, to this same
spec. All three import it from here. A rule that says "use the same word for
the same thing" cannot be worded three ways in three prompts.

Nothing in the digest justifies a paper's presence. The summary and the
questions carry the whole case for reading it. So they take a stated standard.
The model's default register reaches for "novel", "significant" and
"state-of-the-art" unprompted.
"""

from __future__ import annotations

# Three clauses of the house ruleset are not passed through to these calls,
# and one is passed through in a reduced form. Both fields are short prose: a
# summary of one to three sentences, and a question of one sentence.
#
# "State every number with its n and its spread" -- reduced. The block below
# asks for the numbers the paper gives, with whatever the paper attaches to
# them, and forbids supplying any it does not. A summary required to carry an
# n would invent one for every abstract that gives none, and an invented n is
# worse than no number.
#
# "Use a table for three or more parallel items" -- dropped. There is nowhere
# to put one. Both fields are plain strings; render.py wraps a summary in a
# <div> and feed.py escapes it, so a model that drew a table would put pipes
# and dashes in the middle of a sentence.
#
# "Avoid claims in visualisation titles", "label axes and provide keys", "do
# not overlay two elements in the same colour" -- dropped. Neither call draws
# anything. Nothing in this pipeline emits a chart.
#
# "Name the test that would settle it" is not repeated here either: it is the
# question call's whole job, and a summary is not the place to run it.

PLAIN_ENGLISH = """\
WRITING STYLE

Write everything in Simplified Technical English:

- Answer in the first sentence. Say what the work does before any setup,
  method or caveat. The detail comes after it, never instead of it.
- Short sentences. One idea in each. If a sentence needs a comma to hold two
  claims together, make it two sentences.
- Active voice. Present tense. Never the passive. "The model treats agents as
  consumers", not "agents were treated as consumers"; "measures a 12-point
  drop", not "a 12-point drop was measured". Dropping a weak opener must not
  cost the active voice: cut "This paper shows that X does Y" to "X does Y",
  never to "Y is done by X".
- The same word for the same thing, every time. Do not vary wording for
  interest. If the paper says "agent", say "agent" throughout, not "actor"
  then "participant".
- Give facts, not justifications. State the facts and the numbers the paper
  gives, with whatever the paper attaches to them: how many cases, over what
  range, against which baseline. Do not characterise them. "GDP grows without
  bound" is a fact; "GDP grows explosively" is a characterisation.
- Never supply a number, a sample size or a range the paper does not state. A
  missing number is a fact about the paper; an invented one is a false claim
  about it.
- Say which kind of claim it is, in the paper's own terms: measured, observed,
  inferred from a model, or assumed. "Assumes every agent sees one price" and
  "measures a 12-point drop" are different claims and must not read alike.
- Name what the paper does not check, where the paper itself names it: a
  condition left untested, a parameter held fixed, a population it never ran
  on.
- No adjective or adverb of praise, size or surprise: not novel, significant,
  substantial, dramatic, explosive, remarkable, state-of-the-art, powerful.
  No metaphor. No filler. Never stack one hedge on another.
- Do not write "this paper", "the authors", "we". Start with the subject or
  the verb.
- If the abstract does not say, say less. Where you do not know, say the paper
  does not say it. Never guess to fill a sentence, and never pad a short true
  answer into a long vague one.
"""
