# Why the similarity ranking stopped deciding what you read

Measured on system-1, 2026-08-22, against the archived listing for 2026-08-20
(`data/raw/2026-08-20.jsonl.gz`, 185 papers) and the 20-anchor set from PR #7.

This is the evidence behind the two-stage model cascade that replaced the
similarity filter. Full report, with the tables:
<https://claude.ai/code/artifact/dd57e1dd-49c0-4a95-9a74-492c75a0498e>

## The short version

The old ranking produced crap. Nine of its top ten papers were off-profile.

The anchor set is not the fault. The anchors pass their own test. The ranking
built on them is the fault. It decides what reaches a model, and it cannot make
that decision.

## What was measured

**The anchors themselves pass.** Leave-one-out over all 20 new anchors: 19 of 20
returned inside the cut, median rank 1.5, 18 of 20 inside the top 10. The four
anchors testable under both the old 33-anchor set and the new 20 ranked 1st under
both, so narrowing the set cost nothing for the anchors that stayed.

**The ranking has almost no discrimination where it matters.** Over the 185
papers, SPECTER2 cosine spans 0.803 to 0.960. The rank-40 cut sits at 0.911.
Ranks 10 through 40 — thirty papers, most of the old shortlist — are separated by
0.011 in total, about 0.0004 each. Decisions resting on differences that small
are close to arbitrary.

**One anchor dominated the result.** `2602.16136` (*Retrieval Collapses When AI
Pollutes the Web*) won 71 of 185 papers and 21 of the 40 shortlist slots. It is
also the single leave-one-out failure, at rank 59 of 274. Both facts have one
cause: it sits furthest from the rest of the set (cohesion 0.8822 against a set
mean of 0.9163), and `best_match` takes the max, so it owns its region of the
embedding space unopposed. Across the 20 anchors, cohesion correlates −0.676 with
papers won.

**The old top 10 is crap.** It holds retrieval-augmentation, prompt
compression, uncertainty estimation, jailbreak and malicious-skill benchmark
papers. A keyword proxy over title and abstract scores 1 of the 10 as on-theme.
The profile asks for none of the other nine.

**Changing the pooling does not fix it.** Replacing max with a top-3 mean, a
top-5 mean, or a full mean over all 20 anchors gives 0 or 1 on-theme papers in
the top 10 — the same as max. The problem is not how anchor scores are combined.

## The control that matters

Before concluding the filter was failing, count what was available to find. Of
the 185 papers that day, **1 scored 3 or more on the keyword proxy and 2 scored
2 or more**. The single strongest on-theme paper (*Growth Without Us: Machine
Consumers, Corporate Circularity*) was ranked **9th** by the new anchors, well
inside the shortlist.

So the filter finds what the day holds. It does not miss good papers. It fills
the other nine slots with crap, and it cannot tell the difference. That is the
defect this PR fixes.

## What follows from this

Similarity can say a paper is about roughly the right subject. It cannot say
whether the paper is relevant to a stated profile, whether it is good, or whether
it is new. It was being asked to do the first of those and could not.

Screening 200 papers with a cheap model costs about $0.10 a day — measured at
408 tokens per paper across the 150 top-ranked papers of that listing. That is
roughly what the single Opus scoring call cost when it read only 45 papers. Once
reading every paper is that cheap, there is no reason to let a 0.0004-cosine
margin decide what gets read.

So the anchors were demoted rather than deleted. They order a day larger than
`screen_n` so it can be cut to `screen_n`, and on a day under the cap they change
nothing. `tests/leave_one_out.py` still measures that ordering, but it now
measures a narrower thing: a failure means a paper could be pushed past the cap
on a heavy day, not that it would be dropped outright.

## What was not checked

- **The two model calls have never run live.** No `OPENROUTER_API_KEY` was
  available on the machine this was measured on, so the screen and judge prompts
  are tested against stubs only.
- **One day, n = 1.** Only 2026-08-20 is usable; the 2026-08-21 archive is empty.
- **The on-theme measure is a keyword proxy** over 18 theme words in title and
  abstract. It is a signal about the filter, not a substitute for the model's
  judgment, and it will miss on-theme papers that use other vocabulary.
- **Sonnet 5 replacing Opus 5 on the judging call is a capability change**, not
  only a cost change. It has not been evaluated.
