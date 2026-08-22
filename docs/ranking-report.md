# Why the similarity ranking stopped choosing papers

Measured on system-1 on 2026-08-22. Source: the archived listing for
2026-08-20 (`data/raw/2026-08-20.jsonl.gz`, n = 185 papers) and the 20-anchor
set from PR #7.

Tables and charts: <https://claude.ai/code/artifact/dd57e1dd-49c0-4a95-9a74-492c75a0498e>

## The short version

The old ranking produces crap. Nine of its top ten papers are off-profile.

The anchors are not the fault. The anchors pass their own test. The ranking
built on them is the fault. It decides what reaches a model. It cannot make
that decision.

## Measured

**The anchors pass.** Leave-one-out runs over all 20 anchors. 19 of 20 return
inside the cut. Median rank is 1.5. 18 of 20 return inside the top 10.

**Narrowing cost nothing.** Four anchors are testable under both the old
33-anchor set and the new 20. All four rank 1st under both.

**The ranking cannot discriminate.** Cosine over the 185 papers spans 0.803 to
0.960. The rank-40 cut sits at 0.911. Ranks 10 to 40 hold 30 papers. Those 30
span 0.011 in total. That is 0.0004 per paper.

**One anchor dominates.** `2602.16136` (*Retrieval Collapses When AI Pollutes
the Web*) wins 71 of 185 papers. It wins 21 of the 40 shortlist slots. It is
also the only leave-one-out failure, at rank 59 of 274.

**Both facts share one cause.** That anchor sits furthest from the rest of the
set. Its cohesion is 0.8822 against a set mean of 0.9163 (n = 20, range 0.8822
to 0.9363). `best_match` takes the max over anchors. So the anchor owns its
region of the embedding space unopposed. Across the 20 anchors, cohesion
correlates −0.676 with papers won.

**The top 10 is crap.** It holds retrieval-augmentation, prompt compression,
uncertainty estimation, jailbreak and malicious-skill benchmark papers. A
keyword proxy scores 1 of the 10 as on-theme. The profile asks for none of the
other nine.

**Pooling is not the cause.** A top-3 mean, a top-5 mean and a full mean each
give 0 or 1 on-theme papers in the top 10. Max gives 1. Changing how anchor
scores combine changes nothing.

## The control

Count what the day holds before judging the filter. Of 185 papers, 1 scores 3
or more on the keyword proxy. 2 score 2 or more.

The strongest on-theme paper is *Growth Without Us: Machine Consumers,
Corporate Circularity*. The filter ranks it 9th. That is inside the shortlist.

So the filter finds what the day holds. It does not miss good papers. It fills
the other nine slots with crap. It cannot tell the difference. That is the
defect this change fixes.

## What follows

Similarity says a paper is about roughly the right subject. It cannot say
whether the paper suits a stated profile. It cannot say whether the paper is
good. It cannot say whether the paper is new. It was asked the first question
and could not answer it.

A cheap model screens 200 papers for about $0.10 a day. The first live run
measures 380 tokens per paper (n = 155). That is about what the single Opus
call cost to read 45 papers. Reading every paper is now cheap. A 0.0004 margin
no longer decides what gets read.

So the anchors are demoted, not deleted. They order a day larger than
`screen_n` so it can be cut to `screen_n`. On a day under the cap they change
nothing.

`tests/leave_one_out.py` still measures that ordering. It now measures a
narrower thing. A failure means a paper could be pushed past the cap on a heavy
day. It no longer means the paper is dropped outright.

## The live run

The cascade ran against real papers on 2026-08-22. It read the 2026-08-20
listing: 185 fetched, 155 unseen, 155 screened, 2 relevant, 2 kept, 11
questions. Ten model calls. No failures. No retries.

| Stage | Model | Calls | In tok | Out tok | Reasoning | Cost |
|---|---|---:|---:|---:|---:|---:|
| screen | haiku-4.5 | 7 | 66,767 | 16,631 | 9,824 | $0.1499 |
| judge | sonnet-5 | 1 | 2,849 | 427 | 216 | $0.0100 |
| questions | opus-5 | 2 | 6,433 | 1,846 | 72 | $0.0783 |
| total | | 10 | 76,049 | 18,904 | 10,112 | $0.2382 |

The screen kept a paper that similarity ranked **109th**: *A Privacy Budgeting
Framework for Online Experimentation*. The old filter kept ranks 1 to 40 plus 5
drawn at random from ranks 41 to 140. It had about a 5 in 100 chance of
surfacing that paper.

The keyword proxy scored that paper as off-theme. It uses none of the 18 theme
words. The screen read the abstract and saw population-level mechanism design.

Reasoning was 9,824 of the screen's 16,631 output tokens. Haiku output costs
$5 per MTok. So reasoning cost about $0.049 of the $0.150 screen bill, to
answer one yes/no question per paper. The screen now runs at its own effort,
`screen_effort: low`.

## Recall

The 20 anchors are relevant by construction. Feeding them to the screen
measures recall on known positives.

Two runs, both 18 of 20. The two runs reject different papers:

| Run | Rejected |
|---|---|
| 1 | `2503.13395` Causal Emergence 2.0, `2603.23471` Regulating AI Agents |
| 2 | `2503.13395` Causal Emergence 2.0, `2602.11865` Intelligent AI Delegation |

The `Regulating AI Agents` reject drove a prompt change: governance and policy
work now counts as steering, and the screen says yes to it.

One anchor flipped between two identical runs. `llm.py` sets no temperature, so
these calls run at the provider default. A borderline paper is decided partly
by chance. Each paper is screened once in a real run, so the variance never
causes a duplicate or a double miss.

`2503.13395` fails both runs. That is a real disagreement between the canon and
the screen, not noise.

## Not checked

- **The top-10 path has never run live.** Two papers is the most that has
  passed the screen. Ranking more candidates than `top_n` runs in unit tests
  only.
- **Lakera has now run, and is off.** CI ran it on 2026-08-22. It blocked 19
  of 155 screened papers. One was `2608.20231` (*Growth Without Us*), the
  control case named above, flagged `prompt_attack`. The two papers that
  reached the digest instead were ride-hailing dispatch and a lead
  service-line audit. `guard.enabled` is now `false`. The measurement is
  n = 1 day; the false-positive rate is not established, only that it is not
  zero on the paper that matters most.
- **No email was sent.** Every run so far used `--dry-run`.
- **The on-theme measure is a keyword proxy.** It counts 18 theme words in
  title and abstract. It is not the model's judgment. It misses on-theme papers
  that use other words, and the rank-109 paper proves it.
- **n = 1 day.** Only 2026-08-20 is usable. The 2026-08-21 archive holds 0
  records.
- **Sonnet 5 replaces Opus 5 on judging.** That is a capability change, not
  only a cost change. It is unevaluated.
