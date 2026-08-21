# las-new-papers

One email a day: new arXiv papers close to work you already value, and the open
questions each of them leaves — with a note on whether you could actually work
on it.

Standalone, like [`las-usage-stats`](https://github.com/Gigascale-Labs/las-usage-stats):
it scrapes, it commits its own data, and largeagentsystems.org reads that data
over `raw.githubusercontent.com` if and when it wants to. Nothing here writes to
the site's Airtable canon.

Built to `docs/spec.md` (arXiv open questions feed, v2).

## How a day runs

```
arXiv (~300-600 new papers in 7 lists)
  └─ drop anything already sent            seen.json
  └─ embed title + abstract locally        SPECTER2, free, a few seconds
  └─ cosine vs 33 anchor papers            top 40 + 5 random from ranks 41-140
  └─ ONE model call scores 45 papers       significance 1-5, novelty 1-5
  └─ keep the top 10
  └─ ONE model call per paper              open questions + canon tags
  └─ email, then data/YYYY-MM-DD.json
```

Two properties of that pipeline are worth stating plainly, because they are the
whole design:

**The embedding filter is a filter, not a judge.** Cosine similarity says a
paper is *about* what your anchors are about. It cannot tell you whether the
paper is any good or whether it has been done before — only the model calls do
that, and they only ever see 45 papers instead of 500. That is what makes the
whole thing cost ~$0.10 a day.

**Similarity alone would tunnel-vision.** It can only find more of what the
anchors already describe, so a genuinely new subfield — the thing you most want
to hear about — has no anchor to be near and would never surface. The 5 random
papers from just below the cut are the fix. They are marked `[random]` in the
email so you can see whether they ever earn their place.

Similarity is scored as the **maximum** over single anchors, never the mean. The
anchor set deliberately spans simulation, market design, governance and safety;
the mean of those directions points at no real paper, and a market-design paper
should be allowed to score on the market-design anchors alone.

## Setup

```bash
pip install -r requirements.txt          # ~2.5GB: torch comes with sentence-transformers
export ANTHROPIC_API_KEY=sk-ant-...      # model calls
export SMTP_PASSWORD=...                 # Gmail app password, not your account password

python main.py --dry-run                 # full run, writes the JSON, sends nothing
python main.py                           # and send the email
python main.py --date 2026-08-19         # re-read a specific UTC day
python main.py --rebuild-anchors         # force the anchor vectors to rebuild
```

The first run downloads the embedding model (~440MB) and builds the anchor
vectors (one arXiv call plus a few seconds of CPU). Both are cached afterwards.

Everything else lives in `config.yaml`: categories, anchors, your profile,
`shortlist_n` / `explore_n` / `top_n`, the two model names, and the email
address. The anchor vectors rebuild by themselves whenever the anchor list or
the embedding model changes — there is nothing to remember to re-run.

### Daily, without a machine of your own

`.github/workflows/daily_feed.yml` runs the whole thing at 07:23 UTC and commits
that day's archive back to the repo. It needs two repository secrets,
`ANTHROPIC_API_KEY` and `SMTP_PASSWORD`. The embedding model and the anchor
vectors are cached between runs, so a normal day is a couple of minutes of CPU.

The commit step runs even when the run failed: a day that produced its JSON but
could not send the email still exits non-zero (a red X you will notice) while
keeping the data.

## What it writes

| Path | What it is |
|---|---|
| `data/YYYY-MM-DD.json` | The day's full record: every kept paper, its scores, its questions, and every problem hit along the way. |
| `data/latest.json` | A copy of the most recent day, at a stable URL for the site to read. |
| `data/canon/finalists.csv` | Every finalist ever kept, tagged in the site canon's schema. |
| `data/seen.json` | arXiv IDs already sent. Nothing is sent twice. |
| `data/eval/leave-one-out.json` | The most recent retrieval evaluation (see Tests). |
| `data/ground-truth/` | A frozen copy of the human canon. Read on, this one matters. |

## The canon: which copy is the source of truth

Short answer, since it decides what this repo copies: **Airtable is upstream,
the JSON is what the site reads, and the CSV is a fallback you should not treat
as authoritative.**

- **`Canon` table in Airtable** (base `apps8rBIORsmE7ij8`, table
  `tbl2XEeh8Rlnrlw0j`) — where edits actually happen, and where the
  contribute-a-source queue promotes approved rows into. This is the source of
  truth.
- **`data/las-canon.airtable.json`** in the site repo — a daily sync of that
  table, committed to git. `lib/canon-data.ts` reads *this* first, so it is what
  the live site renders.
- **`data/las-canon.csv`** — the original hand-built corpus, now a fallback that
  is only read if the JSON is missing, empty or unparseable. It has known
  mojibake in several `summary` values that the Airtable copies do not have, and
  nothing pushes CSV edits back to Airtable. `docs/las-canon-addendum.md` in the
  site repo says to keep it until the Airtable path has been stable for a while
  — keep it, but don't copy from it.

So `data/ground-truth/las-canon-frozen.csv` here is built from the **Airtable
JSON**, in the CSV's exact column order. `provenance.json` records the source
commit, the row count (42), and the date it was taken.

It is **frozen on purpose**. It is the evaluation set for the retrieval filter,
and an evaluation set that quietly tracks upstream makes every earlier
measurement incomparable. Refresh it deliberately, as its own commit, when you
want to evaluate against a newer canon.

The 33 anchors in `config.yaml` are the arXiv-hosted entries of that frozen
canon. The other 9 are government reports, SSRN, MDPI and an FHI tech report —
no arXiv ID to anchor on.

## Finalists arrive in the canon's schema

Every paper that survives to the top 10 is tagged in the same six dimensions the
human canon uses (`system_type`, `participant_mix`, `observability`,
`focus_area`, `threat_model`, `claim_type`), from the same closed choice lists,
and appended to `data/canon/finalists.csv`. The first 15 columns are the canon's
columns in the canon's order; this repo's own columns (`arxiv_id`, `similarity`,
`nearest_anchor_id`, `significance`, `novelty`, …) come after them.

That file is a **proposal, not a promotion**. Nothing here writes to Airtable or
to the site. A human still decides what is admitted — the point of the shared
schema is that admitting a paper is a copy rather than a re-tagging job.

Two honest limits on those tags: they are read from the abstract only, so every
row is `tag_confidence: summary-only`, exactly like the hand-tagged canon rows;
and `institutions` is left blank, because an abstract does not carry
affiliations and a guessed affiliation is worse than an empty cell. A value that
comes back off the closed list is dropped rather than coerced — an empty
dimension is a legitimate state in the canon, and several hand-tagged rows have
them.

The choice lists are mirrored by hand in `arxiv_feed/canon.py` from
`lib/canon-schema.ts` in the site repo. There is no import path between a
TypeScript file in one repo and a Python file in another, so a change there
needs a change here.

## Tests

```bash
python -m unittest discover tests          # 27 unit tests, no network, no API key
python -m tests.leave_one_out              # the retrieval evaluation, real arXiv
```

The unit tests cover the parts that decide what you read: that the shortlist is
40 by similarity plus 5 from ranks 41-140 and never pads a thin day; that
similarity is max-over-anchors rather than mean; that ranking is by significance
+ novelty with significance breaking ties; that a paper the scoring call omits
is reported rather than given an invented score; that a hallucinated arXiv ID is
dropped; that any label other than exactly `approachable` becomes "not
approachable"; that bad JSON is retried exactly once and then gives up; that a
corrupt seen-list starts empty instead of stopping the run; and that failures
reach the email instead of vanishing.

`tests/leave_one_out.py` is the spec's test 5, and it is the one that says
whether the filter works at all. For each anchor tested it removes that anchor,
fetches every paper submitted to the configured categories on the day that
anchor appeared, ranks them against the remaining anchors, and checks the
held-out paper still lands in the top 40. Threshold: 8 of 10.

**It has been run: 10 of 10 passed** (`data/eval/leave-one-out.json`, seed 0,
`allenai/specter2_base`). Seven of the ten came back at rank 1 in day pools of
63-311 papers; the worst was rank 28 of 166 (zkLLM, a zero-knowledge-proofs
paper — the most methodologically distant anchor in the set), comfortably
inside the top 40. One anchor (`2408.02784`) was not in its own day's pool
because its primary list is outside the configured categories, so it was
injected before ranking; the report marks it `injected_into_pool`.

## Cost

| Step | Cost |
|---|---|
| arXiv API | free |
| Embedding ~500 papers | free, a few seconds of your own CPU |
| Call 1 — score 45 papers | one call/day |
| Call 2 — questions | one call per kept paper, ~10/day |

About US$0.10 a day at Claude Opus 5 rates, so roughly US$3 a month. The Batch
API would halve call 1 and is **not implemented** — the spec lists it as
optional, and polling for a batch result adds a failure mode to a job whose
whole point is arriving once a day without supervision.

## Deliberate departures from the spec

- **Not started from the ArxivDigest fork.** The spec suggests forking it and
  adding five things. What is here is those five things (embedding step, anchor
  store, similarity filter, structured model calls, seen list) written directly,
  which is a smaller amount of code than the fork's web UI, per-user config and
  scheduler would have left to delete. Nothing else about the design changed.
- **`econ` expanded to `econ.GN`, `econ.TH`, `econ.EM`.** arXiv has no bare
  `econ` list to query.
- **The day read is yesterday, UTC.** A 07:23 run asking for "today" would see
  only the few hours of today that exist yet and would silently lose the rest of
  the day forever.
- **The email's `[random]` mark and the nearest-anchor line are load-bearing.**
  Both are in the spec; both are also how you tell whether the anchor set is
  drifting, so neither is decoration to trim.
- **Structured output via `output_config.format`, not tool use.** The spec says
  "use structured output"; the JSON-schema response format is the current way to
  do that, and it constrains generation rather than asking politely for JSON.
  The retry-once-on-bad-JSON rule is implemented anyway, because schema
  enforcement does not cover a refusal or a response truncated at `max_tokens`.

## Known state

- The **model calls have not been run against the live API**. The environment
  this was built in has no `ANTHROPIC_API_KEY`, so calls 1 and 2 are covered by
  unit tests against a scripted client, not by a real request. Everything
  upstream of them — arXiv fetching, embedding, the anchor store, the similarity
  filter — has been run for real, as has the leave-one-out evaluation.
- The **email has never been sent**, for the same reason: no SMTP credentials.
  Rendering is tested; delivery is not.
- Both are covered by one first run: `python main.py --dry-run` exercises
  everything except `smtplib`.
- `python main.py --dry-run` **has been run end to end** against a real arXiv
  day (2026-08-20) with no API key present, to check the failure paths hold: the
  scoring call failed twice and the run fell back to the similarity ranking, all
  ten question calls failed and were recorded per paper, and the JSON was still
  written. Both output files were deleted afterwards rather than committed — an
  archive entry with real papers and no questions would misrepresent a day, and
  `latest.json` is what the site reads.
- **Volume runs lower than the spec's 300-600/day.** These seven lists gave 185
  papers on 2026-08-20; the day pools in the leave-one-out runs ranged from 63
  (2022) to 311 (2024). It costs less than budgeted rather than more, but if you
  want the wider net, add categories (`cs.CY`, `cs.GT`, `q-fin.*`) rather than
  loosening the filter.
- The spec's test 6 ("you read seven days of emails, at least one question a
  week is worth working on") is a judgement only you can make, and the profile
  in `config.yaml` is the first thing to change if it fails.

## Layout

```
main.py                     CLI: --dry-run, --date, --rebuild-anchors, --seed
config.yaml                 categories, anchors, profile, sizes, models, email
arxiv_feed/
  arxiv.py                  the day's papers; 3 attempts, 60s apart
  embed.py                  local SPECTER2, CLS pooling, unit vectors
  anchors.py                anchor store; rebuilds when the list changes
  select.py                 top 40 by similarity + 5 from ranks 41-140
  score.py                  call 1: significance, novelty, one sentence
  questions.py              call 2: open questions, labels, canon tags
  canon.py                  the site canon's schema, mirrored
  llm.py                    structured output, retry once, then give up
  seen.py                   never send the same paper twice
  emailer.py                part 1 (questions), part 2 (papers)
  run.py                    the day, in order
tests/                      unit tests + the leave-one-out evaluation
docs/spec.md                the specification this implements
```

## Embedding model

Observed similarity range on real data: 0.90-0.97 for the ten held-out anchors,
and ~0.82 between a compiler paper and an agent-economics paper. The absolute
numbers are compressed and near-meaningless on their own — read them as a
ranking, not as a percentage.

`allenai/specter2_base`, loaded with **CLS pooling** — SPECTER-family models put
the document vector in the CLS token, and sentence-transformers would otherwise
default to mean pooling and quietly give worse vectors.

The proper SPECTER2 setup also loads a task adapter (`allenai/specter2`, the
proximity adapter) on top of the base encoder, which needs the `adapters`
package. Without it the similarity *range* compresses — unrelated papers still
score fairly high in absolute terms — but the *ranking* is what this pipeline
uses, and the leave-one-out numbers in `data/eval/` are what say whether the
ranking is good enough. If they degrade, the spec's remedies are more anchors
first, then a different embedding model; adding the adapter is the cheapest
version of the second one.
