# las-new-papers

A daily digest of new arXiv papers close to papers you already value, the
open questions each one leaves, and whether you could work on them. Delivered
as an RSS/Atom feed — no account, no password — and, if you set it up, an
email too.

The repository is standalone, like
[`las-usage-stats`](https://github.com/Gigascale-Labs/las-usage-stats): it
scrapes, it commits its own data, and largeagentsystems.org can read that data
over `raw.githubusercontent.com`. Nothing here writes to the site's Airtable
canon.

Built to [`docs/spec.md`](docs/spec.md). To run it, follow
[`docs/running.md`](docs/running.md).

## What happens each day

```
arXiv, 7 lists, ~200-500 new papers
  drop papers already sent
  embed title + abstract locally      free, on your CPU
  rank against 20 anchor papers       ONLY to cut a big day to 200
  SCREEN all of them, cheap model     relevant? yes/no, in batches of 25
  JUDGE what passed, strong model     significance 1-5, novelty 1-5
  keep the top 10
  ONE model call per paper            open questions, marked approachable or not
  write data/feed.xml                 always -- the no-password channel
  write data/YYYY-MM-DD.json
  send an email                       only if FEED_EMAIL_TO is set
```

About US$0.21 a day in token cost, measured against a real 150-paper day: ~$0.10
to screen and ~$0.11 to judge. That is roughly what the single Opus scoring call
cost when it read only 45 papers, so the cascade buys 3.3x the coverage for the
same money. OpenRouter passes through the provider's per-token price and adds a
separate fee on credit purchases, reported at 5.5% by third-party sources as of
this writing. Not verified against OpenRouter's own pricing page.

**A model reads every paper now.** Screening 200 papers with a cheap model costs
about ten cents, so there is no longer a reason to let similarity decide what
gets read. It reads all of them and answers one question per paper: is this
relevant at all. Only what passes reaches the expensive call.

**Similarity no longer filters. It caps.** The anchors order a day larger than
`screen_n` so it can be cut to `screen_n`. On a day under the cap they change
nothing. This is a deliberate demotion. The old ranking produced crap: measured
on 2026-08-20, ranks 10 to 40 are separated by 0.0004 cosine, which is noise,
and 9 of the top 10 papers are off-profile. See
[docs/ranking-report.md](docs/ranking-report.md).

**The explore slice is gone.** It existed because similarity could only find
more of what the anchors already described, so a genuinely new subfield had no
anchor to be near. A screening model reading the whole day does not have that
blind spot, so five random papers are no longer buying anything.

**Similarity is still the highest score against any single anchor, never the
average.** The anchors span simulation, market design, governance and safety.
The average of those directions points at no real paper.

## Setup

```bash
pip install -r requirements.txt          # 2.5GB, mostly PyTorch

export OPENROUTER_API_KEY=sk-or-v1-...      # model calls, any provider OpenRouter lists
export LAKERA_GUARD_API_KEY=...          # optional

# Email is optional. Every run writes data/feed.xml, an RSS/Atom feed you can
# read with no account and no password. Set these three only if you also want
# a daily email:
export SMTP_PASSWORD=...                 # Gmail app password, not your login password
export FEED_EMAIL_TO=you@example.com     # where the email goes
export SMTP_USER=you@example.com         # the account that sends it

python main.py --dry-run                 # full run, writes the JSON, sends nothing
python main.py                           # and send the email
python main.py --date 2026-08-19         # read one past day
python main.py --rebuild-anchors         # rebuild the anchor vectors

python scrapers/arxiv_scraper.py         # scrape and archive one day, no models
```

The first run downloads the embedding model (440MB) and builds the anchor
vectors. Both are cached.

`config.yaml` holds the rest: categories, search queries, anchors, your
profile, the filter sizes, and the two model names. The anchor vectors rebuild
themselves when the anchor list or the embedding model changes.

**No email address is stored in this repository.** It is public, and an address
in a public file is scraped. `FEED_EMAIL_TO`, `FEED_EMAIL_FROM` and `SMTP_USER`
come from the environment. `config.yaml` refuses to load if it finds an
address, and the committed data files do not record the recipient.

### Reading it without a password

Every run writes `data/feed.xml`, an Atom feed of the last 60 days. No SMTP
account, no app password, no 2FA to work around — a feed reader polls the URL.

```
https://raw.githubusercontent.com/<owner>/<repo>/main/data/feed.xml
```

That URL works with no setup. It comes with one known defect: measured
against a real file on `raw.githubusercontent.com`, it serves `.xml` as
`text/plain`, not an XML content type. Most feed readers accept it anyway,
because they read the file's content, not only the header — this is not
guaranteed for every reader.

For the correct content type, enable GitHub Pages once: repo Settings → Pages
→ Deploy from a branch → `main` → `/ (root)`. Then set `feed.base_url` in
`config.yaml` to `https://<owner>.github.io/<repo>` and use:

```
https://<owner>.github.io/<repo>/data/feed.xml
```

I could not get a clean HTTP measurement of GitHub Pages' `.xml` content type
in the environment this was built in. Check it yourself once Pages is on:
`curl -I <pages-url>/data/feed.xml`, and look at the `content-type` header.

Email is now fully optional. Set `FEED_EMAIL_TO` (and `SMTP_USER`,
`SMTP_PASSWORD`) only if you want a daily email in addition to the feed;
leave them all unset for feed-only, and the run never attempts to send.

### Daily, without a machine of your own

`.github/workflows/daily_feed.yml` runs at 07:23 UTC and commits that day's
data, including the rebuilt feed. It needs one repository secret,
`OPENROUTER_API_KEY`. Add `FEED_EMAIL_TO`, `SMTP_USER` and `SMTP_PASSWORD`
only for email too, and `LAKERA_GUARD_API_KEY` if you use Lakera.

The commit step runs even when the run failed. A run that wrote its JSON but
delivered nothing on any enabled channel exits non-zero; a run with email
unconfigured and a feed that wrote successfully is not a failure.

## What it writes

| Path | What it is |
|---|---|
| `data/YYYY-MM-DD.json` | The day's ten papers with questions, every screened paper with its verdict and scores, and any problems. |
| `data/latest.json` | The newest day, at a fixed URL the site can read. |
| `data/feed.xml` | An Atom feed of the last 60 days. No password needed to read it. |
| `data/raw/YYYY-MM-DD.jsonl.gz` | Every paper scraped that day. ~120KB. |
| `data/canon/candidates.csv` | Every screened paper ever seen, in the canon's schema, with the screen's yes/no. |
| `data/seen.json` | IDs already sent. Nothing is sent twice. |
| `data/eval/leave-one-out.json` | The most recent filter evaluation. |
| `data/ground-truth/` | A frozen copy of the human canon. |

`data/raw/` is what makes this an archive. You can re-rank an old day with
different anchors, or a different embedding model, without fetching again.

Nothing either model call learns is thrown away. Every screened paper keeps its
yes/no verdict and its one-line reason, and everything that passed the screen
keeps its significance and novelty, in both `data/YYYY-MM-DD.json` and
`candidates.csv`, at no extra cost. The rejects are kept deliberately: the
record of what the screen threw away is what makes it auditable.

## The canon

`data/ground-truth/` holds the 42-paper human canon, copied from
`data/las-canon.airtable.json` in the site repository — the file the live site
reads. Airtable is upstream of it. Do not copy from `las-canon.csv`; it is a
fallback and has encoding damage.

The copy is frozen. It is the test set for the filter, and a test set that
follows the canon makes older results incomparable. Refresh it as its own
commit when you want to measure against a newer canon.

33 of the 42 canon entries are on arXiv. 20 of those 33 are anchors, kept for
two themes: gradual disempowerment, and dynamics that only appear in very
large agent populations. The other 13 arXiv entries, and the 9 not on arXiv,
stay in the canon but are not anchors.

Every screened paper is appended to `data/canon/candidates.csv`, in the
canon's column order. One file, not two: the ten that were emailed carry their
six dimension tags and an `emailed` mark, and the rest carry their
similarity, rank and scores with the dimensions blank. Blank dimensions are
normal in the canon.

This is the pool the canon grows from. Sort by `significance` and `novelty`,
read what looks worth reading, and copy the row across — the first 15 columns
are the canon's own, in its order.

The file is a proposal, not a promotion. Nothing here writes to Airtable, and a
human decides what is admitted.

Two limits on the tags. They are read from the abstract, so every row is
`tag_confidence: summary-only`. `institutions` stays blank, because an abstract
does not carry affiliations. A value that is not on the closed list is dropped
rather than forced.

The choice lists in `arxiv_feed/canon.py` are copied by hand from
`lib/canon-schema.ts` in the site repository. A change there needs a change
here.

## Defences against untrusted text

An arXiv abstract is untrusted. Anyone who can submit a preprint writes one,
and that text reaches two model calls, an HTML email, a CSV, and a JSON file
the site reads. Three layers, in `arxiv_feed/guard.py`:

1. **Structural. Always on, no key.** Invisible characters removed (zero-width,
   bidirectional overrides, and the Unicode Tags block, which hides
   instructions inside ordinary-looking text), length capped, and each abstract
   fenced in a tag with a random id under a system prompt stating that fenced
   text is data. Text inside cannot close the fence, because it cannot know the
   id. This layer works without recognising the attack, so it is the one that
   holds.
2. **Lakera Guard. Optional.** Screens the papers before any
   model call and withholds what it flags. Up to `screen_n` a day. A withheld
   paper stays in the archive and is named in the email.
3. **Keyword patterns. Always on, never blocking.**

Layer 3 never blocks because this corpus can include papers about prompt
injection and other attack techniques -- that is what the field studies. A
keyword rule strong enough to catch a real attack would hide exactly those
papers.

Each output is protected where it lands:

| Output | Risk | Defence |
|---|---|---|
| HTML email | markup from a title or abstract | every field escaped |
| `candidates.csv` | `=HYPERLINK(...)` in a title runs when Excel opens it | cells starting `= + - @` prefixed with `'` |
| Email headers | a newline adding a header | CR/LF stripped, and rejected at config load |
| Links | a forged ID becoming a `javascript:` href | IDs checked against arXiv's two real formats |
| Model output | invented values or wrong shape | JSON schema, closed lists, IDs checked against the batch |

`guard.on_error` sets what a Lakera outage means: `allow` (default, you still
get the email) or `block` (withhold everything it could not screen).

## Failures never stop a run

- arXiv does not answer: 3 tries, 60 seconds apart.
- The scoring call fails: fall back to the similarity ranking.
- One paper's question call fails twice: drop that paper, keep the rest, name
  it in the email and the feed.
- The email fails (if configured at all): report it as a problem, but the
  feed has already delivered the day, so the papers are still marked seen and
  the run still exits 0.

## Tests

```bash
python -m unittest discover tests     # 70 tests, no network, no keys
python -m tests.leave_one_out         # the filter evaluation, real arXiv, free
```

**Leave-one-out: 10 of 10 passed**, measured against the previous 33-anchor
set. The threshold is 8 of 10. Each round removes one anchor, fetches every
paper from the day that anchor appeared, and ranks against the remaining
anchors. The removed paper must come back in the top 40. Seven came back at
rank 1 in pools of 63-311 papers. The worst was rank 28. The report is in
`data/eval/leave-one-out.json`.

Not yet re-run against the narrowed 20-anchor set -- `python -m
tests.leave_one_out --anchors all` is the command that would settle it.

The unit tests cover the pre-sort cap, highest-against-anchors rather than
average, the screen batching and its yes/no split, one failed batch not costing
the others, the ranking tie-break, an omitted score being reported rather than
invented, hallucinated arXiv IDs being dropped, bad JSON retried exactly once,
and every guard layer.

## Current state

Three things have never run against a live service, because this environment
has no credentials for them: the model calls, the email send, and the Lakera
call. Each is written against the current API and tested with a stub. One real
run confirms all three.

Everything else has run against live data: arXiv, embedding, the anchor store,
the filter, and the leave-one-out evaluation.

Volume is lower than the spec's 300-600 a day. These seven lists gave 185
papers on 2026-08-20. To widen the net, add categories (`cs.CY`, `cs.GT`,
`q-fin.*`) rather than loosening the filter.

`data/feed.xml` is checked as well-formed XML (parsed with
`xml.etree.ElementTree`, 6 unit tests) and validated in a dry run to contain
the expected entries. It has not been opened in a real feed reader, and
GitHub Pages' `.xml` content type has not been measured — see "Reading it
without a password" above for what to check once Pages is on.

## Layout

```
main.py                     CLI: --dry-run, --date, --rebuild-anchors
config.yaml                 categories, search queries, anchors, profile, sizes, models
arxiv_feed/
  arxiv.py                  the day's papers; 3 tries, 60s apart
  embed.py                  local SPECTER2, CLS pooling, unit vectors
  anchors.py                anchor vectors; rebuild when the list changes
  select.py                 top 40 by similarity, plus 5 from ranks 41-140
  score.py                  call 1: significance, novelty, one sentence
  questions.py              call 2: open questions, labels, canon tags
  canon.py                  the site canon's schema, mirrored
  guard.py                  sanitising, fencing, Lakera, output protection
  llm.py                    structured output, one retry
  seen.py                   never send the same paper twice
  emailer.py                part 1 (questions), part 2 (papers)
  feed.py                   rebuilds data/feed.xml from the day files on disk
  run.py                    the day, in order
scrapers/
  arxiv_scraper.py          standalone: one day of papers -> data/raw/
tests/                      unit tests and the leave-one-out evaluation
docs/spec.md                the specification
docs/running.md             how to run it, step by step
```

## Embedding model

`allenai/specter2_base`, with **CLS pooling**. SPECTER models put the document
vector in the CLS token. sentence-transformers defaults to mean pooling for a
plain checkpoint, which gives worse vectors.

The full SPECTER2 setup also loads a task adapter on top of the base encoder,
which needs the `adapters` package. Without it the similarity range compresses:
on real data, the ten held-out anchors scored 0.90-0.97, and an unrelated
compiler paper still scored about 0.82 against an agent-economics paper. Read
those numbers as a ranking, not as a percentage. The ranking is what the filter
uses, and `data/eval/` says whether it is good enough. If it degrades, add more
anchors first, then change the embedding model; adding the adapter is the
cheapest version of the second.
