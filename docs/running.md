# How to run it

Steps in the order you do them. Each step says what it does and how you know
it worked.

## Part 1. Get the keys

**Step 1. Get a model API key.**
Go to https://openrouter.ai/keys. Make an API key. Copy it.
The key starts with `sk-or-v1-`.

**Step 2. Skip Lakera.**
Lakera is off. `config.yaml` sets `guard.enabled: false`. You need no Lakera
account and no Lakera key. The structural defences still run on every paper.
They need no key and no network. See README, "Why Lakera is off".

To turn it on: get a key at https://platform.lakera.ai, set
`guard.enabled: true`, and export `LAKERA_GUARD_API_KEY`.

## Part 2. Set it up on your computer

**Step 3. Get the code.**

```bash
git clone https://github.com/Gigascale-Labs/las-new-papers.git
cd las-new-papers
```

**Step 4. Make a Python environment.**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Do this once. After that, run `source .venv/bin/activate` in each new terminal.

**Step 5. Install the code it needs.**

```bash
pip install -r requirements.txt
```

This downloads about 2.5GB. Most of it is PyTorch. It takes a few minutes.

**Step 6. Put the key in your terminal.**

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
```

This lasts until you close the terminal. To keep it, put the line in
`~/.bashrc` or `~/.zshrc`.

**Step 7. Check the settings.**
Open `config.yaml`. Check one thing:

- `profile` describes you, in about 300 words. It decides which papers count
  as significant and which questions count as approachable. Change it to match
  what you work on and what tools you have.

## Part 3. First run

**Step 8. Run a preview.**

```bash
python main.py --dry-run
```

`--dry-run` does the full run and writes the day's file, but marks no paper
as seen. Run it again later on the same day and you get the same papers back,
not the next unseen batch -- useful while you are still checking the setup.

The first run downloads the embedding model, about 440MB. It also embeds the 33
anchor papers. Both are saved, so this only happens once.
The run takes about 10 minutes. Most of that is embedding papers on your CPU.

**Step 9. Check the result.**
The run prints a summary line. It looks like this:

```
2026-08-20: 185 fetched, 185 screened, 14 relevant, 10 kept, 24 questions, feed 1 entries
```

Then read the file it wrote:

```bash
python -c "import json;d=json.load(open('data/latest.json'));[print(p['title']) for p in d['papers']]"
```

If you see ten paper titles, the system works.
If you see lines starting with `problem:`, read them. They say what failed.

**Step 10. Run it for real.**

```bash
python main.py
```

This marks the day's papers seen and writes `data/feed.xml`. Point a feed
reader at the file's path on your computer to check it now, before it is
anywhere public:

```
file:///absolute/path/to/las-new-papers/data/feed.xml
```

To see the page, serve the repository root over HTTP and open
`web/index.html` -- its `/data/...` paths need that, and a plain `file://`
open will not resolve them:

```bash
python -m http.server 8000               # from the repository root
# then open http://localhost:8000/web/
```

Each paper on the page carries its title, authors, a one-sentence summary,
and its open questions quoted underneath -- one list of papers, no parts, no
label on a question. The feed shows the same content.

## Part 4. Check the filter works

**Step 11. Run the leave-one-out test.**

```bash
python -m tests.leave_one_out
```

It takes one anchor out of the set, then checks the system still finds that
anchor's own paper. It repeats this ten times. At least 8 of 10 must pass.
It costs nothing and makes no model calls.

If it fails: add more anchors first, then change the embedding model.

**Step 12. Run the unit tests.**

```bash
python -m unittest discover tests
```

70 tests. One second. No network, no keys.

## Part 5. Make it daily

**Step 13. Put the key in GitHub.**
Go to your repository on GitHub.
Open Settings, then Secrets and variables, then Actions.
Add this secret, with this exact name:

- `OPENROUTER_API_KEY`

`LAKERA_GUARD_API_KEY` is read only if you set `guard.enabled: true` in
`config.yaml`.

**Step 14. Turn the workflow on.**
Open the Actions tab. Enable workflows if GitHub asks.
The workflow is `Daily arXiv open-questions feed`. It runs at 07:23 UTC each
day.

**Step 15. Test the workflow by hand.**
In the Actions tab, open the workflow. Press "Run workflow".
Set `dry_run` to true for the first test. Press the green button.
Watch it run. It takes a few minutes.
If it is green, do it again with `dry_run` false. Check `data/feed.xml` in the
repository -- it should show today's date.

After this, the system runs by itself. It commits each day's data back to the
repository.

**Step 16. Subscribe to the feed.**
The feed's URL, with no setup:

```
https://raw.githubusercontent.com/<your-username>/las-new-papers/main/data/feed.xml
```

Paste that into a feed reader (NetNewsWire, Feedly, Inoreader, or your
browser's own reader). This URL is measured to send the wrong content type
for XML (`text/plain`) -- most readers accept it anyway, since they read the
content, not only the header.

For the correct content type: repo Settings -> Pages -> Deploy from a branch
-> `main` -> `/ (root)`. Then edit `config.yaml`, set `feed.base_url` to
`https://<your-username>.github.io/las-new-papers`, and the feed URL becomes:

```
https://<your-username>.github.io/las-new-papers/data/feed.xml
```

Once Pages is on, the page is also served, at
`https://<your-username>.github.io/las-new-papers/web/`.

## Part 6. Normal use

**Step 17. Read the page, or the feed, each morning.**
Both show the same list: each paper with its title, a one-sentence summary,
and the open questions it leaves, quoted underneath with no label. Read the
questions. None of them is marked "approachable" for you any more -- that
judgement is still made and saved to the day's JSON file, but it is no longer
shown on either channel.

**Step 18. Judge it after seven days.**
The whole system has one test: at least one question a week worth working on.
If it fails, change `profile` in `config.yaml` first. Change the prompts
second.

**Step 19. Grow the corpus from the shortlist.**
`data/canon/candidates.csv` holds every paper the screen read, not just
the ten kept each day. Sort it by `significance` and `novelty`. The first
15 columns are the canon's own columns, in its order, so a paper you want to
keep is a copy across, not a re-tagging job.

**Step 20. Change the anchors when your interests change.**
Edit the `anchors` list in `config.yaml`. Add or remove arXiv IDs.
Nothing else to do. The system rebuilds the vectors on the next run.

## Other things you can run

**Scrape one day only, with no filtering and no model calls:**

```bash
python scrapers/arxiv_scraper.py --date 2026-08-19
```

This writes `data/raw/2026-08-19.jsonl.gz`, with every paper from that day.

**Re-read a past day:**

```bash
python main.py --date 2026-08-19 --dry-run
```

**Force the anchor vectors to rebuild:**

```bash
python main.py --rebuild-anchors
```

**Get the same random papers every time (for testing):**

```bash
python main.py --dry-run
```

## If something goes wrong

| What you see | What it means | What to do |
|---|---|---|
| `config error: ...` | `config.yaml` is wrong | The message names the key. Fix that key. |
| `Could not resolve authentication method` | No model API key | Do step 6 again in this terminal. |
| `Lakera screening did not run` | `guard.enabled: true` with no key | Set `guard.enabled: false`, or add the key. The other defences run either way. |
| `arXiv did not answer after 3 attempts` | arXiv is down | Wait. Run it again later with `--date` set to that day. |
| `no unseen papers found` | You already ran that day | Normal. Nothing was lost. |
| `question extraction failed` for one paper | One model call failed twice | Normal. The other papers still appear. |
| Every paper says `withheld from the model calls` | Lakera is flagging everything | Set `guard.enabled: false`. That is the default. |
