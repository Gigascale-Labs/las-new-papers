# How to run it

Steps in the order you do them. Each step says what it does and how you know
it worked.

## Part 1. Get the keys

**Step 1. Get a model API key.**
Go to https://console.anthropic.com. Make an API key. Copy it.
The key starts with `sk-ant-`.

**Step 2. Get an email password.**
The system sends email through Gmail. Your normal Gmail password will not work.
Go to https://myaccount.google.com/apppasswords. Make an app password.
Copy the 16 letters.
If you use a different email provider, change `smtp_host`, `smtp_port` and
`smtp_user` in `config.yaml` instead.

**Step 3. Get a Lakera key.**
Go to https://platform.lakera.ai. Make an account. Make a project.
Copy the API key. Copy the project ID if you made one.
This key is optional. Without it the system still runs and still defends
itself. It just does not screen papers with Lakera, and it says so in the
email.

## Part 2. Set it up on your computer

**Step 4. Get the code.**

```bash
git clone https://github.com/one-2/las-new-papers.git
cd las-new-papers
```

**Step 5. Make a Python environment.**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Do this once. After that, run `source .venv/bin/activate` in each new terminal.

**Step 6. Install the code it needs.**

```bash
pip install -r requirements.txt
```

This downloads about 2.5GB. Most of it is PyTorch. It takes a few minutes.

**Step 7. Put the keys in your terminal.**

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export SMTP_PASSWORD=your-16-letter-app-password
export LAKERA_GUARD_API_KEY=lakera-...

export FEED_EMAIL_TO=you@example.com          # where the email goes
export SMTP_USER=you@example.com              # the account that sends it
```

These last until you close the terminal. To keep them, put the lines in
`~/.bashrc` or `~/.zshrc`.

Your address is an environment variable, not a config setting. This repository
is public, and an address in a public file is scraped. `config.yaml` refuses to
load if it finds one.

**Step 8. Check the settings.**
Open `config.yaml`. Check one thing:

- `profile` describes you, in about 300 words. It decides which papers count
  as significant and which questions count as approachable. Change it to match
  what you work on and what tools you have.

## Part 3. First run

**Step 9. Run it without sending email.**

```bash
python main.py --dry-run
```

The first run downloads the embedding model, about 440MB. It also embeds the 33
anchor papers. Both are saved, so this only happens once.
The run takes about 10 minutes. Most of that is embedding papers on your CPU.

**Step 10. Check the result.**
The run prints a summary line. It looks like this:

```
2026-08-20: 185 fetched, 10 kept, 24 questions, email skipped (dry run)
```

Then read the file it wrote:

```bash
python -c "import json;d=json.load(open('data/latest.json'));[print(p['title']) for p in d['papers']]"
```

If you see ten paper titles, the system works.
If you see lines starting with `problem:`, read them. They say what failed.

**Step 11. Send one real email.**

```bash
python main.py
```

Check your inbox. Part 1 of the email is all the questions. Part 2 is the
papers.
If the email does not arrive, the data file is still saved. The error is in the
terminal output.

## Part 4. Check the filter works

**Step 12. Run the leave-one-out test.**

```bash
python -m tests.leave_one_out
```

It takes one anchor out of the set, then checks the system still finds that
anchor's own paper. It repeats this ten times. At least 8 of 10 must pass.
It costs nothing and makes no model calls.

If it fails: add more anchors first, then change the embedding model.

**Step 13. Run the unit tests.**

```bash
python -m unittest discover tests
```

53 tests. One second. No network, no keys.

## Part 5. Make it daily

**Step 14. Put the keys in GitHub.**
Go to your repository on GitHub.
Open Settings, then Secrets and variables, then Actions.
Add these secrets, with these exact names:

- `ANTHROPIC_API_KEY`
- `FEED_EMAIL_TO` -- where the email goes
- `SMTP_USER` -- the account that sends it
- `SMTP_PASSWORD`
- `LAKERA_GUARD_API_KEY` (optional)

The two addresses are secrets, not config. They never appear in the repository
or in the data the workflow commits.

**Step 15. Turn the workflow on.**
Open the Actions tab. Enable workflows if GitHub asks.
The workflow is `Daily arXiv open-questions feed`. It runs at 07:23 UTC each
day.

**Step 16. Test the workflow by hand.**
In the Actions tab, open the workflow. Press "Run workflow".
Set `dry_run` to true for the first test. Press the green button.
Watch it run. It takes a few minutes.
If it is green, do it again with `dry_run` false. Check your inbox.

After this, the system runs by itself. It commits each day's data back to the
repository.

## Part 6. Normal use

**Step 17. Read the email each morning.**
Part 1 is the list of questions. Read that first. `approachable` means you
could start on it within a few weeks, with public data or a simulation you
write yourself.

**Step 18. Judge it after seven days.**
The whole system has one test: at least one question a week worth working on.
If it fails, change `profile` in `config.yaml` first. Change the prompts
second.

**Step 19. Change the anchors when your interests change.**
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
python main.py --dry-run --seed 1
```

## If something goes wrong

| What you see | What it means | What to do |
|---|---|---|
| `config error: ...` | `config.yaml` is wrong | The message names the key. Fix that key. |
| `Could not resolve authentication method` | No model API key | Do step 7 again in this terminal. |
| `SMTP_PASSWORD is not set` | No email password | Do step 7 again. Use an app password, not your login password. |
| `Lakera screening did not run` | No Lakera key | Optional. Add the key, or ignore it. The other defences still run. |
| `arXiv did not answer after 3 attempts` | arXiv is down | Wait. Run it again later with `--date` set to that day. |
| `no unseen papers found` | You already ran that day | Normal. Nothing was lost. |
| `question extraction failed` for one paper | One model call failed twice | Normal. The other papers are still sent. |
| Every paper says `withheld from the model calls` | Lakera is flagging everything | Check the Lakera project policy. Or set `guard.enabled: false` to test. |
