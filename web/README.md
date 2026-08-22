# web

The public viewer. Static HTML, no build step, no dependencies.

Vercel serves this directory. Set the project's **Root Directory** to `web`.
That keeps the Python `requirements.txt` at the repo root out of the build:
it installs torch, which the viewer does not need and which would exceed the
function size limit.

`vercel.json` rewrites `/data/*` to the repository's `data/` directory on
raw.githubusercontent.com. Two consequences:

- The site shows the current day without a redeploy. The daily job commits
  `data/latest.json`, and the next page load reads it.
- The feed gets a correct `Content-Type`. raw.githubusercontent.com serves
  `text/plain`, which the README notes as a defect. This fixes it at the edge.

The page shows the title, the summary and the questions. It shows no score, no
similarity value and no screening reason. Nothing on the page argues for a
paper.
