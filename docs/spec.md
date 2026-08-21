# Spec: arXiv open questions feed, v2

Supersedes 202608212059 Spec; arXiv open questions feed.
Change from v1: an embedding filter replaces the first model call.
Written in simplified technical English.

## 1. Purpose

The system sends you one email each day.
The email lists new arXiv papers that are close to papers you already value.
For each paper, the email lists the open questions in that paper.
For each question, the email says if you can work on it.

## 2. Daily flow

The system gets the new papers from arXiv.
The system removes papers it has already sent.
The system embeds the title and abstract of each new paper.
The system compares each paper to your anchor papers. It uses cosine similarity.
The system keeps the 40 closest papers. It also keeps 5 random papers from the next 100.
One model call scores these 45 papers for significance and novelty.
The system keeps the top 10 papers.
One model call per paper extracts the open questions.
The system writes the email and sends it.
The system saves the results to a file.

Step 5 keeps the 5 random papers on purpose. Similarity finds papers like the
ones you know. It hides new topics. The random papers protect against this.

## 3. Anchor papers

The anchor set is a list of 20 to 40 arXiv IDs.
These are papers you know are in your field.
Pick anchors that cover the whole field, not one corner of it.
The system embeds each anchor once. It saves the vectors to disk.
The system scores a new paper by the highest similarity to any single anchor.
Do not use the average of all anchors. An average of mixed topics points at nothing.
Add or remove anchors at any time. The system rebuilds the vectors when the list changes.

## 4. Inputs

The system reads one config file:

| Name | Meaning | Example |
|---|---|---|
| categories | arXiv lists to read | econ, cs.DC, cs.MA, cs.LG, cs.AI |
| anchors | arXiv IDs of known in-field papers | 20 to 40 IDs |
| profile | your background, skills, and projects | free text, about 300 words |
| shortlist_n | papers kept by similarity | 40 |
| explore_n | random papers added | 5 |
| top_n | papers kept after scoring | 10 |
| embed_model | embedding model | allenai/specter2 |
| model | model for scoring and questions | claude-opus-5 |
| email_to | where to send the email | one address |

The system reads two secrets from the environment.
One secret is the model API key.
One secret is the email password or token.

## 5. Output

The system sends one email with two parts.

Part 1 lists all questions from all papers in one list.

Part 2 lists each paper with these fields:

- title, authors, arXiv ID, and link;
- similarity score, and the name of the nearest anchor paper;
- a mark if the paper came from the random slice;
- score for significance, from 1 to 5;
- score for novelty, from 1 to 5;
- one sentence that says what the paper does;
- the open questions;
- for each question, one label: approachable or not approachable;
- for each question, one short reason.

The nearest anchor is important. It shows you why the system picked the paper.

The system also writes the same data to `data/YYYY-MM-DD.json`.

## 6. Parts to build

Start from the ArxivDigest fork. Add five things.

Embedding step. Load a local model. Embed the title and abstract. Use sentence-transformers.
Anchor store. Embed the anchors. Save the vectors as a .npy file. Rebuild when the list changes.
Similarity filter. Use a numpy dot product on normalised vectors. Sort. Take the top 40. Add 5 random papers.
Model calls with tool use. Use the LiteLLM merge, or copy the method from the ArxivDigest-extra fork.
Seen list. Save the arXiv IDs already sent. Skip them.

Do not build a vector database. The system holds about 500 vectors a day. A numpy array is enough.
Do not build a RAG system. This is a filter, not a question-answering system.

## 7. Model calls

Call 1 — score. One call for all 45 papers.
Input: the profile, and the title and abstract of each paper.
Output: significance, novelty, and one sentence per paper.

Call 2 — extract questions. One call per kept paper.
Input: the profile and the abstract.
Output: the open questions, the label, and the reason.

Both calls must return JSON. Use structured output.
The embedding model cannot judge significance or novelty. Only the model calls do this.

## 8. Limits and cost

The system runs once a day.
The system reads about 300 to 600 new papers a day.
The embedding step runs on your own computer. It takes a few seconds. It costs nothing.
The model calls cost about US$0.10 a day, or about US$3 a month.
Use the Batch API for call 1 if you want to halve that cost. The job is not urgent.

## 9. Error rules

If arXiv does not answer, wait 60 seconds. Try three times in total.
If one paper fails in call 2, keep the other papers. Mark the failure in the email.
If the model returns bad JSON, ask once more. Then skip that paper.
If the email fails to send, keep the JSON file. Write the error to the log.
Never stop the whole run because one paper failed.

## 10. Done when

`python main.py --dry-run` writes a JSON file and sends no email.
The email arrives each day without manual work.
The system never sends the same paper twice.
A run with a broken paper still sends the email.
Leave-one-out test. Remove one anchor from the set. Run the filter on the day that anchor appeared. The system must still rank that paper in the top 40. Repeat for 10 anchors. At least 8 must pass.
You read seven days of emails. At least one question per week is good enough to work on.

Test 5 checks the filter. Test 6 checks the whole system.
If test 5 fails, add more anchors, or change the embedding model.
If test 6 fails, change the profile first. Then change the prompts.

## 11. Not in scope

No web interface. The email is the interface.
No full PDF reading. Abstracts only.
No vector database. Files are enough.
No other users. This tool is for one person.

---

## Implementation notes against this spec

Where the build made a call the spec left open, or departed from it, it is
recorded in the README under "Deliberate departures from the spec". The
additions the spec does not mention — the frozen ground-truth canon, and canon
schema tags on finalist papers — are described in the README under "The canon"
and "Finalists arrive in the canon's schema".
