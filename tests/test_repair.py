"""Tests for scripts/repair_questions.py.

On 2026-09-03 the OpenRouter key ran out partway through the run, and one sent
paper got no questions. A rerun of the day cannot reach it: it is marked seen.
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from arxiv_feed import canon
from arxiv_feed.config import Config
from arxiv_feed.llm import ModelError
from scripts import repair_questions as rq
from tests.stubs import ScriptedClient, paper

DAY = "2026-09-03"


def setUpModule():
    import logging
    logging.disable(logging.CRITICAL)


def tearDownModule():
    import logging
    logging.disable(logging.NOTSET)


def _entry(i: int, questions: list) -> dict:
    p = paper(i)
    return {**p.to_dict(), "url": p.url, "similarity": 0.9, "similarity_rank": i,
            "nearest_anchor_id": "a1", "significance": 3, "novelty": 3,
            "one_sentence": f"Summary {i}.", "open_questions": questions, "canon": {}}


RESPONSE = {
    "open_questions": [{"question": "Does it scale?", "label": "approachable",
                        "reason": "r"}],
    "canon": {"tags": ["t1"], "summary": "Canon summary.",
              "system_type": [canon.SYSTEM_TYPES[0]]},
}


class TestRepairDay(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.data = Path(self._tmp.name)
        self._patch = mock.patch("arxiv_feed.config.DATA_DIR", self.data)
        self._patch.start()
        self.cfg = Config(categories=["cs.MA"], anchors=["a1", "a2"], profile="p")

        self.broken = _entry(1, [])                                   # failed on the day
        self.fine = _entry(2, [{"question": "Q?", "label": "approachable", "reason": ""}])
        self.empty = _entry(3, [])                                    # call worked, no questions
        self.day_path = self.data / f"{DAY}.json"
        self.day_path.write_text(json.dumps({
            "date": DAY,
            "generated_at": "2026-09-04T12:00:00+00:00",
            "counts": {"fetched": 3, "unseen": 3, "screened": 3, "relevant": 3,
                       "kept": 3},
            "papers": [self.broken, self.fine, self.empty],
            "screened": [],
            "problems": [f"{self.broken['arxiv_id']}: question extraction failed "
                         f"(call 3: failed twice: Key limit exceeded)",
                         "some other problem"],
        }, indent=2), encoding="utf-8")

        rows = [canon.to_canon_row(
            paper=paper(i), tags={}, summary=f"Summary {i}.", similarity=0.9,
            similarity_rank=i, nearest_anchor_id="a1", significance=3, novelty=3,
            screen_relevant=True, first_seen=DAY, emailed=True) for i in (1, 2, 3)]
        canon.append_candidates(self.cfg.candidates_csv, rows)

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()

    def _csv(self):
        with self.cfg.candidates_csv.open(newline="", encoding="utf-8") as f:
            return {r["arxiv_id"]: r for r in csv.DictReader(f)}

    def _run(self, responses, dry_run=False):
        client = ScriptedClient(responses)
        out = rq.repair_day(client, self.cfg, self.data, DAY, dry_run=dry_run,
                            tag_vocab=[])
        return out, client

    def test_only_the_paper_whose_call_failed_is_asked_again(self):
        (repaired, problems), client = self._run([RESPONSE])
        self.assertEqual(repaired, [self.broken["arxiv_id"]])
        self.assertEqual(problems, [])
        self.assertEqual(len(client.calls), 1)
        self.assertIn(self.broken["arxiv_id"], client.calls[0]["user"])

    def test_the_day_file_gets_the_questions_and_loses_the_problem(self):
        self._run([RESPONSE])
        data = json.loads(self.day_path.read_text(encoding="utf-8"))
        by_id = {p["arxiv_id"]: p for p in data["papers"]}
        self.assertEqual(by_id[self.broken["arxiv_id"]]["open_questions"][0]["question"],
                         "Does it scale?")
        self.assertEqual(by_id[self.broken["arxiv_id"]]["canon"]["summary"],
                         "Canon summary.")
        self.assertEqual(by_id[self.empty["arxiv_id"]]["open_questions"], [])
        self.assertEqual(data["problems"], ["some other problem"])

    def test_the_csv_row_gets_its_tags_and_keeps_its_scores(self):
        before = self._csv()
        self._run([RESPONSE])
        after = self._csv()
        row = after[self.broken["arxiv_id"]]
        self.assertEqual(row["summary"], "Canon summary.")
        self.assertEqual(row["system_type"], canon.SYSTEM_TYPES[0])
        for col in ("significance", "novelty", "screen_relevant", "emailed", "similarity"):
            self.assertEqual(row[col], before[self.broken["arxiv_id"]][col])
        self.assertEqual(after[self.fine["arxiv_id"]], before[self.fine["arxiv_id"]])

    def test_latest_and_the_feed_are_rewritten(self):
        self._run([RESPONSE])
        self.assertEqual((self.data / "latest.json").read_bytes(),
                         self.day_path.read_bytes())
        # 3 papers: the repaired one now has 1 question, the fine one had 1.
        self.assertIn(f"{DAY} — 3 papers, 2 questions",
                      self.cfg.feed_path.read_text(encoding="utf-8"))

    def test_a_second_failure_writes_nothing(self):
        before = self.day_path.read_bytes()
        csv_before = self.cfg.candidates_csv.read_bytes()
        (repaired, problems), _ = self._run([ModelError("still out of credit")])
        self.assertEqual(repaired, [])
        self.assertTrue(any("failed again" in p for p in problems))
        self.assertEqual(self.day_path.read_bytes(), before)
        self.assertEqual(self.cfg.candidates_csv.read_bytes(), csv_before)

    def test_a_dry_run_writes_nothing(self):
        before = self.day_path.read_bytes()
        (repaired, _), _ = self._run([RESPONSE], dry_run=True)
        self.assertEqual(repaired, [self.broken["arxiv_id"]])
        self.assertEqual(self.day_path.read_bytes(), before)
        self.assertFalse((self.data / "latest.json").exists())


if __name__ == "__main__":
    unittest.main()
