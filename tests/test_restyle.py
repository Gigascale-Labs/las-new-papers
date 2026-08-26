"""Unit tests for the archive restyle script.

Run: python -m unittest discover tests

The script rewrites text that is already published, in files nothing else
backs up. So the cases that matter here are the ones where the model gives a
bad answer or no answer: the original text must survive all of them.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from arxiv_feed import canon
from arxiv_feed.llm import ModelError
from scripts import restyle_descriptions as rs
from tests.stubs import ScriptedClient


def setUpModule():
    import logging
    logging.disable(logging.CRITICAL)


def tearDownModule():
    import logging
    logging.disable(logging.NOTSET)


def _paper(i: int, summary: str) -> dict:
    return {
        "arxiv_id": f"2608.{i:05d}",
        "url": f"https://arxiv.org/abs/2608.{i:05d}",
        "title": f"Test paper {i}",
        "abstract": f"UNIQUE-ABSTRACT-MARKER-{i}",
        "one_sentence": summary,
        "open_questions": [],
        "canon": {"summary": f"Canon summary {i}."},
    }


def _day_file(dir_path: Path, day: str, papers: list[dict]) -> Path:
    payload = {
        "date": day,
        "generated_at": f"{day}T07:30:00+00:00",
        "counts": {"fetched": 10, "unseen": 10, "screened": 10,
                   "relevant": 2, "kept": len(papers)},
        "papers": papers,
        "screened": [{"arxiv_id": p["arxiv_id"], "one_sentence": p["one_sentence"],
                      "kept": True} for p in papers],
        "problems": [],
    }
    path = dir_path / f"{day}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _reply(pairs: list[tuple[str, str]]) -> dict:
    return {"summaries": [{"arxiv_id": aid, "one_sentence": text}
                          for aid, text in pairs]}


class TestCollect(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_reads_every_day_file_oldest_first(self):
        _day_file(self.dir, "2026-08-21", [_paper(1, "One."), _paper(2, "Two.")])
        _day_file(self.dir, "2026-08-20", [_paper(3, "Three.")])
        # latest.json is a copy of a day file; it must not be read as a day.
        (self.dir / "latest.json").write_text("{}", encoding="utf-8")
        items, problems = rs.collect(self.dir)
        self.assertEqual([it.day for it in items],
                         ["2026-08-20", "2026-08-21", "2026-08-21"])
        self.assertEqual(problems, [])

    def test_day_and_limit_narrow_the_work(self):
        _day_file(self.dir, "2026-08-21", [_paper(1, "One."), _paper(2, "Two.")])
        _day_file(self.dir, "2026-08-20", [_paper(3, "Three.")])
        items, _ = rs.collect(self.dir, day="2026-08-21")
        self.assertEqual({it.arxiv_id for it in items}, {"2608.00001", "2608.00002"})
        items, _ = rs.collect(self.dir, limit=2)
        self.assertEqual(len(items), 2)

    def test_an_unreadable_day_is_reported_not_raised(self):
        _day_file(self.dir, "2026-08-20", [_paper(1, "One.")])
        (self.dir / "2026-08-21.json").write_text("{not json", encoding="utf-8")
        items, problems = rs.collect(self.dir)
        self.assertEqual(len(items), 1)
        self.assertTrue(problems and "2026-08-21.json" in problems[0])


class TestRestyleBatch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.path = _day_file(self.dir, "2026-08-20",
                              [_paper(1, "Old one."), _paper(2, "Old two.")])
        self.items, _ = rs.collect(self.dir)

    def _reload(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def test_a_batch_round_trips_and_rewrites_the_right_field(self):
        client = ScriptedClient([_reply([("2608.00001", "New one."),
                                         ("2608.00002", "New two.")])])
        problems = rs.restyle(client, self.items)
        self.assertEqual(problems, [])
        self.assertEqual(len(client.calls), 1)          # batched, not one per paper
        written, problems = rs.write_days(self.items)
        self.assertEqual([p.name for p in written], ["2026-08-20.json"])
        self.assertEqual(problems, [])

        data = self._reload()
        self.assertEqual([p["one_sentence"] for p in data["papers"]],
                         ["New one.", "New two."])
        # Nothing else in the file moves.
        self.assertEqual([p["title"] for p in data["papers"]],
                         ["Test paper 1", "Test paper 2"])
        self.assertEqual(data["papers"][0]["canon"]["summary"], "Canon summary 1.")
        self.assertEqual(data["counts"]["kept"], 2)
        # The screened rows are the record of what the judge wrote on the day.
        self.assertEqual([s["one_sentence"] for s in data["screened"]],
                         ["Old one.", "Old two."])

    def test_the_abstract_never_reaches_the_model(self):
        """Given the abstract the model re-derives rather than rewrites."""
        client = ScriptedClient([_reply([("2608.00001", "New one.")])])
        rs.restyle(client, self.items)
        sent = client.calls[0]["user"]
        self.assertNotIn("UNIQUE-ABSTRACT-MARKER-1", sent)
        self.assertIn("Old one.", sent)
        self.assertIn("Test paper 1", sent)

    def test_a_hallucinated_id_is_dropped(self):
        client = ScriptedClient([_reply([("9999.99999", "Invented."),
                                         ("2608.00001", "New one.")])])
        rs.restyle(client, self.items)
        self.assertEqual([it.new for it in self.items], ["New one.", ""])
        rs.write_days(self.items)
        self.assertEqual([p["one_sentence"] for p in self._reload()["papers"]],
                         ["New one.", "Old two."])

    def test_a_missing_id_is_reported_not_silently_kept(self):
        client = ScriptedClient([_reply([("2608.00001", "New one.")])])
        problems = rs.restyle(client, self.items)
        self.assertEqual(len(problems), 1)
        self.assertIn("2608.00002", problems[0])
        self.assertIn("original text", problems[0])
        self.assertEqual(self.items[1].new, "")

    def test_an_empty_summary_counts_as_no_answer(self):
        client = ScriptedClient([_reply([("2608.00001", "   "),
                                         ("2608.00002", "New two.")])])
        problems = rs.restyle(client, self.items)
        self.assertIn("2608.00001", problems[0])
        rs.write_days(self.items)
        self.assertEqual([p["one_sentence"] for p in self._reload()["papers"]],
                         ["Old one.", "New two."])

    def test_a_failed_call_leaves_the_original_text_untouched(self):
        before = self.path.read_bytes()
        client = ScriptedClient([ModelError("boom")])
        problems = rs.restyle(client, self.items)
        self.assertTrue(problems and "failed" in problems[0])
        written, _ = rs.write_days(self.items)
        self.assertEqual(written, [])
        self.assertEqual(self.path.read_bytes(), before)

    def test_one_failed_batch_does_not_cost_the_other(self):
        client = ScriptedClient([ModelError("boom"),
                                 _reply([("2608.00002", "New two.")])])
        problems = rs.restyle(client, self.items, batch_size=1)
        self.assertEqual(len(problems), 1)
        rs.write_days(self.items)
        self.assertEqual([p["one_sentence"] for p in self._reload()["papers"]],
                         ["Old one.", "New two."])

    def test_text_returned_unchanged_writes_nothing(self):
        before = self.path.read_bytes()
        client = ScriptedClient([_reply([("2608.00001", "Old one."),
                                         ("2608.00002", "Old two.")])])
        rs.restyle(client, self.items)
        self.assertEqual(rs.changed(self.items), [])
        rs.write_days(self.items)
        self.assertEqual(self.path.read_bytes(), before)


class TestBatching(unittest.TestCase):
    def _items(self, ids: list[str]) -> list[rs.Item]:
        return [rs.Item(path=Path("x.json"), day="2026-08-20", arxiv_id=a,
                        url=f"u/{a}", title="t", old="o") for a in ids]

    def test_papers_are_batched(self):
        batches = rs._batches(self._items([f"id{i}" for i in range(9)]), 4)
        self.assertEqual([len(b) for b in batches], [4, 4, 1])

    def test_one_id_never_appears_twice_in_a_batch(self):
        """The model keys its answer by arxiv_id; two rows sharing one cannot."""
        batches = rs._batches(self._items(["a", "b", "a", "c"]), 8)
        self.assertEqual([[it.arxiv_id for it in b] for b in batches],
                         [["a", "b"], ["a", "c"]])


class TestLatestJson(unittest.TestCase):
    def test_latest_is_recopied_from_the_newest_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _day_file(d, "2026-08-20", [_paper(1, "Old.")])
            newest = _day_file(d, "2026-08-21", [_paper(2, "New.")])
            (d / "latest.json").write_text("stale", encoding="utf-8")
            path, problems = rs.refresh_latest(d)
            self.assertEqual(problems, [])
            self.assertEqual(path, newest)
            self.assertEqual((d / "latest.json").read_bytes(), newest.read_bytes())


class TestCandidatesCsv(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.csv_path = Path(self.tmp.name) / "candidates.csv"

    def _write(self, rows: list[dict]) -> None:
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=canon.CANDIDATE_COLUMNS)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in canon.CANDIDATE_COLUMNS})

    def _rows(self) -> list[dict]:
        with self.csv_path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _item(self, aid: str, old: str, new: str) -> rs.Item:
        return rs.Item(path=Path("2026-08-20.json"), day="2026-08-20", arxiv_id=aid,
                       url=f"https://arxiv.org/abs/{aid}", title="t", old=old, new=new)

    def test_matches_on_url_and_keeps_the_column_order(self):
        self._write([
            {"title": "A", "url": "https://arxiv.org/abs/2608.00001",
             "summary": "Old one.", "arxiv_id": "2608.00001", "emailed": "yes"},
            {"title": "B", "url": "https://arxiv.org/abs/2608.00002",
             "summary": "Old two.", "arxiv_id": "2608.00002", "emailed": ""},
        ])
        hit, other, problems = rs.update_candidates_csv(
            self.csv_path, [self._item("2608.00001", "Old one.", "New one.")])
        self.assertEqual((hit, other, problems), (1, 0, []))

        with self.csv_path.open(newline="", encoding="utf-8") as f:
            header = next(csv.reader(f))
        self.assertEqual(header, canon.CANDIDATE_COLUMNS)
        rows = self._rows()
        self.assertEqual([r["summary"] for r in rows], ["New one.", "Old two."])
        self.assertEqual([r["title"] for r in rows], ["A", "B"])
        self.assertEqual([r["emailed"] for r in rows], ["yes", ""])

    def test_a_cell_holding_other_text_is_left_alone(self):
        """A kept paper's cell holds the canon summary, from a different call."""
        self._write([{"url": "https://arxiv.org/abs/2608.00001",
                      "summary": "The canon summary, which is longer.",
                      "arxiv_id": "2608.00001"}])
        before = self.csv_path.read_bytes()
        hit, other, problems = rs.update_candidates_csv(
            self.csv_path, [self._item("2608.00001", "Old one.", "New one.")])
        # Counted and reported, not silently skipped.
        self.assertEqual((hit, other, problems), (0, 1, []))
        self.assertEqual(self._rows()[0]["summary"],
                         "The canon summary, which is longer.")
        self.assertEqual(self.csv_path.read_bytes(), before)

    def test_an_unmatched_url_changes_nothing(self):
        self._write([{"url": "https://arxiv.org/abs/2608.09999",
                      "summary": "Old one.", "arxiv_id": "2608.09999"}])
        before = self.csv_path.read_bytes()
        hit, other, problems = rs.update_candidates_csv(
            self.csv_path, [self._item("2608.00001", "Old one.", "New one.")])
        self.assertEqual((hit, other, problems), (0, 0, []))
        self.assertEqual(self.csv_path.read_bytes(), before)

    def test_a_stale_header_is_refused_not_shifted(self):
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["url", "summary"])
            w.writerow(["https://arxiv.org/abs/2608.00001", "Old one."])
        before = self.csv_path.read_bytes()
        hit, _, problems = rs.update_candidates_csv(
            self.csv_path, [self._item("2608.00001", "Old one.", "New one.")])
        self.assertEqual(hit, 0)
        self.assertTrue(problems and "stale header" in problems[0])
        self.assertEqual(self.csv_path.read_bytes(), before)


class TestMain(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.path = _day_file(self.dir, "2026-08-20",
                              [_paper(1, "Old one."), _paper(2, "Old two.")])

    def _main(self, argv, responses, env=None):
        client = ScriptedClient(list(responses))
        out = io.StringIO()
        err = io.StringIO()
        with mock.patch.dict(os.environ, env if env is not None
                             else {"OPENROUTER_API_KEY": "test-key"}, clear=True), \
             mock.patch.object(rs, "DATA_DIR", self.dir), \
             mock.patch.object(rs, "ModelClient", return_value=client), \
             contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = rs.main(argv)
        return code, out.getvalue(), err.getvalue(), client

    def test_dry_run_prints_the_pairs_and_writes_nothing(self):
        before = self.path.read_bytes()
        code, out, _, client = self._main(
            ["--dry-run"], [_reply([("2608.00001", "New one."),
                                    ("2608.00002", "New two.")])])
        self.assertEqual(code, 0)
        self.assertIn("Old one.", out)
        self.assertIn("New one.", out)
        self.assertIn("nothing written", out)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse((self.dir / "latest.json").exists())

    def test_limit_caps_the_papers_sent(self):
        code, out, _, client = self._main(
            ["--dry-run", "--limit", "1"], [_reply([("2608.00001", "New one.")])])
        self.assertEqual(code, 0)
        self.assertNotIn("2608.00002", client.calls[0]["user"])
        self.assertIn("1 summar(ies) read", out)

    def test_a_missing_api_key_is_named_and_writes_nothing(self):
        before = self.path.read_bytes()
        code, _, err, client = self._main(["--dry-run"], [], env={})
        self.assertEqual(code, 2)
        self.assertIn("OPENROUTER_API_KEY", err)
        self.assertEqual(client.calls, [])
        self.assertEqual(self.path.read_bytes(), before)

    def test_the_default_model_is_the_one_that_wrote_the_summaries(self):
        from arxiv_feed.config import load_config
        cfg = load_config(rs.REPO_ROOT / "config.yaml")
        client = ScriptedClient([_reply([("2608.00001", "New one.")])])
        made = {}

        def factory(model, effort=None, api_key=None):
            made["model"] = model
            return client

        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "k"}, clear=True), \
             mock.patch.object(rs, "DATA_DIR", self.dir), \
             mock.patch.object(rs, "ModelClient", side_effect=factory), \
             contextlib.redirect_stdout(io.StringIO()):
            rs.main(["--dry-run", "--limit", "1"])
        self.assertEqual(made["model"], cfg.judge_model)


class TestRepairSpacing(unittest.TestCase):
    """A full stop that lost its space is a formatting defect with one correct
    repair, so it is fixed in code rather than spent on a model call. Measured
    on the archive when this was written: 9 of 36 summaries (25%)."""

    def test_restores_the_space_after_a_full_stop(self):
        self.assertEqual(
            rs.repair_spacing("across tiers.Finds improved recommendations."),
            "across tiers. Finds improved recommendations.")

    def test_handles_question_and_exclamation_marks(self):
        self.assertEqual(rs.repair_spacing("Why?Because."), "Why? Because.")
        self.assertEqual(rs.repair_spacing("Stop!Then go."), "Stop! Then go.")

    def test_repairs_every_occurrence_not_just_the_first(self):
        self.assertEqual(rs.repair_spacing("One.Two.Three."), "One. Two. Three.")

    def test_leaves_correct_text_alone(self):
        for text in ("One idea. Then another.", "", "No full stop here"):
            self.assertEqual(rs.repair_spacing(text), text)

    def test_does_not_touch_a_decimal_or_a_lowercase_continuation(self):
        # "8.5-million-user" and "e.g. agents" must survive untouched: the
        # rule only fires before a capital.
        self.assertEqual(rs.repair_spacing("Uses an 8.5-million-user sample."),
                         "Uses an 8.5-million-user sample.")
        self.assertEqual(rs.repair_spacing("arXiv.org listings"), "arXiv.org listings")

    def test_an_arxiv_id_is_not_split(self):
        self.assertEqual(rs.repair_spacing("Paper 2608.24851 shows"),
                         "Paper 2608.24851 shows")

    def test_the_model_output_is_repaired_too(self):
        """A model handed clean input can still hand back a run-on."""
        items = [rs.Item(path=Path("x.json"), day="2026-08-20",
                         arxiv_id="2608.00001", url="u", title="T",
                         old="Old text.")]
        client = ScriptedClient([_reply([("2608.00001", "First.Second.")])])
        out, problems = rs.restyle_batch(client, items, "test")
        self.assertEqual(out["2608.00001"], "First. Second.")
        self.assertEqual(problems, [])

    def test_the_model_sees_repaired_input(self):
        items = [rs.Item(path=Path("x.json"), day="2026-08-20",
                         arxiv_id="2608.00001", url="u", title="T",
                         old="across tiers.Finds more.")]
        rendered = rs._render(items)
        self.assertIn("across tiers. Finds more.", rendered)
        self.assertNotIn("tiers.Finds", rendered)


if __name__ == "__main__":
    unittest.main()
