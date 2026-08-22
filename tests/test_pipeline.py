"""Unit tests for the parts that decide what you read.

Run: python -m unittest discover tests
"""

from __future__ import annotations

import csv
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from arxiv_feed import canon, emailer, questions, score
from arxiv_feed.llm import ModelClient, ModelError
from arxiv_feed.seen import SeenStore
from arxiv_feed.select import shortlist
from tests.stubs import ScriptedClient, paper, store, unit


def setUpModule():
    # The failure paths under test log warnings by design; silence them so a
    # passing run is quiet and a real problem stands out.
    import logging
    logging.disable(logging.CRITICAL)


def tearDownModule():
    import logging
    logging.disable(logging.NOTSET)


class TestShortlist(unittest.TestCase):
    def setUp(self):
        # 200 papers whose similarity to anchor 0 decreases with index, so the
        # expected ranking is simply the index order.
        self.papers = [paper(i) for i in range(200)]
        vecs = []
        for i in range(200):
            v = np.zeros(4, dtype=np.float32)
            v[0] = 1.0 - i * 0.004
            v[1] = 0.1
            vecs.append(unit(v))
        self.vectors = np.vstack(vecs)
        self.store = store()

    def test_keeps_top_n_by_similarity_then_random_tail(self):
        picked = shortlist(self.papers, self.vectors, self.store,
                           shortlist_n=40, explore_n=5, rng=random.Random(0))
        self.assertEqual(len(picked), 45)

        top = [c for c in picked if not c.from_random]
        self.assertEqual(len(top), 40)
        self.assertEqual([c.paper.arxiv_id for c in top],
                         [p.arxiv_id for p in self.papers[:40]])
        # Ranks are 1-based and descending in similarity.
        self.assertEqual([c.rank for c in top], list(range(1, 41)))
        self.assertTrue(all(a.similarity >= b.similarity for a, b in zip(top, top[1:])))

    def test_random_slice_comes_from_the_next_hundred_only(self):
        picked = shortlist(self.papers, self.vectors, self.store,
                           shortlist_n=40, explore_n=5, rng=random.Random(7))
        rand = [c for c in picked if c.from_random]
        self.assertEqual(len(rand), 5)
        for c in rand:
            self.assertIn(c.rank, range(41, 141))
        # No paper is in both halves.
        self.assertEqual(len({c.paper.arxiv_id for c in picked}), 45)

    def test_nearest_anchor_is_reported(self):
        # A paper aligned with anchor 2 must name anchor 2, not anchor 0.
        vecs = self.vectors.copy()
        vecs[5] = unit([0.0, 0.0, 1.0, 0.0])
        picked = shortlist(self.papers, vecs, self.store, shortlist_n=40,
                           explore_n=0, rng=random.Random(0))
        match = next(c for c in picked if c.paper.arxiv_id == self.papers[5].arxiv_id)
        self.assertEqual(match.nearest_anchor_id, "anchor2")
        self.assertAlmostEqual(match.similarity, 1.0, places=5)

    def test_max_not_mean_across_anchors(self):
        """A paper matching one anchor exactly must not be penalised by the rest."""
        s = store(n_anchors=3)
        v = unit([0, 1, 0, 0]).reshape(1, -1)
        best, idx = s.best_match(v)
        self.assertAlmostEqual(float(best[0]), 1.0, places=5)
        self.assertEqual(int(idx[0]), 1)
        # The mean over anchors would be ~0.33 -- the number the spec warns against.
        self.assertAlmostEqual(float(s.similarities(v).mean()), 1 / 3, places=5)

    def test_empty_day_is_not_an_error(self):
        self.assertEqual(shortlist([], np.zeros((0, 4), np.float32), self.store), [])

    def test_thin_day_does_not_pad(self):
        picked = shortlist(self.papers[:12], self.vectors[:12], self.store,
                           shortlist_n=40, explore_n=5, rng=random.Random(1))
        self.assertEqual(len(picked), 12)
        self.assertTrue(all(not c.from_random for c in picked))


class TestRanking(unittest.TestCase):
    def test_significance_breaks_ties_on_the_sum(self):
        papers = [paper(i) for i in range(3)]
        cands = shortlist(papers, np.vstack([unit([1, 0, 0, 0])] * 3), store(),
                          shortlist_n=3, explore_n=0, rng=random.Random(0))
        scores = {
            papers[0].arxiv_id: {"significance": 3, "novelty": 5, "one_sentence": ""},
            papers[1].arxiv_id: {"significance": 5, "novelty": 3, "one_sentence": ""},
            papers[2].arxiv_id: {"significance": 5, "novelty": 5, "one_sentence": ""},
        }
        ranked = score.rank(cands, scores, top_n=3)
        self.assertEqual(
            [c.paper.arxiv_id for c in ranked],
            [papers[2].arxiv_id, papers[1].arxiv_id, papers[0].arxiv_id],
        )

    def test_unscored_papers_cannot_win_a_slot(self):
        papers = [paper(i) for i in range(2)]
        cands = shortlist(papers, np.vstack([unit([1, 0, 0, 0])] * 2), store(),
                          shortlist_n=2, explore_n=0, rng=random.Random(0))
        scores = {papers[1].arxiv_id: {"significance": 1, "novelty": 1, "one_sentence": ""}}
        ranked = score.rank(cands, scores, top_n=2)
        self.assertEqual([c.paper.arxiv_id for c in ranked], [papers[1].arxiv_id])


class TestScoringCall(unittest.TestCase):
    def _cands(self, n=3):
        papers = [paper(i) for i in range(n)]
        return papers, shortlist(papers, np.vstack([unit([1, 0, 0, 0])] * n), store(),
                                 shortlist_n=n, explore_n=0, rng=random.Random(0))

    def test_one_call_for_the_whole_shortlist(self):
        papers, cands = self._cands(3)
        client = ScriptedClient([{ "scores": [
            {"arxiv_id": p.arxiv_id, "significance": 3, "novelty": 4, "one_sentence": "Does a thing."}
            for p in papers
        ]}])
        scores, problems = score.score_candidates(client, "profile", cands)
        self.assertEqual(len(client.calls), 1)          # not one per paper
        self.assertEqual(len(scores), 3)
        self.assertEqual(problems, [])

    def test_missing_paper_is_reported_not_invented(self):
        papers, cands = self._cands(3)
        client = ScriptedClient([{ "scores": [
            {"arxiv_id": papers[0].arxiv_id, "significance": 3, "novelty": 4, "one_sentence": "x"}
        ]}])
        scores, problems = score.score_candidates(client, "profile", cands)
        self.assertEqual(set(scores), {papers[0].arxiv_id})
        self.assertEqual(len(problems), 1)
        self.assertIn(papers[1].arxiv_id, problems[0])

    def test_hallucinated_ids_are_dropped(self):
        papers, cands = self._cands(2)
        client = ScriptedClient([{ "scores": [
            {"arxiv_id": "9999.99999", "significance": 5, "novelty": 5, "one_sentence": "x"}
        ]}])
        scores, _ = score.score_candidates(client, "profile", cands)
        self.assertEqual(scores, {})

    def test_failed_call_returns_a_problem_not_an_exception(self):
        _, cands = self._cands(2)
        client = ScriptedClient([ModelError("boom")])
        scores, problems = score.score_candidates(client, "profile", cands)
        self.assertEqual(scores, {})
        self.assertTrue(problems and "failed" in problems[0])


class TestQuestionExtraction(unittest.TestCase):
    def test_unknown_label_falls_back_to_not_approachable(self):
        client = ScriptedClient([{
            "open_questions": [
                {"question": "Q1?", "label": "approachable", "reason": "public data"},
                {"question": "Q2?", "label": "maybe", "reason": "unclear"},
                {"question": "  ", "label": "approachable", "reason": "empty"},
            ],
            "canon": {"summary": "s", "tags": [], "system_type": [], "participant_mix": [],
                      "observability": [], "focus_area": [], "threat_model": [], "claim_type": []},
        }])
        out = questions.extract(client, "profile", paper(1), tag_vocab=[])
        labels = [q["label"] for q in out["open_questions"]]
        self.assertEqual(labels, ["approachable", "not approachable"])   # blank dropped

    def test_no_questions_is_a_valid_answer(self):
        client = ScriptedClient([{"open_questions": [], "canon": {}}])
        out = questions.extract(client, "profile", paper(1), tag_vocab=[])
        self.assertEqual(out["open_questions"], [])


class TestCanonRows(unittest.TestCase):
    def test_values_off_the_closed_list_are_dropped(self):
        row = canon.to_canon_row(
            paper=paper(1),
            tags={"system_type": ["social network", "social networks", "made up"],
                  "focus_area": ["Simulation"], "claim_type": ["empirical study"],
                  "tags": ["Simulation and Digital Twins"]},
            summary="A summary.", similarity=0.55, similarity_rank=3, nearest_anchor_id="2502.14143",
            significance=4, novelty=3, from_random=False, first_seen="2026-08-21",
        )
        self.assertEqual(row["system_type"], "social network")
        self.assertEqual(row["tag_confidence"], "summary-only")
        self.assertEqual(row["itemType"], "preprint")
        self.assertEqual(row["institutions"], "")

    def test_column_order_matches_the_canon(self):
        self.assertEqual(canon.CANDIDATE_COLUMNS[:15], canon.CANON_COLUMNS)
        self.assertEqual(
            canon.CANON_COLUMNS,
            ["title", "itemType", "creators", "date", "url", "tags", "summary",
             "system_type", "participant_mix", "observability", "focus_area",
             "threat_model", "claim_type", "tag_confidence", "institutions"],
        )

    def test_frozen_ground_truth_uses_the_same_header(self):
        header = canon.GROUND_TRUTH_CSV.read_text(encoding="utf-8").splitlines()[0]
        self.assertEqual(header.split(","), canon.CANON_COLUMNS)

    def test_append_skips_duplicate_urls(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "candidates.csv"
            row = canon.to_canon_row(
                paper=paper(1), tags={}, summary="s", similarity=0.5, similarity_rank=1,
                nearest_anchor_id="a", significance=3, novelty=3,
                from_random=False, first_seen="2026-08-21")
            self.assertEqual(canon.append_candidates(path, [row]), 1)
            self.assertEqual(canon.append_candidates(path, [row]), 0)
            self.assertEqual(len(path.read_text(encoding="utf-8").strip().splitlines()), 2)


class TestCandidateRecord(unittest.TestCase):
    """Every shortlisted paper is recorded, not just the ten that were sent."""

    def _row(self, *, emailed, tags=None, rank=7):
        return canon.to_canon_row(
            paper=paper(1),
            tags=tags or {},
            summary="What it does.",
            similarity=0.61,
            similarity_rank=rank,
            nearest_anchor_id="2509.10147",
            significance=4,
            novelty=3,
            from_random=False,
            first_seen="2026-08-21",
            emailed=emailed,
        )

    def test_a_shortlisted_paper_keeps_its_filter_data_without_tags(self):
        row = self._row(emailed=False)
        self.assertEqual(row["similarity"], "0.6100")
        self.assertEqual(row["similarity_rank"], 7)
        self.assertEqual(row["nearest_anchor_id"], "2509.10147")
        self.assertEqual(row["significance"], 4)
        self.assertEqual(row["emailed"], "")
        # No question call was made for it, so every dimension is blank.
        for column in ("system_type", "focus_area", "claim_type", "threat_model"):
            self.assertEqual(row[column], "")

    def test_an_emailed_paper_is_marked_and_tagged(self):
        row = self._row(emailed=True, tags={"focus_area": ["Simulation"],
                                            "claim_type": ["empirical study"]})
        self.assertEqual(row["emailed"], "yes")
        self.assertEqual(row["focus_area"], "Simulation")

    def test_one_file_holds_both(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "candidates.csv"
            sent = self._row(emailed=True, rank=1)
            sent["url"] = "https://arxiv.org/abs/2608.00001"
            not_sent = self._row(emailed=False, rank=30)
            not_sent["url"] = "https://arxiv.org/abs/2608.00002"

            self.assertEqual(canon.append_candidates(path, [sent, not_sent]), 2)
            with path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            self.assertEqual([r["emailed"] for r in rows], ["yes", ""])
            # The canon's own columns come first, so a row lifts straight in.
            self.assertEqual(list(rows[0])[:15], canon.CANON_COLUMNS)


class TestSeen(unittest.TestCase):
    def test_round_trip_and_filtering(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "seen.json"
            s = SeenStore(path)
            s.mark([paper(1).arxiv_id], "2026-08-20")
            s.save()

            again = SeenStore(path)
            self.assertIn(paper(1).arxiv_id, again)
            remaining = again.filter_unseen([paper(1), paper(2)])
            self.assertEqual([p.arxiv_id for p in remaining], [paper(2).arxiv_id])

    def test_corrupt_file_does_not_stop_the_run(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "seen.json"
            path.write_text("{not json", encoding="utf-8")
            self.assertEqual(len(SeenStore(path)), 0)


class TestRetryRule(unittest.TestCase):
    """Exercises ModelClient._once() against a stub shaped like an OpenRouter
    ChatResult: resp.choices[0].finish_reason and .message.content."""

    class _Choice:
        def __init__(self, text, finish_reason="stop"):
            self.finish_reason = finish_reason
            self.message = type("M", (), {"content": text})()

    class _Resp:
        def __init__(self, choice):
            self.choices = [choice]

    def _client(self, choices):
        client = ModelClient("anthropic/claude-opus-5")
        calls = {"n": 0}
        responses = [self._Resp(c) for c in choices]

        class Chat:
            def send(_self, **kwargs):
                i = calls["n"]
                calls["n"] += 1
                return responses[i]

        client._client = type("C", (), {"chat": Chat()})()
        return client, calls

    def test_bad_json_is_retried_once_and_then_succeeds(self):
        client, calls = self._client([self._Choice("{oops"), self._Choice('{"ok": true}')])
        self.assertEqual(client.structured(system="s", user="u", schema={}), {"ok": True})
        self.assertEqual(calls["n"], 2)

    def test_bad_json_twice_raises_so_the_paper_is_skipped(self):
        client, calls = self._client([self._Choice("{oops"), self._Choice("{still bad")])
        with self.assertRaises(ModelError):
            client.structured(system="s", user="u", schema={})
        self.assertEqual(calls["n"], 2)                  # asks once more, not forever

    def test_refusal_and_truncation_are_reported_clearly(self):
        client, _ = self._client([self._Choice("", "content_filter"),
                                  self._Choice("", "content_filter")])
        with self.assertRaises(ModelError) as ctx:
            client.structured(system="s", user="u", schema={})
        self.assertIn("refused", str(ctx.exception))

        client, _ = self._client([self._Choice("{", "length")] * 2)
        with self.assertRaises(ModelError) as ctx:
            client.structured(system="s", user="u", schema={})
        self.assertIn("truncated", str(ctx.exception))

    def test_provider_error_finish_reason_is_reported(self):
        client, _ = self._client([self._Choice("", "error")] * 2)
        with self.assertRaises(ModelError) as ctx:
            client.structured(system="s", user="u", schema={})
        self.assertIn("provider reported an error", str(ctx.exception))


class TestSchemaName(unittest.TestCase):
    def test_strips_characters_outside_the_allowed_set(self):
        from arxiv_feed.llm import _schema_name

        self.assertEqual(_schema_name("call 2 (2608.12345)"), "call_2_2608_12345")
        self.assertEqual(_schema_name("call 1 (scoring)"), "call_1_scoring")

    def test_never_empty_and_never_over_64_chars(self):
        from arxiv_feed.llm import _schema_name

        self.assertEqual(_schema_name(""), "response")
        self.assertEqual(_schema_name("!!!"), "response")
        self.assertLessEqual(len(_schema_name("x" * 200)), 64)


class TestEmail(unittest.TestCase):
    def _result(self, problems=None):
        return {
            "date": "2026-08-21",
            "counts": {"fetched": 500, "unseen": 480, "shortlisted": 45, "kept": 2, "anchors": 33},
            "config": {"shortlist_n": 40, "explore_n": 5, "top_n": 10},
            "papers": [
                {"arxiv_id": "2608.00001", "title": "Agent markets", "authors": ["A B"],
                 "url": "https://arxiv.org/abs/2608.00001", "similarity": 0.71,
                 "nearest_anchor_id": "2509.10147", "nearest_anchor_title": "Virtual Agent Economies",
                 "from_random": False, "significance": 4, "novelty": 3,
                 "one_sentence": "Simulates a market of agents.",
                 "open_questions": [
                     {"question": "Does it hold at 10k agents?", "label": "approachable",
                      "reason": "simulation is public"},
                     {"question": "What do real venues do?", "label": "not approachable",
                      "reason": "needs proprietary exchange data"}]},
                {"arxiv_id": "2608.00002", "title": "Unrelated compiler paper", "authors": ["C D"],
                 "url": "https://arxiv.org/abs/2608.00002", "similarity": 0.21,
                 "nearest_anchor_id": "2502.14143", "nearest_anchor_title": "Multi-Agent Risks",
                 "from_random": True, "significance": 1, "novelty": 4,
                 "one_sentence": "Optimises register allocation.", "open_questions": []},
            ],
            "problems": problems or [],
            "email": {"sent": False, "to": "x@example.com", "error": None, "dry_run": True},
        }

    def test_text_email_has_both_parts_and_the_anchor_reason(self):
        text = emailer.render_text(self._result())
        self.assertIn("PART 1 -- OPEN QUESTIONS (2, 1 approachable)", text)
        self.assertIn("PART 2 -- PAPERS (2)", text)
        self.assertIn("Virtual Agent Economies", text)           # why it was picked
        self.assertIn("needs proprietary exchange data", text)
        self.assertIn("[random]", text)                          # explore-slice mark
        self.assertIn("significance 4/5, novelty 3/5", text)

    def test_failures_appear_in_the_email(self):
        text = emailer.render_text(self._result(["2608.00009: question extraction failed"]))
        self.assertIn("PROBLEMS", text)
        self.assertIn("2608.00009", text)
        html = emailer.render_html(self._result(["2608.00009: failed"]))
        self.assertIn("Problems", html)

    def test_html_escapes_paper_text(self):
        result = self._result()
        result["papers"][0]["title"] = "Agents <script>alert(1)</script> & markets"
        html = emailer.render_html(result)
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)

    def test_json_output_is_serialisable(self):
        json.dumps(self._result())


if __name__ == "__main__":
    unittest.main()
