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

from arxiv_feed import canon, emailer, judge, questions, screen
from arxiv_feed.llm import ModelClient, ModelError
from arxiv_feed.seen import SeenStore
from arxiv_feed.select import preselect
from tests.stubs import ScriptedClient, paper, store, unit


def setUpModule():
    # The failure paths under test log warnings by design; silence them so a
    # passing run is quiet and a real problem stands out.
    import logging
    logging.disable(logging.CRITICAL)


def tearDownModule():
    import logging
    logging.disable(logging.NOTSET)


class TestPreselect(unittest.TestCase):
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

    def test_orders_by_similarity_and_caps(self):
        picked = preselect(self.papers, self.vectors, self.store, screen_n=40)
        self.assertEqual(len(picked), 40)
        self.assertEqual([c.paper.arxiv_id for c in picked],
                         [p.arxiv_id for p in self.papers[:40]])
        # Ranks are 1-based and descending in similarity.
        self.assertEqual([c.rank for c in picked], list(range(1, 41)))
        self.assertTrue(all(a.similarity >= b.similarity
                            for a, b in zip(picked, picked[1:])))

    def test_a_day_under_the_cap_loses_nothing(self):
        """The common case: the pre-sort is a cap, not a filter."""
        picked = preselect(self.papers[:120], self.vectors[:120], self.store,
                           screen_n=200)
        self.assertEqual(len(picked), 120)
        self.assertEqual({c.paper.arxiv_id for c in picked},
                         {p.arxiv_id for p in self.papers[:120]})

    def test_a_day_over_the_cap_drops_only_the_tail(self):
        picked = preselect(self.papers, self.vectors, self.store, screen_n=150)
        self.assertEqual(len(picked), 150)
        dropped = {p.arxiv_id for p in self.papers[150:]}
        self.assertEqual({c.paper.arxiv_id for c in picked} & dropped, set())

    def test_nearest_anchor_is_reported(self):
        # A paper aligned with anchor 2 must name anchor 2, not anchor 0.
        vecs = self.vectors.copy()
        vecs[5] = unit([0.0, 0.0, 1.0, 0.0])
        picked = preselect(self.papers, vecs, self.store, screen_n=40)
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
        self.assertEqual(preselect([], np.zeros((0, 4), np.float32), self.store), [])


class TestRanking(unittest.TestCase):
    def test_significance_breaks_ties_on_the_sum(self):
        papers = [paper(i) for i in range(3)]
        cands = preselect(papers, np.vstack([unit([1, 0, 0, 0])] * 3), store())
        scores = {
            papers[0].arxiv_id: {"significance": 3, "novelty": 5, "one_sentence": ""},
            papers[1].arxiv_id: {"significance": 5, "novelty": 3, "one_sentence": ""},
            papers[2].arxiv_id: {"significance": 5, "novelty": 5, "one_sentence": ""},
        }
        ranked = judge.rank(cands, scores, top_n=3)
        self.assertEqual(
            [c.paper.arxiv_id for c in ranked],
            [papers[2].arxiv_id, papers[1].arxiv_id, papers[0].arxiv_id],
        )

    def test_unjudged_papers_cannot_win_a_slot(self):
        papers = [paper(i) for i in range(2)]
        cands = preselect(papers, np.vstack([unit([1, 0, 0, 0])] * 2), store())
        scores = {papers[1].arxiv_id: {"significance": 1, "novelty": 1, "one_sentence": ""}}
        ranked = judge.rank(cands, scores, top_n=2)
        self.assertEqual([c.paper.arxiv_id for c in ranked], [papers[1].arxiv_id])


def _cands(n=3):
    papers = [paper(i) for i in range(n)]
    return papers, preselect(papers, np.vstack([unit([1, 0, 0, 0])] * n), store())


class TestScreeningCall(unittest.TestCase):
    """Call 1: the cheap model reads every paper and answers relevant yes/no."""

    def _verdicts(self, papers, relevant=True):
        return {"verdicts": [
            {"arxiv_id": p.arxiv_id, "relevant": relevant, "reason": "r"}
            for p in papers
        ]}

    def test_papers_are_batched_not_sent_one_per_call(self):
        papers, cands = _cands(10)
        client = ScriptedClient([self._verdicts(papers[:4]),
                                 self._verdicts(papers[4:8]),
                                 self._verdicts(papers[8:])])
        verdicts, problems = screen.screen_candidates(
            client, "profile", cands, batch_size=4)
        self.assertEqual(len(client.calls), 3)       # ceil(10/4), not 10
        self.assertEqual(len(verdicts), 10)
        self.assertEqual(problems, [])

    def test_only_the_yeses_go_forward(self):
        papers, cands = _cands(4)
        client = ScriptedClient([{"verdicts": [
            {"arxiv_id": papers[0].arxiv_id, "relevant": True, "reason": "fits"},
            {"arxiv_id": papers[1].arxiv_id, "relevant": False, "reason": "single agent"},
            {"arxiv_id": papers[2].arxiv_id, "relevant": True, "reason": "fits"},
            {"arxiv_id": papers[3].arxiv_id, "relevant": False, "reason": "attack method"},
        ]}])
        verdicts, _ = screen.screen_candidates(client, "profile", cands, batch_size=10)
        passed = screen.relevant(cands, verdicts)
        self.assertEqual([c.paper.arxiv_id for c in passed],
                         [papers[0].arxiv_id, papers[2].arxiv_id])

    def test_a_day_with_no_yeses_is_not_an_error(self):
        papers, cands = _cands(3)
        client = ScriptedClient([self._verdicts(papers, relevant=False)])
        verdicts, problems = screen.screen_candidates(
            client, "profile", cands, batch_size=10)
        self.assertEqual(problems, [])
        self.assertEqual(screen.relevant(cands, verdicts), [])

    def test_one_failed_batch_does_not_cost_the_others(self):
        papers, cands = _cands(8)
        client = ScriptedClient([ModelError("boom"), self._verdicts(papers[4:])])
        verdicts, problems = screen.screen_candidates(
            client, "profile", cands, batch_size=4)
        self.assertEqual(len(verdicts), 4)                  # the second batch survived
        self.assertTrue(problems and "failed" in problems[0])
        self.assertIn("4 paper(s) unscreened", problems[0])

    def test_missing_paper_is_reported_not_defaulted(self):
        papers, cands = _cands(3)
        client = ScriptedClient([{"verdicts": [
            {"arxiv_id": papers[0].arxiv_id, "relevant": True, "reason": "r"}
        ]}])
        verdicts, problems = screen.screen_candidates(
            client, "profile", cands, batch_size=10)
        self.assertEqual(set(verdicts), {papers[0].arxiv_id})
        self.assertEqual(len(problems), 1)
        self.assertIn(papers[1].arxiv_id, problems[0])

    def test_hallucinated_ids_are_dropped(self):
        _, cands = _cands(2)
        client = ScriptedClient([{"verdicts": [
            {"arxiv_id": "9999.99999", "relevant": True, "reason": "x"}
        ]}])
        verdicts, _ = screen.screen_candidates(client, "profile", cands, batch_size=10)
        self.assertEqual(screen.relevant(cands, verdicts), [])


class TestJudgingCall(unittest.TestCase):
    """Call 2: the strong model scores only what the screen passed."""

    def test_one_call_for_everything_that_passed(self):
        papers, cands = _cands(3)
        client = ScriptedClient([{ "judgements": [
            {"arxiv_id": p.arxiv_id, "significance": 3, "novelty": 4, "one_sentence": "Does a thing."}
            for p in papers
        ]}])
        judgements, problems = judge.judge_candidates(client, "profile", cands)
        self.assertEqual(len(client.calls), 1)          # not one per paper
        self.assertEqual(len(judgements), 3)
        self.assertEqual(problems, [])

    def test_nothing_passed_the_screen_makes_no_call(self):
        client = ScriptedClient([])
        judgements, problems = judge.judge_candidates(client, "profile", [])
        self.assertEqual(client.calls, [])
        self.assertEqual(judgements, {})
        self.assertEqual(problems, [])

    def test_missing_paper_is_reported_not_invented(self):
        papers, cands = _cands(3)
        client = ScriptedClient([{ "judgements": [
            {"arxiv_id": papers[0].arxiv_id, "significance": 3, "novelty": 4, "one_sentence": "x"}
        ]}])
        judgements, problems = judge.judge_candidates(client, "profile", cands)
        self.assertEqual(set(judgements), {papers[0].arxiv_id})
        self.assertEqual(len(problems), 1)
        self.assertIn(papers[1].arxiv_id, problems[0])

    def test_hallucinated_ids_are_dropped(self):
        _, cands = _cands(2)
        client = ScriptedClient([{ "judgements": [
            {"arxiv_id": "9999.99999", "significance": 5, "novelty": 5, "one_sentence": "x"}
        ]}])
        judgements, _ = judge.judge_candidates(client, "profile", cands)
        self.assertEqual(judgements, {})

    def test_failed_call_returns_a_problem_not_an_exception(self):
        _, cands = _cands(2)
        client = ScriptedClient([ModelError("boom")])
        judgements, problems = judge.judge_candidates(client, "profile", cands)
        self.assertEqual(judgements, {})
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
            significance=4, novelty=3, screen_relevant=True, first_seen="2026-08-21",
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
                screen_relevant=True, first_seen="2026-08-21")
            self.assertEqual(canon.append_candidates(path, [row]), 1)
            self.assertEqual(canon.append_candidates(path, [row]), 0)
            self.assertEqual(len(path.read_text(encoding="utf-8").strip().splitlines()), 2)


class TestCandidateRecord(unittest.TestCase):
    """Every screened paper is recorded, not just the ten that were sent."""

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
            screen_relevant=True,
            first_seen="2026-08-21",
            emailed=emailed,
        )

    def test_a_screened_paper_keeps_its_filter_data_without_tags(self):
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
            "counts": {"fetched": 500, "unseen": 480, "screened": 200,
                       "relevant": 12, "kept": 2, "anchors": 20},
            "config": {"screen_n": 200, "screen_batch_size": 25, "top_n": 10,
                       "screen_model": "anthropic/claude-haiku-4.5",
                       "judge_model": "anthropic/claude-sonnet-5"},
            "papers": [
                {"arxiv_id": "2608.00001", "title": "Agent markets", "authors": ["A B"],
                 "url": "https://arxiv.org/abs/2608.00001", "similarity": 0.71,
                 "nearest_anchor_id": "2509.10147", "nearest_anchor_title": "Virtual Agent Economies",
                 "significance": 4, "novelty": 3,
                 "one_sentence": "Simulates a market of agents.",
                 "open_questions": [
                     {"question": "Does it hold at 10k agents?", "label": "approachable",
                      "reason": "simulation is public"},
                     {"question": "What do real venues do?", "label": "not approachable",
                      "reason": "needs proprietary exchange data"}]},
                {"arxiv_id": "2608.00002", "title": "Unrelated compiler paper", "authors": ["C D"],
                 "url": "https://arxiv.org/abs/2608.00002", "similarity": 0.21,
                 "nearest_anchor_id": "2502.14143", "nearest_anchor_title": "Multi-Agent Risks",
                 "significance": 1, "novelty": 4,
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
        # The two-stage cascade is stated in the footer, not the explore slice.
        self.assertIn("200 paper(s) screened by anthropic/claude-haiku-4.5", text)
        self.assertIn("12 judged relevant", text)
        # Scores order Part 2. They are not shown -- see emailer's docstring.
        self.assertNotIn("significance", text)
        self.assertNotIn("novelty", text)

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


class TestRunWiring(unittest.TestCase):
    """The daily run, end to end, with the network and both models stubbed.

    This is the test that catches a cascade wired up backwards: the screen
    reading the wrong papers, the judge reading everything instead of only what
    passed, or the counts reporting a stage that never ran.
    """

    def _cfg(self, **over):
        from arxiv_feed.config import Config
        base = dict(
            categories=["cs.MA"],
            anchors=["a1", "a2"],
            profile="profile text",
            screen_n=200,
            screen_batch_size=25,
            top_n=2,
            guard={"enabled": False},
        )
        base.update(over)
        return Config(**base)

    def _run(self, cfg, papers, screen_responses, judge_response,
             question_response=None):
        from unittest import mock

        from arxiv_feed import run as run_mod

        clients = [
            ScriptedClient(list(screen_responses)),
            ScriptedClient([judge_response] if judge_response is not None else []),
            ScriptedClient([question_response] * 10 if question_response else []),
        ]
        made: list = []

        def client_factory(model, effort=None, api_key=None):
            c = clients[len(made)] if len(made) < len(clients) else ScriptedClient([])
            c.model = model
            made.append(c)
            return c

        vectors = np.vstack([unit([1.0 - i * 0.01, 0.1, 0, 0])
                             for i in range(len(papers))])

        class _Emb:
            def __init__(self, *a, **k): pass
            def encode(self, texts): return vectors[: len(texts)]

        with mock.patch.object(run_mod, "scrape_day", return_value=papers), \
             mock.patch.object(run_mod, "Embedder", _Emb), \
             mock.patch.object(run_mod.anchors_mod, "load_or_build",
                               return_value=store()), \
             mock.patch.object(run_mod, "ModelClient", side_effect=client_factory), \
             mock.patch.object(run_mod, "write_output"), \
             mock.patch.object(run_mod.canon, "append_candidates", return_value=0), \
             mock.patch.object(run_mod.canon, "known_tags", return_value=[]), \
             mock.patch.object(run_mod.feed_mod, "rebuild", return_value=1), \
             mock.patch.object(run_mod, "SeenStore") as seen:
            seen.return_value.filter_unseen.side_effect = lambda ps: ps
            result = run_mod.run(cfg, day="2026-08-20", dry_run=True)
        return result, made

    def test_screen_reads_everything_and_judge_reads_only_the_passes(self):
        papers = [paper(i) for i in range(6)]
        screen = {"verdicts": [
            {"arxiv_id": p.arxiv_id, "relevant": i < 2, "reason": "r"}
            for i, p in enumerate(papers)
        ]}
        judged = {"judgements": [
            {"arxiv_id": papers[0].arxiv_id, "significance": 5, "novelty": 4,
             "one_sentence": "One."},
            {"arxiv_id": papers[1].arxiv_id, "significance": 3, "novelty": 3,
             "one_sentence": "Two."},
        ]}
        result, clients = self._run(
            self._cfg(), papers, [screen], judged,
            {"open_questions": [], "canon": {}})

        self.assertEqual(result["counts"]["screened"], 6)   # every paper
        self.assertEqual(result["counts"]["relevant"], 2)   # the two yeses
        self.assertEqual(result["counts"]["kept"], 2)

        # The judge saw exactly the two that passed, and no others.
        judge_user = clients[1].calls[0]["user"]
        self.assertIn(papers[0].arxiv_id, judge_user)
        self.assertIn(papers[1].arxiv_id, judge_user)
        for p in papers[2:]:
            self.assertNotIn(p.arxiv_id, judge_user)

    def test_every_screened_paper_is_recorded_including_the_rejects(self):
        papers = [paper(i) for i in range(4)]
        screen = {"verdicts": [
            {"arxiv_id": p.arxiv_id, "relevant": i == 0, "reason": f"reason {i}"}
            for i, p in enumerate(papers)
        ]}
        judged = {"judgements": [
            {"arxiv_id": papers[0].arxiv_id, "significance": 4, "novelty": 4,
             "one_sentence": "x"}]}
        result, _ = self._run(self._cfg(), papers, [screen], judged,
                              {"open_questions": [], "canon": {}})

        self.assertEqual(len(result["screened"]), 4)
        by_id = {r["arxiv_id"]: r for r in result["screened"]}
        self.assertIs(by_id[papers[1].arxiv_id]["relevant"], False)
        self.assertEqual(by_id[papers[1].arxiv_id]["screen_reason"], "reason 1")
        # A rejected paper carries no scores: the judge never saw it.
        self.assertIsNone(by_id[papers[1].arxiv_id]["significance"])
        self.assertTrue(by_id[papers[0].arxiv_id]["kept"])

    def test_no_unseen_papers_still_writes_output_and_rebuilds_the_feed(self):
        """A weekend day with nothing unseen must still advance latest.json and
        the feed. Left unwritten, the web UI freezes on the last day that had
        papers -- arXiv's own Friday-to-Sunday gap, every week."""
        from unittest import mock

        from arxiv_feed import run as run_mod

        class _Emb:
            def __init__(self, *a, **k): pass
            def encode(self, texts): return np.zeros((0, 4), dtype=np.float32)

        with mock.patch.object(run_mod, "scrape_day", return_value=[]), \
             mock.patch.object(run_mod, "Embedder", _Emb), \
             mock.patch.object(run_mod.anchors_mod, "load_or_build",
                               return_value=store()), \
             mock.patch.object(run_mod, "ModelClient") as model_client, \
             mock.patch.object(run_mod, "write_output") as write_output, \
             mock.patch.object(run_mod.canon, "append_candidates", return_value=0), \
             mock.patch.object(run_mod.feed_mod, "rebuild", return_value=7) as rebuild, \
             mock.patch.object(run_mod, "SeenStore") as seen:
            seen.return_value.filter_unseen.side_effect = lambda ps: ps
            result = run_mod.run(self._cfg(), day="2026-08-21", dry_run=True)

        model_client.assert_not_called()               # no screen, no judge
        write_output.assert_called_once_with(self._cfg(), result, "2026-08-21")
        rebuild.assert_called_once()
        self.assertEqual(result["counts"]["fetched"], 0)
        self.assertEqual(result["feed"]["entries"], 7)
        self.assertTrue(any("no unseen papers" in p for p in result["problems"]))

    def test_a_day_the_screen_rejects_entirely_sends_nothing_and_says_so(self):
        papers = [paper(i) for i in range(3)]
        screen = {"verdicts": [
            {"arxiv_id": p.arxiv_id, "relevant": False, "reason": "no"}
            for p in papers
        ]}
        result, clients = self._run(self._cfg(), papers, [screen], None)
        self.assertEqual(result["counts"]["relevant"], 0)
        self.assertEqual(result["counts"]["kept"], 0)
        self.assertEqual(result["papers"], [])
        self.assertEqual(clients[1].calls, [])          # the judge was never called
        self.assertTrue(any("found nothing relevant" in p for p in result["problems"]))

    def test_a_failed_judge_falls_back_to_what_the_screen_passed(self):
        papers = [paper(i) for i in range(3)]
        screen = {"verdicts": [
            {"arxiv_id": p.arxiv_id, "relevant": True, "reason": "r"} for p in papers
        ]}
        result, _ = self._run(self._cfg(top_n=2), papers, [screen],
                              ModelError("judge down"),
                              {"open_questions": [], "canon": {}})
        self.assertEqual(result["counts"]["kept"], 2)   # capped at top_n
        self.assertTrue(any("falling back to the screen" in p
                            for p in result["problems"]))

    def test_the_pre_sort_caps_a_day_larger_than_screen_n(self):
        papers = [paper(i) for i in range(30)]
        screen = {"verdicts": [
            {"arxiv_id": p.arxiv_id, "relevant": False, "reason": "no"}
            for p in papers[:10]
        ]}
        result, clients = self._run(self._cfg(screen_n=10), papers, [screen], None)
        self.assertEqual(result["counts"]["screened"], 10)   # not 30
        screened_user = clients[0].calls[0]["user"]
        self.assertNotIn(papers[20].arxiv_id, screened_user)


class TestCandidatesCsvSchema(unittest.TestCase):
    def test_a_stale_header_is_refused_not_silently_misaligned(self):
        """Appending under an old header would write values under wrong names."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "candidates.csv"
            stale = [c if c != "screen_relevant" else "from_random"
                     for c in canon.CANDIDATE_COLUMNS]
            with path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=stale).writeheader()
            row = canon.to_canon_row(
                paper=paper(1), tags={}, summary="s", similarity=0.5,
                similarity_rank=1, nearest_anchor_id="a", significance=3,
                novelty=3, screen_relevant=True, first_seen="2026-08-21")
            with self.assertRaises(ValueError) as ctx:
                canon.append_candidates(path, [row])
            self.assertIn("stale header", str(ctx.exception))

    def test_the_committed_candidates_file_matches_the_current_schema(self):
        path = Path(__file__).resolve().parent.parent / "data" / "canon" / "candidates.csv"
        if not path.exists():
            self.skipTest("no candidates.csv committed")
        header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
        self.assertEqual(header, canon.CANDIDATE_COLUMNS)


class TestEffortSplit(unittest.TestCase):
    def test_the_screen_runs_at_its_own_effort(self):
        """Reasoning is a third of the screen's cost and it answers yes/no."""
        from arxiv_feed.config import load_config
        cfg = load_config("config.yaml")
        self.assertEqual(cfg.screen_effort, "low")
        self.assertNotEqual(cfg.screen_effort, cfg.effort)

    def test_a_bad_screen_effort_is_refused(self):
        from arxiv_feed.config import ConfigError, load_config
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "c.yaml"
            path.write_text(
                "categories: [cs.MA]\nanchors: ['1', '2']\nprofile: p\n"
                "screen_effort: turbo\n", encoding="utf-8")
            with self.assertRaises(ConfigError) as ctx:
                load_config(path)
            self.assertIn("screen_effort", str(ctx.exception))

    def test_the_three_calls_have_distinct_labels(self):
        """call 2 (judge) and call 2 (<id>) collided in the live run."""
        from arxiv_feed import judge, questions
        judge_src = Path(judge.__file__).read_text(encoding="utf-8")
        q_src = Path(questions.__file__).read_text(encoding="utf-8")
        self.assertIn('label="call 2 (judge)"', judge_src)
        self.assertIn('label=f"call 3 (', q_src)
        self.assertNotIn('label=f"call 2 (', q_src)


class TestScoresAreRankingOnly(unittest.TestCase):
    """Significance and novelty order Part 2. They are never shown."""

    def _result(self):
        return {
            "date": "2026-08-20",
            "counts": {"fetched": 185, "unseen": 155, "screened": 155,
                       "relevant": 2, "kept": 2, "anchors": 20},
            "config": {"screen_n": 200, "top_n": 10,
                       "screen_model": "anthropic/claude-haiku-4.5",
                       "judge_model": "anthropic/claude-sonnet-5"},
            "papers": [{
                "arxiv_id": "2608.20231", "title": "Growth Without Us",
                "authors": ["A B"], "url": "https://arxiv.org/abs/2608.20231",
                "similarity": 0.922, "nearest_anchor_id": "2501.16946",
                "nearest_anchor_title": "Gradual Disempowerment",
                "significance": 3, "novelty": 4,
                "one_sentence": "Models a post-AGI economy.",
                "open_questions": [{"question": "Q?", "label": "approachable",
                                    "reason": "public data"}]}],
            "problems": [],
            "email": {"sent": False, "error": None, "dry_run": True},
        }

    def test_nothing_justifies_a_paper_in_the_text_email(self):
        text = emailer.render_text(self._result())
        for token in ("significance", "novelty", "3/5", "4/5", "similarity", "0.92"):
            self.assertNotIn(token, text)
        self.assertIn("Models a post-AGI economy.", text)   # the sentence stays
        # The anchor survives as a bearing, without its number.
        self.assertIn("nearest in your canon: Gradual Disempowerment", text)

    def test_nothing_justifies_a_paper_in_the_html_or_feed_body(self):
        html = emailer.render_body_html(self._result())
        for token in ("significance", "novelty", "3/5", "4/5", "similarity", "0.92"):
            self.assertNotIn(token, html)
        self.assertIn("Growth Without Us", html)
        self.assertIn("nearest in your canon:", html)

    def test_the_scores_still_rank(self):
        """Removing them from the render must not remove them from the sort."""
        papers = [paper(i) for i in range(3)]
        cands = preselect(papers, np.vstack([unit([1, 0, 0, 0])] * 3), store())
        j = {papers[0].arxiv_id: {"significance": 1, "novelty": 1, "one_sentence": ""},
             papers[1].arxiv_id: {"significance": 5, "novelty": 5, "one_sentence": ""},
             papers[2].arxiv_id: {"significance": 3, "novelty": 3, "one_sentence": ""}}
        ranked = judge.rank(cands, j, top_n=3)
        self.assertEqual([c.paper.arxiv_id for c in ranked],
                         [papers[1].arxiv_id, papers[2].arxiv_id, papers[0].arxiv_id])

    def test_the_archive_still_carries_them(self):
        """Hidden on the frontend, auditable in the record."""
        row = canon.to_canon_row(
            paper=paper(1), tags={}, summary="s", similarity=0.9,
            similarity_rank=1, nearest_anchor_id="a", significance=3,
            novelty=4, screen_relevant=True, first_seen="2026-08-20")
        self.assertEqual(row["significance"], 3)
        self.assertEqual(row["novelty"], 4)


class TestScreenIncludesPolicy(unittest.TestCase):
    def test_the_prompt_says_yes_to_governance_and_policy(self):
        """The live recall test rejected 'Regulating AI Agents'. That was wrong."""
        self.assertIn("Governance and policy work counts as steering", screen.SYSTEM)
        self.assertNotIn("say no to governance", screen.SYSTEM)

    def test_scale_not_rule_making_is_the_test(self):
        """Compliance tooling for one model must still be a no."""
        self.assertIn("the scale of the subject", screen.SYSTEM)


class TestFrontendTextHasAStyle(unittest.TestCase):
    def test_both_writing_calls_carry_the_same_spec(self):
        from arxiv_feed import judge as j, questions as q
        from arxiv_feed.style import PLAIN_ENGLISH
        for name, system in (("judge", j.SYSTEM), ("questions", q.SYSTEM)):
            with self.subTest(call=name):
                self.assertIn(PLAIN_ENGLISH, system)

    def test_the_spec_bans_the_words_that_matter(self):
        from arxiv_feed.style import PLAIN_ENGLISH
        for word in ("novel", "significant", "state-of-the-art", "metaphor"):
            self.assertIn(word, PLAIN_ENGLISH)
        self.assertIn("Active voice", PLAIN_ENGLISH)
        self.assertIn("Present tense", PLAIN_ENGLISH)


class TestSummaryDoesNotArgue(unittest.TestCase):
    def test_the_judge_is_told_to_describe_not_assess(self):
        """First draft invited advocacy and got it -- see the style test."""
        self.assertIn("Describe the work. Do not assess it.", judge.SYSTEM)
        self.assertIn("not explain why it was selected", judge.SYSTEM)

    def test_the_summary_is_capped(self):
        self.assertIn("At most three\nsentences, each under 20 words", judge.SYSTEM)


class TestEmbeddingStaysOffTheGPU(unittest.TestCase):
    def test_cpu_is_the_default(self):
        """Production is GitHub Actions, which has no GPU."""
        from arxiv_feed.config import load_config
        self.assertEqual(load_config("config.yaml").embed_device, "cpu")

    def test_the_device_reaches_the_embedder(self):
        from arxiv_feed.embed import Embedder
        self.assertEqual(Embedder("m").device, "cpu")
        self.assertEqual(Embedder("m", device="cuda").device, "cuda")
