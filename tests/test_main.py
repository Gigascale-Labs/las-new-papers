"""Tests for main.py's day loop.

One day's arXiv fetch failing for good must not stop the rest of the run --
least of all the primary, requested day, which is always last in the list.

A model stage that had work and returned nothing must fail the job, on any
day, even when the feed still has entries from earlier days.
"""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import main
from arxiv_feed.arxiv import ArxivError


def setUpModule():
    import logging
    logging.disable(logging.CRITICAL)


def tearDownModule():
    import logging
    logging.disable(logging.NOTSET)


def _result(date, feed_entries=1, failed=()):
    return {
        "date": date,
        "counts": {"fetched": 1, "screened": 1, "relevant": 1, "kept": 1},
        "papers": [],
        "problems": [],
        "failed": list(failed),
        "feed": {"entries": feed_entries},
    }


def _run_main(run_side_effect, backfill_days_return, argv=()):
    out = io.StringIO()
    with patch("main.load_config"), \
         patch("main.backfill_days", return_value=backfill_days_return), \
         patch("main.run", side_effect=run_side_effect) as run_mock, \
         patch("main.arxiv.default_day", return_value="2026-08-25"), \
         redirect_stdout(out):
        code = main.main(list(argv))
    return code, run_mock, out.getvalue()


class TestDayLoopFaultIsolation(unittest.TestCase):
    def test_a_backfill_days_fetch_failure_does_not_stop_the_primary_day(self):
        def side_effect(cfg, day, dry_run, rebuild_anchors):
            if day == "2026-08-22":
                raise ArxivError("still rate-limited")
            return _result(day)

        code, run_mock, _ = _run_main(side_effect, ["2026-08-21", "2026-08-22"])

        # All three days were attempted, in order -- the failure on the
        # middle one did not stop 2026-08-25 from being reached.
        self.assertEqual([c.kwargs["day"] for c in run_mock.call_args_list],
                         ["2026-08-21", "2026-08-22", "2026-08-25"])
        self.assertEqual(code, 0)   # the primary day still delivered

    def test_the_primary_day_itself_failing_is_still_reported_as_a_failure(self):
        def side_effect(cfg, day, dry_run, rebuild_anchors):
            if day == "2026-08-25":
                raise ArxivError("still rate-limited")
            return _result(day)

        code, _, _ = _run_main(side_effect, [])
        self.assertEqual(code, 1)

    def test_every_day_failing_still_finishes_instead_of_crashing(self):
        code, run_mock, _ = _run_main(
            lambda cfg, day, dry_run, rebuild_anchors: (_ for _ in ()).throw(
                ArxivError("down")),
            ["2026-08-21"],
        )
        self.assertEqual(run_mock.call_count, 2)   # both days attempted
        self.assertEqual(code, 1)


class TestModelFailuresFailTheJob(unittest.TestCase):
    """From 2026-09-04 to 2026-09-09 an exhausted OpenRouter key failed every
    screening call. The feed still held 21 entries from earlier days, so each
    run exited 0 and nobody was told."""

    FAILED = ["every screening call failed; 200 paper(s) unscreened"]

    def test_a_failed_stage_fails_the_job_even_when_the_feed_has_entries(self):
        code, _, _ = _run_main(
            lambda cfg, day, dry_run, rebuild_anchors:
                _result(day, feed_entries=21, failed=self.FAILED),
            [],
        )
        self.assertEqual(code, 1)

    def test_a_failed_backfill_day_fails_the_job(self):
        def side_effect(cfg, day, dry_run, rebuild_anchors):
            return _result(day, failed=self.FAILED if day == "2026-08-22" else ())

        code, run_mock, _ = _run_main(side_effect, ["2026-08-22"])
        self.assertEqual(run_mock.call_count, 2)   # the primary day still ran
        self.assertEqual(code, 1)

    def test_a_dry_run_with_a_failed_stage_still_fails(self):
        code, _, _ = _run_main(
            lambda cfg, day, dry_run, rebuild_anchors: _result(day, failed=self.FAILED),
            [],
            argv=["--dry-run"],
        )
        self.assertEqual(code, 1)

    def test_the_failure_is_printed_for_the_issue_to_quote(self):
        _, _, out = _run_main(
            lambda cfg, day, dry_run, rebuild_anchors: _result(day, failed=self.FAILED),
            [],
        )
        self.assertIn("  FAILED: every screening call failed", out)
        self.assertIn("model calls failed for 2026-08-25", out)


if __name__ == "__main__":
    unittest.main()
