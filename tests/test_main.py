"""Tests for main.py's day loop.

One day's arXiv fetch failing for good must not stop the rest of the run --
least of all the primary, requested day, which is always last in the list.
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


def _result(date, feed_entries=1):
    return {
        "date": date,
        "counts": {"fetched": 1, "screened": 1, "relevant": 1, "kept": 1},
        "papers": [],
        "problems": [],
        "feed": {"entries": feed_entries},
    }


class TestDayLoopFaultIsolation(unittest.TestCase):
    def _run_main(self, run_side_effect, backfill_days_return):
        with patch("main.load_config"), \
             patch("main.backfill_days", return_value=backfill_days_return), \
             patch("main.run", side_effect=run_side_effect) as run_mock, \
             patch("main.arxiv.default_day", return_value="2026-08-25"), \
             redirect_stdout(io.StringIO()):
            code = main.main([])
        return code, run_mock

    def test_a_backfill_days_fetch_failure_does_not_stop_the_primary_day(self):
        def side_effect(cfg, day, dry_run, rebuild_anchors):
            if day == "2026-08-22":
                raise ArxivError("still rate-limited")
            return _result(day)

        code, run_mock = self._run_main(side_effect, ["2026-08-21", "2026-08-22"])

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

        code, _ = self._run_main(side_effect, [])
        self.assertEqual(code, 1)

    def test_every_day_failing_still_finishes_instead_of_crashing(self):
        code, run_mock = self._run_main(
            lambda cfg, day, dry_run, rebuild_anchors: (_ for _ in ()).throw(
                ArxivError("down")),
            ["2026-08-21"],
        )
        self.assertEqual(run_mock.call_count, 2)   # both days attempted
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
