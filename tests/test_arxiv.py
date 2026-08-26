"""Tests for the arXiv API query construction and its retry rule."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from arxiv_feed import arxiv

_EMPTY_FEED = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
)


def setUpModule():
    # The retry-exhaustion tests log warnings by design; silence them so a
    # passing run is quiet and a real problem stands out.
    import logging
    logging.disable(logging.CRITICAL)


def tearDownModule():
    import logging
    logging.disable(logging.NOTSET)


def _http_error(status_code, headers=None):
    """A requests.HTTPError shaped like what resp.raise_for_status() raises."""
    import requests

    response = MagicMock()
    response.status_code = status_code
    response.headers = headers or {}
    return requests.exceptions.HTTPError(f"{status_code} error", response=response)


class TestQueryConstruction(unittest.TestCase):
    def _fetch(self, categories, search_queries=None):
        captured = {}

        def fake_get(params):
            captured.update(params)
            return _EMPTY_FEED

        with patch.object(arxiv, "_get", side_effect=fake_get):
            arxiv.fetch_new_papers(categories, day="2026-08-20", search_queries=search_queries)
        return captured["search_query"]

    def test_categories_only(self):
        self.assertEqual(
            self._fetch(["cs.MA", "cs.AI"]),
            "(cat:cs.MA OR cat:cs.AI) AND submittedDate:[202608200000 TO 202608202359]",
        )

    def test_no_search_queries_matches_the_categories_only_shape(self):
        self.assertEqual(self._fetch(["cs.MA"], search_queries=[]), self._fetch(["cs.MA"]))

    def test_search_queries_are_ored_in_alongside_categories(self):
        self.assertEqual(
            self._fetch(["cs.MA"], search_queries=["gradual disempowerment", "cooperative ai"]),
            '((cat:cs.MA) OR (all:"gradual disempowerment" OR all:"cooperative ai")) '
            "AND submittedDate:[202608200000 TO 202608202359]",
        )


class TestRateLimitBackoff(unittest.TestCase):
    """A 429 gets a longer, growing wait; anything else keeps the old fixed one."""

    def test_a_non_429_error_keeps_the_fixed_wait(self):
        self.assertEqual(arxiv._retry_wait(TimeoutError("slow"), attempt=1), arxiv._RETRY_WAIT)
        self.assertEqual(arxiv._retry_wait(TimeoutError("slow"), attempt=4), arxiv._RETRY_WAIT)

    def test_a_429_with_no_retry_after_backs_off_exponentially(self):
        exc = _http_error(429)
        waits = [arxiv._retry_wait(exc, attempt=a) for a in (1, 2, 3)]
        # Each step roughly doubles the last, allowing for the +/-15% jitter.
        self.assertLess(waits[0], waits[1])
        self.assertLess(waits[1], waits[2])
        self.assertGreater(waits[0], arxiv._RATE_LIMIT_BASE_WAIT * 0.8)

    def test_a_429_backoff_never_exceeds_the_cap(self):
        exc = _http_error(429)
        for attempt in range(1, 10):
            self.assertLessEqual(arxiv._retry_wait(exc, attempt), arxiv._RATE_LIMIT_MAX_WAIT)

    def test_a_429_honours_retry_after_when_the_server_sends_one(self):
        exc = _http_error(429, headers={"Retry-After": "45"})
        self.assertEqual(arxiv._retry_wait(exc, attempt=1), 45.0)

    def test_a_retry_after_longer_than_the_cap_is_still_capped(self):
        exc = _http_error(429, headers={"Retry-After": "9999"})
        self.assertEqual(arxiv._retry_wait(exc, attempt=1), arxiv._RATE_LIMIT_MAX_WAIT)

    def test_a_malformed_retry_after_falls_back_to_the_backoff(self):
        exc = _http_error(429, headers={"Retry-After": "not-a-number"})
        wait = arxiv._retry_wait(exc, attempt=1)
        self.assertGreater(wait, 0)
        self.assertLessEqual(wait, arxiv._RATE_LIMIT_MAX_WAIT)


class TestGetRetries(unittest.TestCase):
    """_get() itself: does it actually retry, back off, and eventually give up."""

    def setUp(self):
        arxiv._last_request = 0.0

    def test_succeeds_after_a_transient_failure(self):
        with patch("arxiv_feed.arxiv.requests.get",
                   side_effect=[TimeoutError("slow"), _OkResponse(_EMPTY_FEED)]) as get, \
             patch("arxiv_feed.arxiv.time.sleep"):
            self.assertEqual(arxiv._get({}), _EMPTY_FEED)
        self.assertEqual(get.call_count, 2)

    def test_sends_an_identifying_user_agent(self):
        with patch("arxiv_feed.arxiv.requests.get", return_value=_OkResponse(_EMPTY_FEED)) as get, \
             patch("arxiv_feed.arxiv.time.sleep"):
            arxiv._get({})
        self.assertIn("las-new-papers", get.call_args.kwargs["headers"]["User-Agent"])

    def test_gives_up_after_every_attempt_fails(self):
        with patch("arxiv_feed.arxiv.requests.get", side_effect=_http_error(429)) as get, \
             patch("arxiv_feed.arxiv.time.sleep"):
            with self.assertRaises(arxiv.ArxivError):
                arxiv._get({})
        self.assertEqual(get.call_count, arxiv._ATTEMPTS)


class _OkResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


if __name__ == "__main__":
    unittest.main()
