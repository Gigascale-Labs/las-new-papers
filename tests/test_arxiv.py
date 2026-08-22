"""Tests for the arXiv API query construction."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from arxiv_feed import arxiv

_EMPTY_FEED = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
)


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


if __name__ == "__main__":
    unittest.main()
