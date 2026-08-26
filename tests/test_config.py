"""Tests for config loading."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from arxiv_feed.config import load_config

_MINIMAL = """
categories: [cs.MA]
anchors: ["1234.56789", "2345.67890"]
profile: "test profile"
"""


class TestSearchQueries(unittest.TestCase):
    def _load(self, extra_yaml: str = ""):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config.yaml"
            path.write_text(_MINIMAL + extra_yaml, encoding="utf-8")
            return load_config(path)

    def test_defaults_to_empty(self):
        self.assertEqual(self._load().search_queries, [])

    def test_reads_and_strips_the_list(self):
        cfg = self._load('search_queries: ["  gradual disempowerment  ", "cooperative ai"]\n')
        self.assertEqual(cfg.search_queries, ["gradual disempowerment", "cooperative ai"])


class TestFeedUrls(unittest.TestCase):
    """The feed's two addresses are different things: where it lives, and
    where its days are read. Entry links are built from the second."""

    def _load(self, extra_yaml: str = ""):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config.yaml"
            path.write_text(_MINIMAL + extra_yaml, encoding="utf-8")
            return load_config(path)

    def test_the_committed_config_points_at_the_papers_page(self):
        cfg = load_config("config.yaml")
        self.assertEqual(cfg.feed_site_url, "https://largeagentsystems.org/papers")
        self.assertEqual(cfg.feed_url, "https://largeagentsystems.org/papers/feed.xml")

    def test_self_url_names_the_feed_outright(self):
        cfg = self._load('feed:\n  self_url: https://example.com/f.xml\n')
        self.assertEqual(cfg.feed_url, "https://example.com/f.xml")

    def test_base_url_still_works_as_the_older_spelling(self):
        cfg = self._load('feed:\n  base_url: https://example.com/repo\n')
        self.assertEqual(cfg.feed_url, "https://example.com/repo/data/feed.xml")

    def test_self_url_wins_over_base_url(self):
        cfg = self._load(
            'feed:\n  base_url: https://example.com/repo\n'
            '  self_url: https://example.com/f.xml\n')
        self.assertEqual(cfg.feed_url, "https://example.com/f.xml")

    def test_trailing_slashes_do_not_double_up(self):
        cfg = self._load('feed:\n  site_url: https://example.com/papers/\n')
        self.assertEqual(cfg.feed_site_url, "https://example.com/papers")


if __name__ == "__main__":
    unittest.main()
