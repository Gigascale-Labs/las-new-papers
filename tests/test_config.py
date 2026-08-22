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


if __name__ == "__main__":
    unittest.main()
