"""Covers the kept-paper vector file: rounding, and that a write never subtracts.

Run: python -m unittest discover tests
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from arxiv_feed import vectors


def setUpModule():
    import logging
    logging.disable(logging.CRITICAL)


def tearDownModule():
    import logging
    logging.disable(logging.NOTSET)


class TestWrite(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "embeddings" / "2026-08-26.json"

    def tearDown(self):
        self.dir.cleanup()

    def read(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def test_writes_one_row_per_id_and_rounds(self):
        n = vectors.write(
            self.path, "2026-08-26", "stub-model",
            {"2608.00001": np.array([0.123456, -0.987654], dtype=np.float32)},
        )
        self.assertEqual(n, 1)
        payload = self.read()
        self.assertEqual(payload["date"], "2026-08-26")
        self.assertEqual(payload["model"], "stub-model")
        self.assertEqual(payload["dim"], 2)
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["vectors"]["2608.00001"], [0.1235, -0.9877])

    def test_a_paper_with_no_vector_is_skipped_not_nulled(self):
        # A re-run: result["papers"] holds the paper because the merge brought
        # it back from the first run, and this run never encoded it.
        vectors.write(
            self.path, "2026-08-26", "stub-model",
            {"2608.00001": np.array([1.0, 0.0]), "2608.00002": None},
        )
        self.assertEqual(list(self.read()["vectors"]), ["2608.00001"])

    def test_a_second_run_of_a_day_adds_and_never_subtracts(self):
        vectors.write(self.path, "2026-08-26", "m", {"a": np.array([1.0, 0.0])})
        vectors.write(self.path, "2026-08-26", "m", {"b": np.array([0.0, 1.0])})
        payload = self.read()
        self.assertEqual(sorted(payload["vectors"]), ["a", "b"])
        self.assertEqual(payload["count"], 2)

    def test_this_run_wins_a_repeated_id(self):
        vectors.write(self.path, "2026-08-26", "m", {"a": np.array([1.0, 0.0])})
        vectors.write(self.path, "2026-08-26", "m", {"a": np.array([0.0, 1.0])})
        self.assertEqual(self.read()["vectors"]["a"], [0.0, 1.0])

    def test_dim_and_count_describe_the_merged_file_not_the_run(self):
        vectors.write(self.path, "2026-08-26", "m", {"a": np.array([1.0, 0.0, 0.0])})
        # A run that encoded nothing new still rewrites the file.
        vectors.write(self.path, "2026-08-26", "m", {"b": None})
        payload = self.read()
        self.assertEqual(payload["dim"], 3)
        self.assertEqual(payload["count"], 1)

    def test_a_file_for_another_day_is_replaced_not_merged(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(
            {"date": "2026-08-25", "model": "m", "vectors": {"old": [1.0, 0.0]}}
        ), encoding="utf-8")
        vectors.write(self.path, "2026-08-26", "m", {"new": np.array([0.0, 1.0])})
        self.assertEqual(list(self.read()["vectors"]), ["new"])

    def test_an_unreadable_file_does_not_stop_the_write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("{not json", encoding="utf-8")
        n = vectors.write(self.path, "2026-08-26", "m", {"a": np.array([1.0, 0.0])})
        self.assertEqual(n, 1)
        self.assertEqual(list(self.read()["vectors"]), ["a"])

    def test_a_day_that_kept_nothing_writes_an_empty_file(self):
        n = vectors.write(self.path, "2026-08-26", "m", {})
        self.assertEqual(n, 0)
        payload = self.read()
        self.assertEqual(payload["vectors"], {})
        self.assertEqual(payload["dim"], 0)
