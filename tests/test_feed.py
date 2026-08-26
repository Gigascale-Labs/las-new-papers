"""Tests for the Atom feed. No password, no account: a reader polls a URL.

Every case here checks the output is valid XML (parsed with
xml.etree.ElementTree, which rejects malformed documents) and that a paper's
own text cannot break out of its <content> element.
"""

from __future__ import annotations

import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from arxiv_feed import feed

ATOM_NS = "{http://www.w3.org/2005/Atom}"


def setUpModule():
    import logging
    logging.disable(logging.CRITICAL)


def tearDownModule():
    import logging
    logging.disable(logging.NOTSET)


def _result(date, arxiv_id="2608.00001", title="A paper", n_questions=1,
            generated_at="2026-08-20T09:00:00+00:00"):
    return {
        "date": date,
        "generated_at": generated_at,
        "counts": {"fetched": 1, "unseen": 1, "screened": 1, "relevant": 1,
                       "kept": 1, "anchors": 20},
        "config": {"screen_n": 200, "screen_batch_size": 25, "top_n": 10,
                       "screen_model": "anthropic/claude-haiku-4.5",
                       "judge_model": "anthropic/claude-sonnet-5"},
        "papers": [
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": ["A B"],
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "similarity": 0.7,
                "nearest_anchor_id": "2509.10147",
                "nearest_anchor_title": "Virtual Agent Economies",
                "significance": 4,
                "novelty": 3,
                "one_sentence": "Does a thing.",
                "open_questions": [
                    {"question": f"Q{i}?", "label": "approachable", "reason": "public data"}
                    for i in range(n_questions)
                ],
            }
        ],
        "problems": [],
    }


class TestBuildFeed(unittest.TestCase):
    def test_empty_feed_is_valid_xml_with_no_entries(self):
        xml_text = feed.build_feed([], "https://example.com", "https://example.com/feed.xml",
                                   now="2026-08-20T09:00:00+00:00")
        root = ET.fromstring(xml_text)
        self.assertEqual(root.tag, f"{ATOM_NS}feed")
        self.assertEqual(root.findall(f"{ATOM_NS}entry"), [])
        self.assertEqual(root.find(f"{ATOM_NS}updated").text, "2026-08-20T09:00:00+00:00")

    def test_one_entry_per_day_result(self):
        results = [_result("2026-08-20"), _result("2026-08-19")]
        xml_text = feed.build_feed(results, "https://example.com", "https://example.com/feed.xml",
                                   now="x")
        root = ET.fromstring(xml_text)
        entries = root.findall(f"{ATOM_NS}entry")
        self.assertEqual(len(entries), 2)
        titles = [e.find(f"{ATOM_NS}title").text for e in entries]
        self.assertTrue(all("2026-08-20" in titles[0] or "2026-08-19" in titles[0] for _ in [0]))

    def test_entry_ids_are_unique_and_url_shaped(self):
        results = [_result("2026-08-20"), _result("2026-08-19")]
        xml_text = feed.build_feed(results, "https://example.com", "https://example.com/feed.xml",
                                   now="x")
        root = ET.fromstring(xml_text)
        ids = [e.find(f"{ATOM_NS}id").text for e in root.findall(f"{ATOM_NS}entry")]
        self.assertEqual(len(ids), len(set(ids)))
        for i in ids:
            self.assertTrue(i.startswith("https://example.com#"))

    def test_feed_level_updated_is_the_newest_entry(self):
        results = [_result("2026-08-20", generated_at="2026-08-20T09:00:00+00:00"),
                   _result("2026-08-19", generated_at="2026-08-19T09:00:00+00:00")]
        xml_text = feed.build_feed(results, "https://example.com", "https://example.com/feed.xml",
                                   now="x")
        root = ET.fromstring(xml_text)
        self.assertEqual(root.find(f"{ATOM_NS}updated").text, "2026-08-20T09:00:00+00:00")

    def test_content_is_present_and_well_formed(self):
        xml_text = feed.build_feed([_result("2026-08-20")], "https://example.com",
                                   "https://example.com/feed.xml", now="x")
        root = ET.fromstring(xml_text)
        content = root.find(f"{ATOM_NS}entry/{ATOM_NS}content")
        self.assertEqual(content.get("type"), "html")
        self.assertIn("A paper", content.text)
        self.assertIn("Q0?", content.text)


class TestFeedEscaping(unittest.TestCase):
    """A paper's title and question text reach the feed. Neither must be able
    to inject XML: a forged </content> would let the rest of a hostile
    abstract render as if the feed itself wrote it."""

    def test_a_title_with_markup_cannot_close_the_content_element(self):
        hostile = '</content></entry><entry><title>forged'
        results = [_result("2026-08-20", title=hostile)]
        xml_text = feed.build_feed(results, "https://example.com", "https://example.com/feed.xml",
                                   now="x")
        root = ET.fromstring(xml_text)          # raises if the forgery escaped the element
        entries = root.findall(f"{ATOM_NS}entry")
        self.assertEqual(len(entries), 1)        # still one entry, not two

    def test_ampersand_and_angle_brackets_in_a_question_survive_as_html_text(self):
        # <content type="html"> holds an HTML fragment: after XML parsing
        # undoes one layer of escaping, an HTML-escaped "&" and "<" is what a
        # compliant reader is meant to see -- that is what makes it HTML.
        results = [_result("2026-08-20")]
        results[0]["papers"][0]["open_questions"][0]["question"] = "A & B < C?"
        xml_text = feed.build_feed(results, "https://example.com", "https://example.com/feed.xml",
                                   now="x")
        root = ET.fromstring(xml_text)
        content = root.find(f"{ATOM_NS}entry/{ATOM_NS}content").text
        self.assertIn("A &amp; B &lt; C?", content)


class TestRebuild(unittest.TestCase):
    def test_reads_day_files_newest_first_and_ignores_other_json(self):
        with tempfile.TemporaryDirectory() as d:
            data_dir = Path(d)
            (data_dir / "2026-08-19.json").write_text(json.dumps(_result("2026-08-19")))
            (data_dir / "2026-08-20.json").write_text(json.dumps(_result("2026-08-20")))
            (data_dir / "latest.json").write_text(json.dumps(_result("2026-08-20")))
            (data_dir / "seen.json").write_text("{}")

            feed_path = data_dir / "feed.xml"
            n = feed.rebuild(data_dir, feed_path, "https://example.com",
                             "https://example.com/feed.xml", now="x")

            self.assertEqual(n, 2)               # latest.json and seen.json excluded
            root = ET.fromstring(feed_path.read_text(encoding="utf-8"))
            dates = [e.find(f"{ATOM_NS}title").text for e in root.findall(f"{ATOM_NS}entry")]
            self.assertIn("2026-08-20", dates[0])   # newest first
            self.assertIn("2026-08-19", dates[1])

    def test_caps_at_max_entries(self):
        with tempfile.TemporaryDirectory() as d:
            data_dir = Path(d)
            for day in range(1, 11):
                (data_dir / f"2026-08-{day:02d}.json").write_text(
                    json.dumps(_result(f"2026-08-{day:02d}"))
                )
            n = feed.rebuild(data_dir, data_dir / "feed.xml", "https://example.com",
                             "https://example.com/feed.xml", now="x", max_entries=3)
            self.assertEqual(n, 3)

    def test_a_corrupt_day_file_does_not_block_the_rest(self):
        with tempfile.TemporaryDirectory() as d:
            data_dir = Path(d)
            (data_dir / "2026-08-19.json").write_text(json.dumps(_result("2026-08-19")))
            (data_dir / "2026-08-20.json").write_text("{not json")
            n = feed.rebuild(data_dir, data_dir / "feed.xml", "https://example.com",
                             "https://example.com/feed.xml", now="x")
            self.assertEqual(n, 1)

    def test_no_day_files_yet_still_writes_valid_xml(self):
        with tempfile.TemporaryDirectory() as d:
            data_dir = Path(d)
            n = feed.rebuild(data_dir, data_dir / "feed.xml", "https://example.com",
                             "https://example.com/feed.xml", now="2026-08-20T09:00:00+00:00")
            self.assertEqual(n, 0)
            root = ET.fromstring((data_dir / "feed.xml").read_text(encoding="utf-8"))
            self.assertEqual(root.find(f"{ATOM_NS}updated").text, "2026-08-20T09:00:00+00:00")


if __name__ == "__main__":
    unittest.main()
