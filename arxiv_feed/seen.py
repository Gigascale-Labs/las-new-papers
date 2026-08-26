"""The seen list: arXiv IDs already sent.

Plain JSON. About ten new IDs a day, so the file stays small and readable.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


class SeenStore:
    def __init__(self, path: Path):
        self.path = path
        self._entries: dict[str, str] = {}      # arxiv_id -> ISO date first sent
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._entries = dict(raw.get("sent", {}))
        except Exception as exc:
            # A corrupt seen list must not stop the run: worst case is one
            # repeated paper, which is better than losing the day entirely.
            log.warning("seen list at %s unreadable (%s); starting empty", self.path, exc)
            self._entries = {}

    def __contains__(self, arxiv_id: str) -> bool:
        return arxiv_id in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def filter_unseen(self, papers: list) -> list:
        return [p for p in papers if p.arxiv_id not in self._entries]

    def mark(self, arxiv_ids: list[str], day: str | None = None) -> None:
        stamp = day or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        for a in arxiv_ids:
            self._entries.setdefault(a, stamp)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "count": len(self._entries),
            # Sorted so a daily commit diff shows the additions, not a reshuffle.
            "sent": dict(sorted(self._entries.items())),
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.info("seen list saved: %d IDs", len(self._entries))
