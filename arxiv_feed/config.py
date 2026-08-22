"""Config loading and checking.

One YAML file holds every setting. Secrets and email addresses come from the
environment instead, because this repository is public.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


class ConfigError(Exception):
    """Raised for a config file that cannot produce a valid run."""


@dataclass
class Config:
    categories: list[str]
    anchors: list[str]
    profile: str
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    shortlist_n: int = 40
    explore_n: int = 5
    top_n: int = 10
    embed_model: str = "allenai/specter2_base"
    model: str = "anthropic/claude-opus-5"
    effort: str = "medium"
    guard: dict = field(default_factory=dict)
    feed: dict = field(default_factory=dict)
    path: Path | None = field(default=None, compare=False)

    # ---- derived paths -----------------------------------------------------
    @property
    def anchors_npy(self) -> Path:
        return DATA_DIR / "anchors.npy"

    @property
    def anchors_meta(self) -> Path:
        return DATA_DIR / "anchors.meta.json"

    @property
    def seen_path(self) -> Path:
        return DATA_DIR / "seen.json"

    @property
    def candidates_csv(self) -> Path:
        return DATA_DIR / "canon" / "candidates.csv"

    @property
    def feed_path(self) -> Path:
        return DATA_DIR / "feed.xml"

    @property
    def feed_max_entries(self) -> int:
        return int(self.feed.get("max_entries", 60))

    @property
    def feed_url(self) -> str:
        """Where data/feed.xml is fetchable, once this repository serves it.

        Defaults to raw.githubusercontent.com, which needs no setup but sends
        the wrong Content-Type for a feed (measured: text/plain, not xml). Set
        feed.base_url in config.yaml to a GitHub Pages URL once Pages is
        enabled, for the correct type -- see README.
        """
        base = str(
            self.feed.get("base_url")
            or "https://raw.githubusercontent.com/one-2/las-new-papers/main"
        ).rstrip("/")
        return f"{base}/data/feed.xml"

    def output_path(self, day: str) -> Path:
        return DATA_DIR / f"{day}.json"

    # ---- secrets -----------------------------------------------------------
    @staticmethod
    def openrouter_key() -> str | None:
        return os.environ.get("OPENROUTER_API_KEY")

    @staticmethod
    def smtp_password() -> str | None:
        return os.environ.get("SMTP_PASSWORD")

    @staticmethod
    def lakera_key() -> str | None:
        return os.environ.get("LAKERA_GUARD_API_KEY")

    # ---- addresses ---------------------------------------------------------
    #
    # Never stored in config.yaml. This is a public repository, and an address
    # committed to one is scraped. They come from the environment, and nothing
    # writes them to the archive files the daily job commits.
    @staticmethod
    def email_to() -> str:
        return os.environ.get("FEED_EMAIL_TO", "").strip()

    @classmethod
    def email_from(cls) -> str:
        return os.environ.get("FEED_EMAIL_FROM", "").strip() or cls.email_to()

    @classmethod
    def smtp_user(cls) -> str:
        return os.environ.get("SMTP_USER", "").strip() or cls.email_from()

    # ---- guard settings, with the defaults spelled out ---------------------
    @property
    def guard_enabled(self) -> bool:
        return bool(self.guard.get("enabled", True))

    @property
    def guard_on_error(self) -> str:
        return str(self.guard.get("on_error", "allow"))

    @property
    def guard_project_id(self) -> str | None:
        return self.guard.get("project_id") or None

    @property
    def guard_endpoint(self) -> str:
        return str(self.guard.get("endpoint") or "https://api.lakera.ai/v2/guard")

    @property
    def guard_timeout(self) -> float:
        return float(self.guard.get("timeout_seconds", 20))

    @property
    def max_title_chars(self) -> int:
        return int(self.guard.get("max_title_chars", 500))

    @property
    def max_abstract_chars(self) -> int:
        return int(self.guard.get("max_abstract_chars", 6000))


_REQUIRED = ["categories", "anchors", "profile"]

_EFFORTS = {"low", "medium", "high", "xhigh", "max"}


def load_config(path: str | Path = REPO_ROOT / "config.yaml") -> Config:
    """Read and validate the config file.

    Fails loudly here rather than three steps into a run: a bad anchor ID or an
    empty category list is cheaper to catch before the embedding model loads.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"no config file at {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    missing = [k for k in _REQUIRED if not raw.get(k)]
    if missing:
        raise ConfigError(f"{path}: missing or empty required key(s): {', '.join(missing)}")

    cfg = Config(
        categories=[str(c).strip() for c in raw["categories"]],
        anchors=[str(a).strip() for a in raw["anchors"]],
        profile=str(raw["profile"]).strip(),
        smtp_host=str(raw.get("smtp_host", "smtp.gmail.com")).strip(),
        smtp_port=int(raw.get("smtp_port", 587)),
        shortlist_n=int(raw.get("shortlist_n", 40)),
        explore_n=int(raw.get("explore_n", 5)),
        top_n=int(raw.get("top_n", 10)),
        embed_model=str(raw.get("embed_model", "allenai/specter2_base")).strip(),
        model=str(raw.get("model", "claude-opus-5")).strip(),
        effort=str(raw.get("effort", "medium")).strip(),
        guard=dict(raw.get("guard") or {}),
        feed=dict(raw.get("feed") or {}),
        path=path,
    )

    if len(cfg.anchors) < 2:
        raise ConfigError("at least 2 anchors are needed for the filter to mean anything")
    if len(set(cfg.anchors)) != len(cfg.anchors):
        dupes = sorted({a for a in cfg.anchors if cfg.anchors.count(a) > 1})
        raise ConfigError(f"duplicate anchor IDs: {', '.join(dupes)}")
    if cfg.guard_on_error not in ("allow", "block"):
        raise ConfigError(
            f"guard.on_error must be 'allow' or 'block', got {cfg.guard_on_error!r}"
        )
    for key in ("email_to", "email_from", "smtp_user"):
        if raw.get(key):
            raise ConfigError(
                f"{path}: {key} must not be set in this file -- it is public. "
                f"Use the FEED_EMAIL_TO / FEED_EMAIL_FROM / SMTP_USER environment "
                f"variables instead."
            )
    if cfg.effort not in _EFFORTS:
        raise ConfigError(f"effort must be one of {sorted(_EFFORTS)}, got {cfg.effort!r}")
    for n in ("shortlist_n", "explore_n", "top_n"):
        if getattr(cfg, n) < 0:
            raise ConfigError(f"{n} must be >= 0")
    if cfg.top_n > cfg.shortlist_n + cfg.explore_n:
        raise ConfigError(
            f"top_n ({cfg.top_n}) exceeds the shortlist it selects from "
            f"({cfg.shortlist_n} + {cfg.explore_n})"
        )
    return cfg


def anchor_count_warning(cfg: Config) -> str | None:
    """The spec's 20-40 anchor range is advice, not a hard limit -- warn, don't fail."""
    n = len(cfg.anchors)
    if n < 20:
        return f"only {n} anchors configured; the spec suggests 20-40 for field coverage"
    if n > 40:
        return f"{n} anchors configured; the spec suggests 20-40 (more is slower, not wrong)"
    return None
