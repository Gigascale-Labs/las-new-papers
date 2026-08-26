"""Config loading and checking.

One YAML file holds every setting. Secrets come from the environment
instead, because this repository is public.
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
    search_queries: list[str] = field(default_factory=list)
    screen_n: int = 200
    screen_batch_size: int = 25
    top_n: int = 10
    embed_model: str = "allenai/specter2_base"
    embed_device: str = "cpu"
    screen_model: str = "anthropic/claude-haiku-4.5"
    judge_model: str = "anthropic/claude-sonnet-5"
    model: str = "anthropic/claude-opus-5"
    effort: str = "medium"
    screen_effort: str = "low"
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
            or "https://raw.githubusercontent.com/Gigascale-Labs/las-new-papers/main"
        ).rstrip("/")
        return f"{base}/data/feed.xml"

    def output_path(self, day: str) -> Path:
        return DATA_DIR / f"{day}.json"

    # ---- secrets -----------------------------------------------------------
    @staticmethod
    def openrouter_key() -> str | None:
        return os.environ.get("OPENROUTER_API_KEY")

    @staticmethod
    def lakera_key() -> str | None:
        return os.environ.get("LAKERA_GUARD_API_KEY")

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
        search_queries=[str(q).strip() for q in raw.get("search_queries") or []],
        screen_n=int(raw.get("screen_n", 200)),
        screen_batch_size=int(raw.get("screen_batch_size", 25)),
        top_n=int(raw.get("top_n", 10)),
        embed_model=str(raw.get("embed_model", "allenai/specter2_base")).strip(),
        embed_device=str(raw.get("embed_device", "cpu")).strip(),
        screen_model=str(raw.get("screen_model", "anthropic/claude-haiku-4.5")).strip(),
        judge_model=str(raw.get("judge_model", "anthropic/claude-sonnet-5")).strip(),
        model=str(raw.get("model", "claude-opus-5")).strip(),
        effort=str(raw.get("effort", "medium")).strip(),
        screen_effort=str(raw.get("screen_effort", "low")).strip(),
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
    for name in ("effort", "screen_effort"):
        value = getattr(cfg, name)
        if value not in _EFFORTS:
            raise ConfigError(
                f"{name} must be one of {sorted(_EFFORTS)}, got {value!r}")
    for n in ("screen_n", "top_n"):
        if getattr(cfg, n) < 0:
            raise ConfigError(f"{n} must be >= 0")
    if cfg.screen_batch_size < 1:
        raise ConfigError("screen_batch_size must be >= 1")
    if cfg.top_n > cfg.screen_n:
        raise ConfigError(
            f"top_n ({cfg.top_n}) exceeds the number of papers screened "
            f"({cfg.screen_n})"
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
