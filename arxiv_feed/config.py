"""Config loading and validation.

One YAML file holds everything except the two secrets, which are read from the
environment: ANTHROPIC_API_KEY and SMTP_PASSWORD.
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
    email_to: str
    email_from: str
    smtp_host: str
    smtp_port: int
    smtp_user: str
    shortlist_n: int = 40
    explore_n: int = 5
    top_n: int = 10
    embed_model: str = "allenai/specter2_base"
    model: str = "claude-opus-5"
    effort: str = "medium"
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
    def finalists_csv(self) -> Path:
        return DATA_DIR / "canon" / "finalists.csv"

    def output_path(self, day: str) -> Path:
        return DATA_DIR / f"{day}.json"

    # ---- secrets -----------------------------------------------------------
    @staticmethod
    def anthropic_key() -> str | None:
        return os.environ.get("ANTHROPIC_API_KEY")

    @staticmethod
    def smtp_password() -> str | None:
        return os.environ.get("SMTP_PASSWORD")


_REQUIRED = ["categories", "anchors", "profile", "email_to"]

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

    email_to = str(raw["email_to"]).strip()
    cfg = Config(
        categories=[str(c).strip() for c in raw["categories"]],
        anchors=[str(a).strip() for a in raw["anchors"]],
        profile=str(raw["profile"]).strip(),
        email_to=email_to,
        email_from=str(raw.get("email_from") or email_to).strip(),
        smtp_host=str(raw.get("smtp_host", "smtp.gmail.com")).strip(),
        smtp_port=int(raw.get("smtp_port", 587)),
        smtp_user=str(raw.get("smtp_user") or raw.get("email_from") or email_to).strip(),
        shortlist_n=int(raw.get("shortlist_n", 40)),
        explore_n=int(raw.get("explore_n", 5)),
        top_n=int(raw.get("top_n", 10)),
        embed_model=str(raw.get("embed_model", "allenai/specter2_base")).strip(),
        model=str(raw.get("model", "claude-opus-5")).strip(),
        effort=str(raw.get("effort", "medium")).strip(),
        path=path,
    )

    if len(cfg.anchors) < 2:
        raise ConfigError("at least 2 anchors are needed for the filter to mean anything")
    if len(set(cfg.anchors)) != len(cfg.anchors):
        dupes = sorted({a for a in cfg.anchors if cfg.anchors.count(a) > 1})
        raise ConfigError(f"duplicate anchor IDs: {', '.join(dupes)}")
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
