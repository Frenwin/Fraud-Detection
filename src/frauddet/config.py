"""Load configs/config.yaml and resolve project paths."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    raw: Path
    processed: Path
    artifacts: Path
    reports: Path

    @property
    def results(self) -> Path:
        return self.reports / "results"

    @property
    def figures(self) -> Path:
        return self.reports / "figures"

    @property
    def production(self) -> Path:
        return self.artifacts / "production"

    def ensure(self) -> "Paths":
        for p in (self.raw, self.processed, self.artifacts, self.results, self.figures, self.production):
            p.mkdir(parents=True, exist_ok=True)
        return self


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict:
    path = path or os.environ.get("FRAUDDET_CONFIG", str(ROOT / "configs" / "config.yaml"))
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # env override so Docker / CI can point at another database
    cfg["db_url"] = os.environ.get("FRAUDDET_DB_URL", cfg["db_url"])
    return cfg


def get_paths(cfg: dict | None = None) -> Paths:
    cfg = cfg or load_config()
    p = cfg["paths"]
    return Paths(*(ROOT / p[k] for k in ("raw", "processed", "artifacts", "reports"))).ensure()
