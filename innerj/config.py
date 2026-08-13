"""Paths and environment, read once from ``.env`` rather than hardcoded.

Absolute paths baked into source are why a repository only runs on the machine it was
written
on. Copy ``.env.example`` to ``.env``; it is gitignored.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal ``.env`` reader: ``KEY=value``, ``#`` comments, no interpolation.

    Existing environment variables win, so an explicit ``export`` still overrides the
    file.
    Handles the three things a naive reader gets wrong: an ``export`` prefix, quotes,
    and a
    trailing ``# comment`` -- a quoted value keeps its ``#``.
    """
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip().removeprefix("export ").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if value[:1] in {"'", '"'} and value[-1:] == value[:1] and len(value) > 1:
            value = value[1:-1]
        else:
            value = value.partition(" #")[0].strip()
        os.environ.setdefault(key.strip(), value)


_load_dotenv(REPO_ROOT / ".env")


def _path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


#: Where datasets, sweeps and result artifacts are written. Large; keep it off the
#: repo volume on a shared machine.
DATA_ROOT = _path("INNERJ_DATA_ROOT", REPO_ROOT / "data")

#: Long-running job logs.
LOG_DIR = _path("INNERJ_LOG_DIR", REPO_ROOT / "logs")

#: Figure output. Repo-relative by default; point it at the paper's figure directory
#: to rebuild the deck in place.
FIGURE_DIR = _path("INNERJ_FIGURE_DIR", REPO_ROOT / "figures")

#: The paper source, for ``innerj audit``. The paper is not part of this repository, so
#: this is the one path that legitimately points outside it.
PAPER_TEX = _path("INNERJ_PAPER_TEX", REPO_ROOT / "paper" / "iclr" / "main.tex")
