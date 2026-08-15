"""Where things live. Repo-relative, with no configuration.

Absolute paths baked into source are why a repository only runs on the machine it was
written on, so every location here is derived from the package's own position. There is
no ``.env`` and nothing to set up: a fresh checkout runs as-is.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Datasets, sweeps and result artifacts. A few GB for a full run.
DATA_ROOT = REPO_ROOT / "data"

#: Long-running job logs.
LOG_DIR = REPO_ROOT / "logs"

#: Figure output. ``make sync-figures`` copies the deck into the paper from here.
FIGURE_DIR = REPO_ROOT / "figures"

#: The paper source, for ``innerj audit``. The paper is not part of this repository, so
#: this is the one path that legitimately points outside it; ``--tex`` overrides it.
PAPER_TEX = REPO_ROOT / "paper" / "iclr" / "main.tex"
