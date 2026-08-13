"""Rebuild the paper's figure deck from the artifacts on disk.

Nothing here is transcribed: every number plotted is read from a JSON or JSONL
artifact, so a figure and the analysis notes cannot disagree without the artifact
settling which of them is wrong.

Usage:
    innerj figures
"""

from __future__ import annotations

import argparse

from innerj.figures.build import main as build


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    build()


# Without this, `python -m innerj.cli.figures` imports the module, builds nothing and
# exits 0 -- a build step that silently does not build, which is indistinguishable from
# a successful one. Every other entry point here has the guard; this one lacked it.
if __name__ == "__main__":
    main()
