"""Stub a dataset for release, or rebuild it from FLORES.

Usage:
    innerj flores --stub data/language/*.jsonl
    innerj flores --rehydrate data/release/*.jsonl

A released record file quotes FLORES-200 verbatim and is therefore a CC BY-SA 4.0
derivative of it. ``--stub`` writes a copy carrying only this project's own fields plus
a reference to each passage; ``--rehydrate`` rebuilds the byte-identical original from
the cached FLORES snapshot. See :mod:`innerj.tasks.flores_ref`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from innerj import console
from innerj.tasks import flores_ref as fr
from innerj.tasks.language import LANGUAGE_NAMES, load_parallel


def _read(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"{path} is empty")
    return rows


def _dump(records: list[dict]) -> str:
    """Serialise as the generator does -- what makes the round trip exact."""
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--stub", action="store_true",
                      help="remove FLORES text, leaving a reference to it")
    mode.add_argument("--rehydrate", action="store_true",
                      help="rebuild the original from the cached FLORES snapshot")
    parser.add_argument("records", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path,
                        help="write beside the inputs by default")
    parser.add_argument("--span", type=int, default=3,
                        help="sentences per passage, for --stub (default 3)")
    args = parser.parse_args()

    frame = load_parallel(sorted(LANGUAGE_NAMES))
    console.step(
        f"FLORES loaded: {frame.shape[0]} rows, {frame.shape[1] - 1} languages"
    )

    for path in args.records:
        records = _read(path)
        if args.stub:
            fr.stub(records, frame, span=args.span)
        else:
            fr.rehydrate(records, frame)
        text = _dump(records)

        out_dir = args.out_dir or path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / path.name
        if out.resolve() == path.resolve():
            raise SystemExit(
                f"{out} would overwrite its own input; pass --out-dir"
            )
        out.write_text(text)
        digest = hashlib.sha256(text.encode()).hexdigest()[:8]
        console.wrote(f"{out}  ({len(records)} records, sha256 {digest})")


if __name__ == "__main__":
    main()
