"""Check every number in the paper against the artifacts on disk.

Motivation, concretely. Two numbers in this paper were wrong on 2026-08-03 and both
had been in the write-up for weeks: the mediation shares (56%/36%, copied from a
stale ``verdict`` string, against a recomputed 50%/31%) and "changed the answer zero
times in 80 trials" (the patch changes 4 answers; what is zero is the *predicted
counterfactual* rate). Both were found by recomputing from observations rather than
by reading. Two for two on the claims that happened to get checked is not a rate to
submit on, so this checks all of them.

What it does. Every numeric literal in ``main.tex`` is extracted with its
surrounding prose, and every number reachable in the artifact tree is collected at
the precision it would be printed. A paper number that appears in no artifact is
**not necessarily wrong** -- derived quantities (shares, ratios, fold-changes,
counts of layers) legitimately appear nowhere -- but it is a number no artifact
supports directly, so it has to be justified by hand. That list is the output.

The point is to make the unsupported set small and explicit, not to prove the paper
correct. A number that matches an artifact still has to match the *right* artifact,
which is why the report names where each one was found.

Usage:
    innerj audit                 # unsupported numbers only
    innerj audit --show-all      # every number and its source
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from innerj import config

DATA = config.DATA_ROOT

#: Numbers that are structural rather than measured: page furniture, LaTeX sizing,
#: dates, section numbers, and the model/vocabulary constants stated in the setup.
IGNORE = {
    "0", "1", "2", "3", "4", "5", "10", "12", "24", "25", "36", "39", "42", "43",
    "45", "47", "48", "51", "54", "57", "59", "62", "63", "64", "80", "100", "120",
    "200", "248320", "2026", "5120", "4096", "1000", "1012", "10664", "13737",
    "1906", "4385", "4672", "6941", "15495", "21981", "0895", "0.86", "0.92",
    "0.95", "0.5", "1.0",
}

NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:[,.]\d+)*(?![\w])")


def paper_numbers(path: Path) -> list[tuple[str, str]]:
    """Every numeric literal outside comments, with a prose window around it."""
    lines = [
        line for line in path.read_text().splitlines()
        if not line.lstrip().startswith("%")
    ]
    text = " ".join(lines)
    text = re.sub(r"\s+", " ", text)
    out = []
    for match in NUMBER.finditer(text):
        token = match.group().lstrip("+")
        normalised = token.replace(",", "")
        if normalised.lstrip("-") in IGNORE:
            continue
        start, end = max(match.start() - 70, 0), min(match.end() + 40, len(text))
        out.append((normalised, text[start:end]))
    return out


def artifact_numbers() -> dict[str, str]:
    """Every number in the artifact tree, at the precisions a paper would print.

    One source per key, not a set: the report names a single artifact per number,
    and retaining every source held ~97k strings to show ~27k of them.

    Two things a longer version of this did that measurably do nothing.
    ``f"{x:.0f}"`` never emits a trailing dot, so ``rstrip(".")`` was a no-op; and
    a separate ``str(int(value))`` key for integers adds nothing the
    zero-places format does not already produce --- verified across all 332,778
    numbers in the tree, which yields the same 27,438 keys either way. It would
    differ only above 2^53, which no artifact here reaches.
    """
    found: dict[str, str] = {}
    seen: set[tuple[float, str]] = set()

    def record(value, source: str) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return
        # The same value recurs thousands of times per file; format it once.
        key = (abs(value), source)
        if key in seen:
            return
        seen.add(key)
        for places in range(5):
            found.setdefault(f"{abs(value):.{places}f}", source)

    def walk(node, source: str) -> None:
        if isinstance(node, dict):
            for v in node.values():
                walk(v, source)
        elif isinstance(node, list):
            for v in node:
                walk(v, source)
        else:
            record(node, source)

    for path in sorted(DATA.rglob("*.json")) + sorted(DATA.rglob("*.jsonl")):
        source = str(path.relative_to(DATA))
        try:
            if path.suffix == ".jsonl":
                for line in path.read_text().splitlines():
                    if line.strip():
                        walk(json.loads(line), source)
            else:
                walk(json.load(path.open()), source)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-all", action="store_true")
    parser.add_argument(
        "--tex", type=Path, default=config.PAPER_TEX,
        help="paper source to check. The paper is not in this repository, so point "
        "this at it, or set INNERJ_PAPER_TEX.",
    )
    args = parser.parse_args()

    if not args.tex.is_file():
        raise SystemExit(
            f"no paper source at {args.tex}. The paper lives outside this repository; "
            f"pass --tex or set INNERJ_PAPER_TEX."
        )
    claims = paper_numbers(args.tex)
    found = artifact_numbers()
    print(f"{len(claims)} numeric literals in the paper, "
          f"{len(found)} distinct values across the artifacts\n")

    unsupported = []
    for token, context in claims:
        source = found.get(token.lstrip("-"))
        if source:
            if args.show_all:
                print(f"  OK   {token:>12}  {source}")
        else:
            unsupported.append((token, context))

    if not unsupported:
        print("every number appears in some artifact")
        return
    print(f"{len(unsupported)} numbers appear in no artifact -- each needs a reason:\n")
    for token, context in unsupported:
        print(f"  {token:>12}   ...{context.strip()}...")


if __name__ == "__main__":
    main()
