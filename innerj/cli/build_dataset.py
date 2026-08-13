"""Build a JGateBench family and write it to disk.

Usage:
    python -m innerj.cli.build_dataset --family language --n 400
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from innerj import config, console
from innerj.cli import common
from innerj.model import model_slug
from innerj.tasks.base import group_by_instance, write_jsonl

DATA_ROOT = config.DATA_ROOT


def main() -> None:
    parser = common.parser(__doc__, needs=("model", "seed"))
    parser.add_argument(
        "--family", default="language", choices=["language", "tracking"]
    )
    parser.add_argument("--n", type=int, default=400, help="semantic instances")
    parser.add_argument("--n-choices", type=int, default=4)
    parser.add_argument(
        "--swaps",
        type=int,
        default=3,
        help="tracking family only: swap operations. More swaps means deeper "
        "tracking, traded against whether the model can do the task at all.",
    )
    parser.add_argument(
        "--unmatched-legend",
        action="store_true",
        help="show the operator table only in the flexible arm (sensitivity "
        "variant; never a primary result)",
    )
    parser.add_argument(
        "--matched-length",
        action="store_true",
        help="language family only: instructions that tokenise to the same "
        "length in every arm, sharing a 9-token tail. Needed for any measure "
        "read over a span of query positions; use --last-n 9 downstream.",
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--tag-extra", default="")
    args = parser.parse_args()

    if args.matched_length and args.family != "language":
        raise SystemExit(
            "--matched-length is implemented for the language family only; "
            "tracking length-matches too but its gather is a linear-attention "
            "layer with no attention matrix, so the measurement it enables is "
            "unavailable there."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.family == "language":
        from innerj.tasks.language import LanguageConfig, generate

        config = LanguageConfig(
            n_instances=args.n,
            n_choices=args.n_choices,
            matched_legend=not args.unmatched_legend,
            matched_length=args.matched_length,
            seed=args.seed,
        )
        records = generate(tokenizer, config)
    elif args.family == "tracking":
        from innerj.tasks.tracking import TrackingConfig, generate

        config = TrackingConfig(
            n_instances=args.n,
            n_people=args.n_choices,
            n_swaps=args.swaps,
            matched_legend=not args.unmatched_legend,
            seed=args.seed,
        )
        records = generate(tokenizer, config)
    else:  # pragma: no cover - argparse restricts this
        raise SystemExit(f"unknown family {args.family}")

    if not records:
        raise SystemExit(
            "generator produced no records. An empty dataset reports a clean "
            "null downstream, which looks like a finding."
        )

    suffix = "unmatched" if args.unmatched_legend else "matched"
    if args.matched_length:
        suffix = f"{suffix}_len"
    slug = model_slug(args.model)
    out = Path(args.out) if args.out else (
        DATA_ROOT / args.family
        / f"{slug}_{suffix}_n{args.n}_s{args.seed}{args.tag_extra}.jsonl"
    )
    n = write_jsonl(records, out)

    counts = Counter(str(r.condition) for r in records)
    instances = group_by_instance(records)
    # Count against the conditions this family actually emits, not a hard-coded 4.
    # With a fifth arm the old `len(g) == 4` counted instances *missing* one and
    # reported 38 of 399 as complete, which reads as a broken generator.
    arms = {r.condition for r in records}
    complete = sum(1 for g in instances.values() if arms <= set(g))
    console.detail(f"wrote {n} records to {out}")
    console.detail(
        f"  instances: {len(instances)} ({complete} with all {len(arms)} conditions)"
    )
    for condition, count in sorted(counts.items()):
        console.detail(f"  {condition:10} {count}")
    console.detail(f"  languages: {len({r.latent_value for r in records})}")


if __name__ == "__main__":
    main()
