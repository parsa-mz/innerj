"""Stage 2b: mismatched-donor specificity control.

Usage:
    python -m innerj.cli.specificity --records <jsonl> \
        --components attn.L39 mlp.L46 attn.L43 --pairs 40
"""

from __future__ import annotations

from innerj import console
from innerj.cli import common
from innerj.experiments.screen import readout_layers
from innerj.experiments.specificity import (
    mismatched_pairs,
    pool_specificity,
    specificity,
)
from innerj.tasks.base import Condition

CONTRAST = (Condition.FLEXIBLE, Condition.AUTOMATIC)


def main() -> None:
    parser = common.parser(
        __doc__,
        needs=(
            "records", "model", "lens", "lens_device", "device", "pairs", "seed",
            "tag",
        ),
    )
    parser.add_argument("--components", nargs="+", required=True)
    parser.set_defaults(pairs=40)
    parser.add_argument("--n-readout", type=int, default=6)
    parser.add_argument(
        "--target-condition",
        default="automatic",
        choices=["automatic", "flexible"],
        help="which arm receives the patch. Use 'flexible' when the readout must "
        "explain a counterfactual measured on the flexible arm.",
    )
    parser.set_defaults(tag="specificity")
    args = parser.parse_args()

    components = [common.parse_component(c) for c in args.components]
    grouped = common.instances(args, CONTRAST)
    flexible = {i: g[Condition.FLEXIBLE] for i, g in grouped.items()}
    targets = (
        flexible
        if args.target_condition == "flexible"
        else {i: g[Condition.AUTOMATIC] for i, g in grouped.items()}
    )
    pairs = mismatched_pairs(flexible, targets, seed=args.seed)
    console.detail(f"{len(pairs)} mismatched-donor pairs")

    model, lens = common.load(args)
    read = readout_layers(
        [c.layer for c in components], model.n_layers, n_readout=args.n_readout
    )
    console.detail(f"components {[str(c) for c in components]}, reading R_z at {read}")

    observations = specificity(model, lens, pairs, components, read_layers=read)
    results = pool_specificity(observations, seed=args.seed)

    console.table(
        "does the patch carry the donor's value, or just disrupt the target's?",
        ["component", "delta donor R_z", "delta target R_z", "verdict"],
        [
            [r.component, str(r.delta_donor), str(r.delta_target), r.verdict()]
            for r in results
        ],
    )

    common.save(
        "specificity", args,
        observations=observations, results=results,
        read_layers=read, n_pairs=len(pairs),
    )


if __name__ == "__main__":
    main()
