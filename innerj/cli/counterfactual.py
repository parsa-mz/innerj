"""Stage 2c: does the transported value change the answer?

Usage:
    python -m innerj.cli.counterfactual --records <jsonl> \
        --components attn.L39 attn.L39.H15 attn.L43 --pairs 60
"""

from __future__ import annotations

from innerj import console
from innerj.cli import common
from innerj.experiments.counterfactual import (
    buildable_pairs,
    counterfactual,
    pool_counterfactual,
)
from innerj.positions import build
from innerj.tasks.base import Condition


def main() -> None:
    parser = common.parser(
        __doc__,
        # `lens` is declared even though the default run does not load one: the
        # --lens-readout path needs it, and a missing argument surfaces only at the
        # point of use, which is after the model has been loaded.
        needs=(
            "records", "model", "lens", "lens_device", "device", "pairs", "seed",
            "tag",
        ),
    )
    parser.add_argument(
        "--components",
        nargs="+",
        required=True,
        help="component specs; join with '+' to patch several together, e.g. "
        "attn.L39+mlp.L39",
    )
    parser.add_argument(
        "--last-n",
        type=int,
        default=1,
        help="span in tokens",
    )
    parser.add_argument(
        "--where",
        nargs="+",
        default=["query"],
        choices=["query", "passage"],
        help="'query' patches the end of the prompt, 'passage' the end of the "
        "context where the latent variable is constructed. Both runs one after the "
        "other for a direct comparison.",
    )
    parser.add_argument(
        "--lens-readout",
        action="store_true",
        help="also record the donor concept's J-lens readout on the same trial, "
        "giving a within-trial readout-versus-behaviour comparison instead of one "
        "across two differently sized samples. Off by default: the headline "
        "behavioural result is deliberately measured without touching the lens.",
    )
    parser.set_defaults(tag="counterfactual")
    args = parser.parse_args()

    # "a+b+c" patches those components jointly under one label.
    groups = [
        (spec, [common.parse_component(part) for part in spec.split("+")])
        for spec in args.components
    ]
    grouped = common.instances(args, (Condition.FLEXIBLE,))
    flexible = {i: g[Condition.FLEXIBLE] for i, g in grouped.items()}
    pairs = buildable_pairs(flexible, seed=args.seed)
    console.detail(f"{len(pairs)} counterfactual pairs")

    model, lens = common.load(args, lens=args.lens_readout)
    read_layers = None
    if args.lens_readout:
        # Read above every patched layer, so the readout measures something the
        # patch could have caused. Matched across groups so the columns are
        # comparable between components.
        deepest = max(c.layer for _, group in groups for c in group)
        read_layers = [
            layer for layer in lens.source_layers if deepest < layer <= deepest + 5
        ]
        if not read_layers:
            raise SystemExit(
                f"no fitted lens layer sits within 5 layers above L{deepest}; the "
                f"readout would not be comparable to the published ones"
            )
        console.detail(f"reading the donor concept at {read_layers}")

    observations = []
    for mode in args.where:
        console.step(f"patching {mode}, span {args.last_n}")
        observations.extend(
            counterfactual(
                model, pairs, groups, positions=build(mode, args.last_n),
                lens=lens if args.lens_readout else None,
                read_layers=read_layers,
            )
        )
    results = pool_counterfactual(observations, seed=args.seed)

    console.table(
        "does the patch install the donor's answer?",
        ["group", "where", "donor - distractor", "donor", "dist",
         "acc clean->patched", "verdict"],
        [
            [r.component, r.positions, str(r.delta_donor_vs_other),
             f"{r.donor_symbol_rate:.3f}", f"{r.other_symbol_rate:.3f}",
             f"{r.clean_accuracy:.3f}->{r.patched_accuracy:.3f}", r.verdict()]
            for r in sorted(
                results, key=lambda r: (r.positions, -r.delta_donor_vs_other.point)
            )
        ],
    )

    # The graded view. A flip rate of zero beside a moving logit margin is a
    # different result from a flip rate of zero beside a flat one, and only the
    # second is "the patch is inert".
    console.table(
        "graded behaviour, and the readout measured on the same trials",
        ["group", "where", "delta donor logit", "delta donor-distractor margin",
         "delta readout -log10 rank", "trial r"],
        [
            [r.component, r.positions, str(r.delta_donor_logit),
             str(r.delta_donor_vs_other_margin),
             "--" if r.delta_readout_logrank is None
             else str(r.delta_readout_logrank),
             "--" if r.readout_behaviour_r is None
             else f"{r.readout_behaviour_r:+.3f} (n={r.readout_behaviour_n})"]
            for r in sorted(
                results, key=lambda r: (r.positions, -r.delta_donor_vs_other.point)
            )
        ],
    )

    common.save(
        "counterfactual", args,
        observations=observations, results=results, n_pairs=len(pairs),
    )


if __name__ == "__main__":
    main()
