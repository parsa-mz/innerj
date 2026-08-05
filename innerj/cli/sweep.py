"""2D sweep: patch layer x readout layer.

Usage:
    python -m innerj.cli.sweep --records <jsonl> --layers 10 40 --step 2 --pairs 40
"""

from __future__ import annotations

from innerj import console
from innerj.cli import common
from innerj.experiments.specificity import mismatched_pairs
from innerj.experiments.sweep import at_distance, decay_profile, sweep
from innerj.patch import Component
from innerj.tasks.base import Condition

CONTRAST = (Condition.FLEXIBLE, Condition.AUTOMATIC)


def main() -> None:
    parser = common.parser(
        __doc__,
        needs=(
            "records", "model", "lens", "lens_device", "device", "pairs", "last_n",
            "seed", "tag",
        ),
    )
    parser.add_argument(
        "--layers", type=int, nargs=2, default=[10, 48], metavar=("LO", "HI")
    )
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument(
        "--kinds", nargs="+", default=["resid", "attn", "mlp"],
        choices=["resid", "attn", "mlp"],
    )
    parser.add_argument(
        "--heads",
        type=int,
        nargs="*",
        default=None,
        help="layers to decompose head-by-head instead of as whole branches; head "
        "count is read per layer, since it differs between full- and "
        "linear-attention layers on a hybrid checkpoint",
    )
    parser.set_defaults(pairs=40)
    parser.set_defaults(last_n=1)
    parser.add_argument(
        "--distance", type=int, default=4, help="matched-distance slice to print"
    )
    parser.set_defaults(tag="sweep2d")
    args = parser.parse_args()

    grouped = common.instances(args, CONTRAST)
    flexible = {i: g[Condition.FLEXIBLE] for i, g in grouped.items()}
    pairs = mismatched_pairs(flexible, flexible, seed=args.seed)
    console.detail(f"{len(pairs)} mismatched pairs")

    lo, hi = args.layers
    layers = list(range(lo, hi + 1, args.step))
    model, lens = common.load(args)

    components = [Component(kind, layer) for layer in layers for kind in args.kinds]
    if args.heads:
        from innerj.patch import n_attention_heads

        for layer in args.heads:
            n_heads = n_attention_heads(model, layer)
            components += [Component("attn", layer, head=h) for h in range(n_heads)]
            console.detail(f"L{layer}: {n_heads} heads")
    console.detail(
        f"{len(components)} components over L{lo}-L{hi} step {args.step}"
    )
    positions, position_label = common.query_span(args)
    cells = sweep(
        model, lens, pairs, components, positions=positions, max_read_layer=hi + 12
    )

    slice_ = sorted(
        at_distance(cells, args.distance), key=lambda c: -c.delta_donor.point
    )

    def flag(cell) -> str:
        """``*`` where R_z is significantly positive, ``L`` where log-rank is.

        Printed together because the whole question is whether they agree. A cell
        marked ``*`` alone is significant only under the saturating measure.
        """
        marks = ""
        if cell.delta_donor.excludes_zero and cell.delta_donor.point > 0:
            marks += "*"
        log = cell.delta_donor_logrank
        if log.excludes_zero and log.point > 0:
            marks += "L"
        return marks or " "

    console.table(
        f"matched-distance slice, ~{args.distance} layers above the patch",
        ["", "component", "read", "dist", "delta donor R_z", "delta -log10 rank",
         "delta M_z"],
        [
            [flag(c), c.component, str(c.read_layer), str(c.distance),
             str(c.delta_donor), str(c.delta_donor_logrank), str(c.delta_donor_mz)]
            for c in slice_[:18]
        ],
    )

    # The number that decides whether the transport window is a property of the
    # model or of a metric that saturates above 0.999 in most of the cells it is
    # read from. A large disagreement count means no window claim can rest on R_z
    # alone.
    def positive(estimate) -> bool:
        return estimate.excludes_zero and estimate.point > 0

    full = at_distance(cells, args.distance)
    both = sum(1 for c in full if positive(c.delta_donor) and positive(
        c.delta_donor_logrank))
    r_only = sum(1 for c in full if positive(c.delta_donor) and not positive(
        c.delta_donor_logrank))
    log_only = sum(1 for c in full if positive(c.delta_donor_logrank) and not positive(
        c.delta_donor))
    console.detail(
        f"metric agreement over {len(full)} cells at distance ~{args.distance}: "
        f"{both} significant under both, {r_only} under R_z only, "
        f"{log_only} under log-rank only"
    )
    if r_only or log_only:
        console.detail(
            "   the two measures disagree; no window boundary may be quoted from "
            "R_z alone"
        )

    console.table(
        "decay with distance, best few components",
        ["component", "delta donor R_z, nearest 10 readouts above the patch"],
        [
            [c.component,
             "  ".join(f"+{d}:{v:+.3f}"
                       for d, v in decay_profile(cells, c.component)[:10])]
            for c in slice_[:4]
        ],
    )

    common.write(
        "sweep",
        common.stem(args, dataset=False),
        {"positions": position_label, "n_pairs": len(pairs),
         "cells": [c.to_dict() for c in cells]},
        args=args,
    )


if __name__ == "__main__":
    main()
