"""Stage 2: screen components for the automatic-to-flexible write effect.

Coarse pass first (one attention block + one MLP per layer), then heads within the
layers that survive.

Usage:
    python -m innerj.cli.screen --records <jsonl> --pass coarse --pairs 40
    python -m innerj.cli.screen --records <jsonl> --pass heads --layers 40 42 44
"""

from __future__ import annotations

from innerj import console
from innerj.cli import common
from innerj.experiments.screen import (
    coarse_components,
    head_components,
    pool,
    readout_layers,
    screen,
    survivors,
)
from innerj.model import band
from innerj.tasks.base import Condition

CONTRAST = (Condition.FLEXIBLE, Condition.AUTOMATIC)


def main() -> None:
    parser = common.parser(
        __doc__,
        needs=("records", "model", "lens", "lens_device", "device", "pairs", "seed",
               "tag"),
    )
    parser.add_argument(
        "--pass", dest="which", default="coarse", choices=["coarse", "heads"]
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=None,
        help="layers to screen; defaults to L36-L48, where the Stage-1 effect "
        "plateaus",
    )
    parser.set_defaults(pairs=40)
    parser.add_argument("--n-readout", type=int, default=6)
    parser.add_argument(
        "--directions",
        nargs="*",
        default=["flex_to_auto", "auto_to_flex"],
        help="a real write component must show both",
    )
    args = parser.parse_args()
    # Unlike the other CLIs the tag names the pass rather than the experiment,
    # since a coarse and a heads run over the same records are different results.
    args.tag = args.tag or args.which

    grouped = common.instances(args, CONTRAST)
    pairs = [
        (g[Condition.FLEXIBLE], g[Condition.AUTOMATIC]) for g in grouped.values()
    ]

    model, lens = common.load(args)
    layers = args.layers or [
        layer for layer in band(model.n_layers) if 36 <= layer <= 48
    ]
    read = readout_layers(layers, model.n_layers, n_readout=args.n_readout)

    components = (
        coarse_components(layers) if args.which == "coarse"
        else head_components(model, layers)
    )
    console.detail(
        f"screening {len(components)} components at L{layers[0]}-L{layers[-1]}, "
        f"reading R_z at {read}"
    )

    all_observations = []
    all_results = []
    for direction in args.directions:
        console.step(direction)
        observations = screen(
            model, lens, pairs, components, read_layers=read, direction=direction
        )
        results = pool(observations, seed=args.seed)
        kept = survivors(results)
        kept_log = survivors(results, metric="delta_logrank")
        all_observations.extend(observations)
        all_results.extend(results)

        console.table(
            f"{direction}: {len(kept)}/{len(results)} survive FDR correction "
            f"under R_z, {len(kept_log)}/{len(results)} under log-rank",
            ["", "component", "delta R_z", "delta -log10 rank", "recovery", "gap"],
            [
                # "*" survives correction under R_z, "L" under log-rank. A
                # component marked one way only is significant under a metric that
                # saturates or under one that does not -- which is a claim about
                # the measure, and must not be quoted as a bare transport effect.
                [("*" if r in kept else "") + ("L" if r in kept_log else "") or " ",
                 r.component, str(r.delta_entry), str(r.delta_logrank),
                 # NaN by design: recovery is a ratio guarded against a vanishing
                 # denominator, and a mean of ratios there has already produced one
                 # retracted claim.
                 "nan" if r.recovery != r.recovery else f"{r.recovery:+.3f}",
                 f"{r.recovery_gap:+.4f}"]
                for r in results[:12]
            ],
        )
        disagree = set(id(r) for r in kept) ^ set(id(r) for r in kept_log)
        if disagree:
            console.detail(
                f"   {len(disagree)} component(s) survive under one metric and not "
                f"the other; no window boundary may rest on R_z alone"
            )

    common.save(
        "screen", args,
        observations=all_observations, results=all_results,
        layers=layers, read_layers=read, n_pairs=len(pairs),
    )


if __name__ == "__main__":
    main()
