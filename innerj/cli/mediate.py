"""Stage 6: J-space mediation of the behavioural effect.

Usage:
    python -m innerj.cli.mediate --records <jsonl> --component resid.L39 --pairs 80
"""

from __future__ import annotations

from innerj import console
from innerj.cli import common
from innerj.experiments.counterfactual import buildable_pairs
from innerj.experiments.mediate import (
    CONTROLS,
    DERIVATIONS,
    PROJECTIONS,
    mediate,
    pool,
    verdict,
    wrong_concept_spread,
)
from innerj.tasks.base import Condition


def main() -> None:
    parser = common.parser(
        __doc__,
        needs=(
            "records", "model", "lens", "lens_device", "device", "pairs", "last_n",
            "seed", "tag",
        ),
    )
    parser.add_argument("--components", nargs="+", default=["resid.L39"])
    parser.add_argument(
        "--derivations",
        nargs="+",
        default=["static"],
        choices=list(DERIVATIONS),
        help="how the concept direction is derived. 'static' is J^T W_U[z], what "
        "every published number used; 'gradient' differentiates through the final "
        "norm, which is the direction that actually raises the normalised logit; "
        "'margin' does the same for the contrastive concept score.",
    )
    parser.add_argument(
        "--controls",
        nargs="+",
        default=["gold", "random"],
        choices=list(CONTROLS),
        help="which directions to remove. 'gold' is the candidate mediator; the "
        "rest are nulls of increasing strength. 'wrong' (another concept) and "
        "'norm_matched' (equal activation norm removed) are the two that a "
        "reviewer will ask for.",
    )
    parser.add_argument(
        "--doses",
        type=float,
        nargs="+",
        default=[1.0],
        help="fractions of the direction to remove, for a dose-response curve, "
        "e.g. --doses 0 0.25 0.5 0.75 1.0",
    )
    parser.add_argument(
        "--projections",
        nargs="+",
        default=["absolute"],
        choices=list(PROJECTIONS),
        help="'absolute' removes the direction from the donor's residual, as "
        "published; 'delta' removes it only from the donor-minus-target difference, "
        "preserving everything the two share and excising only what the substitution "
        "introduces.",
    )
    parser.set_defaults(pairs=80)
    parser.set_defaults(tag="mediate")
    args = parser.parse_args()

    groups = common.instances(args, (Condition.FLEXIBLE,))
    flexible = {i: g[Condition.FLEXIBLE] for i, g in groups.items()}
    pairs = buildable_pairs(flexible, seed=args.seed)
    console.detail(f"{len(pairs)} counterfactual pairs")

    model, lens = common.load(args)
    positions, position_label = common.query_span(args)

    # The family's whole concept vocabulary, for the wrong_all sweep. Taken from the
    # records rather than a hard-coded language list, so it is correct for any family.
    concept_ids = sorted({r.latent_token_id for r in flexible.values()})
    # Each pair costs one reference pass plus one per cell, so say the cost before
    # spending it: with wrong_all the product runs to dozens of cells per pair.
    wide = len(concept_ids) - 1 if "wrong_all" in args.controls else 0
    cells = (
        len(args.derivations)
        * (len(args.controls) - (1 if wide else 0) + wide)
        * len(args.doses)
        * len(args.projections)
    )
    console.detail(
        f"{cells} cell(s) per pair over {len(args.derivations)} derivation(s), "
        f"{len(args.controls)} control(s), {len(args.doses)} dose(s), "
        f"{len(args.projections)} projection(s)"
        + (f"; wrong_all sweeps {wide} rival concepts" if wide else "")
    )

    observations = []
    for spec in args.components:
        component = common.parse_component(spec)
        console.step(str(component))
        observations.extend(
            mediate(
                model, lens, pairs, component,
                positions=positions, position_label=position_label,
                seed=args.seed,
                derivations=tuple(args.derivations),
                controls=tuple(args.controls),
                doses=tuple(args.doses),
                projections=tuple(args.projections),
                concept_ids=concept_ids,
            )
        )
    results = pool(observations, seed=args.seed)

    console.table(
        "does the transported value mediate the behaviour?",
        ["component", "mode", "donor", "dist", "acc", "donor - distractor",
         "norm removed"],
        [
            [r.component, r.mode, f"{r.donor_symbol_rate:.3f}",
             f"{r.other_symbol_rate:.3f}", f"{r.accuracy:.3f}",
             str(r.delta_vs_other),
             "--" if r.mean_removed_norm != r.mean_removed_norm
             else f"{r.mean_removed_norm:.2f}"]
            for r in results
        ],
    )
    # The wrong-concept distribution, which the verdict deliberately excludes: its
    # value is the spread over the vocabulary, above all the worst case.
    for spec in args.components:
        for projection in args.projections:
            spread = wrong_concept_spread(
                results, component=spec, projection=projection
            )
            if spread:
                console.detail(
                    f"{spec} [{projection}] wrong-concept null over "
                    f"{spread['n_rivals']} rivals: mean loss "
                    f"{spread['mean_loss']:+.4f}, worst {spread['max_loss']:+.4f}, "
                    f"best {spread['min_loss']:+.4f}, sd {spread['sd_loss']:.4f} "
                    f"(reference rate {spread['reference_rate']:.3f})"
                )

    verdicts = verdict(observations)
    for component, text in verdicts.items():
        console.detail(f"{component}: {text}")

    common.save(
        "mediate", args,
        observations=observations, results=results,
        n_pairs=len(pairs), verdict=verdicts,
    )


if __name__ == "__main__":
    main()
