"""Stage 4: necessity by ablation.

Usage:
    python -m innerj.cli.ablate --records <jsonl> \
        --components attn.L39 attn.L39.H15 mlp.L39 resid.L39 --mode mean
"""

from __future__ import annotations

from innerj import console
from innerj.cli import common
from innerj.experiments.ablate import ablate, dissociation, pool
from innerj.tasks.base import Condition

ARMS = (Condition.FLEXIBLE, Condition.REPORT, Condition.CONTROL)


def main() -> None:
    parser = common.parser(
        __doc__,
        needs=("records", "model", "device", "last_n", "seed", "tag"),
    )
    parser.add_argument("--components", nargs="+", required=True)
    parser.add_argument(
        "--modes", nargs="+", default=["mean"], choices=["mean", "zero"]
    )
    parser.add_argument("--instances", type=int, default=60)
    parser.set_defaults(tag="ablate")
    args = parser.parse_args()

    components = [common.parse_component(c) for c in args.components]
    grouped = common.instances(args, ARMS, limit=args.instances)
    # ARMS only: the groups carry every condition the instance has, and the
    # automatic arm has no demand for z, so including it would add a fourth row
    # to a three-arm design.
    records = [group[arm] for group in grouped.values() for arm in ARMS]

    model, _ = common.load(args, lens=False)
    positions, position_label = common.query_span(args)
    console.detail(f"components {[str(c) for c in components]}")

    observations = []
    for mode in args.modes:
        console.step(f"{mode} ablation")
        observations.extend(
            ablate(
                model, records, components, mode=mode, positions=positions,
                position_label=position_label,
            )
        )
    results = pool(observations, seed=args.seed)

    console.table(
        "accuracy cost of removing each component",
        ["component", "mode", "arm", "delta accuracy", "acc clean->ablated"],
        [
            [r.component, r.mode, r.condition, str(r.delta_accuracy),
             f"{r.accuracy_clean:.3f}->{r.accuracy_ablated:.3f}"]
            for r in results
        ],
    )

    # Selectivity is the whole point: a component that costs accuracy everywhere
    # is not carrying z, it is load-bearing for the forward pass.
    verdicts = dissociation(results)
    console.table(
        "selectivity: cost where z is needed, minus cost in control",
        ["component", "selectivity", "verdict"],
        [
            [component,
             "n/a" if entry.get("selectivity") is None
             else f"{entry['selectivity']:+.4f}",
             entry.get("verdict", "incomplete")]
            for component, entry in sorted(
                verdicts.items(), key=lambda kv: kv[1].get("selectivity", 0.0)
            )
        ],
    )

    common.save(
        "ablate", args,
        observations=observations, results=results, selectivity=verdicts,
    )


if __name__ == "__main__":
    main()
