"""Is the gather's attention route itself demand-dependent?

Records the attention pattern at one full-attention layer and reports, per head, how
much of the query positions' attention mass lands on the passage --- flexible against
control, paired within instance. No patching and no lens: this measures the route, not
its payload.

Usage:
    innerj attention --records <jsonl> --layer 39
"""

from __future__ import annotations

from innerj import console
from innerj.cli import common
from innerj.experiments.attention import attention_mass, pool, verdict
from innerj.patch import layer_type
from innerj.positions import build, describe
from innerj.tasks.base import Condition

ARMS = (Condition.FLEXIBLE, Condition.CONTROL, Condition.REPORT, Condition.AUTOMATIC)


def main() -> None:
    parser = common.parser(
        __doc__,
        needs=("records", "model", "device", "last_n", "seed", "max_seq_len", "tag"),
    )
    parser.add_argument("--layer", type=int, default=39,
                        help="full-attention layer whose route is in question")
    parser.add_argument(
        "--control-layers", type=int, nargs="*", default=[15, 27, 51],
        help="full-attention layers that do NOT gather, measured identically. The "
        "arms' instructions differ in length, so every layer's passage mass shifts "
        "between arms; only the excess over these controls is about the gather.",
    )
    parser.add_argument("--limit", type=int, default=200,
                        help="cap on semantic instances")
    parser.set_defaults(tag="attention")
    args = parser.parse_args()

    grouped = common.instances(args, ARMS, limit=args.limit)
    records = [r for group in grouped.values() for r in group.values()]

    model, _ = common.load(args, lens=False)
    layers = [args.layer, *args.control_layers]
    for layer in layers:
        kind = layer_type(model, layer)
        if kind != "full_attention":
            raise SystemExit(
                f"L{layer} is {kind}. Only full-attention layers expose a pattern; "
                f"on this checkpoint that is every fourth layer."
            )
    label = describe(build("query", args.last_n))
    console.detail(
        f"gather L{args.layer} against controls "
        f"{', '.join(f'L{x}' for x in args.control_layers)}; "
        f"mass from {label} onto the passage"
    )

    observations = []
    for layer in layers:
        console.step(f"L{layer}")
        observations.extend(
            attention_mass(
                model, records, layer=layer, last_n=args.last_n,
                position_label=label, max_seq_len=args.max_seq_len,
            )
        )
    results = pool(observations, seed=args.seed)

    console.table(
        "passage attention mass, flexible - control, top heads per layer",
        ["layer", "head", "flex", "ctrl", "delta (span)", "delta (last token)"],
        [
            [f"L{r.layer}", f"H{r.head}", f"{r.mean_flexible:.4f}",
             f"{r.mean_control:.4f}", str(r.delta_passage_mass),
             str(r.delta_passage_mass_last)]
            for layer in layers
            for r in sorted(
                (x for x in results if x.layer == layer),
                key=lambda r: -abs(r.delta_passage_mass.point),
            )[:3]
        ],
    )
    text = verdict(results, gather_layer=args.layer,
                   control_layers=args.control_layers)
    (console.warn if text.startswith(("ROUTE FLAT", "REFUSED", "NOT SPECIFIC"))
     else console.step)(text)

    common.save(
        "attention", args,
        observations=observations, results=results,
        gather_layer=args.layer, control_layers=args.control_layers,
        positions=label, n_instances=len(grouped), verdict=text,
    )


if __name__ == "__main__":
    main()
