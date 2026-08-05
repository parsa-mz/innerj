"""Stage 1: is workspace entry different from latent availability?

Reads `R_z` at the query position for every condition present in the records, over the
same contexts, and reports each paired contrast with its accuracy difference beside it.
The accuracy column is not decoration: an entry effect between arms of unequal
difficulty is a difficulty effect, which is what `Dissociation.verdict` refuses. When
the records carry the `supplied` arm the 2x2 contrasts are reported too.

Usage:
    python -m innerj.cli.stage1 \
        --records data/language/Qwen3.6-27B_matched_n400_s0.jsonl \
        --limit 200
"""

from __future__ import annotations

from innerj import console
from innerj.cli import common
from innerj.experiments.entry import (
    dissociation,
    interaction,
    layer_profile,
    run,
    save_trials,
)
from innerj.tasks.base import Condition

#: Every arm. Every contrast below is read off the same instances, so an
#: instance missing any one arm is dropped from all of them rather than
#: contributing to some contrasts and not others.
ARMS = tuple(Condition)

#: Both baselines are reported. AUTOMATIC is the source design's contrast but is
#: not format-matched, and a question-answering cue alone lifts entry. CONTROL
#: carries the identical format and needs the latent variable for nothing, so
#: flexible-vs-control is the one that isolates demand for z.
CONTRASTS = [
    (Condition.FLEXIBLE, Condition.AUTOMATIC),
    (Condition.REPORT, Condition.AUTOMATIC),
    (Condition.FLEXIBLE, Condition.CONTROL),
    (Condition.REPORT, Condition.CONTROL),
    (Condition.CONTROL, Condition.AUTOMATIC),
    # The 2x2. `supplied` carries the operator without needing the inference, so
    # crossing it with `report` separates latent-variable demand from the
    # compositional work of applying an operator:
    #   (flexible - supplied) - (report - control).
    (Condition.FLEXIBLE, Condition.SUPPLIED),
    (Condition.SUPPLIED, Condition.CONTROL),
    (Condition.REPORT, Condition.SUPPLIED),
]


def main() -> None:
    parser = common.parser(
        __doc__,
        needs=("records", "model", "lens", "lens_device", "device",
               "max_seq_len", "tag"),
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="cap on semantic instances")
    parser.add_argument("--layers", type=int, nargs="*", default=None,
                        help="readout layers; defaults to the workspace band")
    parser.set_defaults(tag="stage1")
    args = parser.parse_args()

    grouped = common.instances(args, ARMS, limit=args.limit)
    records = [r for group in grouped.values() for r in group.values()]

    model, lens = common.load(args)
    layers = common.readout_layers(model, lens, args.layers)

    trials = run(model, lens, records, layers=layers, max_seq_len=args.max_seq_len)

    stem = common.stem(args)
    save_trials(trials, common.DATA_ROOT / "stage1" / f"{stem}_trials.jsonl")

    contrasts, rows, verdicts = {}, [], []
    for high, low in CONTRASTS:
        result = dissociation(trials, high=high, low=low)
        contrasts[f"{high}_vs_{low}"] = result.to_dict()
        verdict = result.verdict()
        verdicts.append((high, low, verdict))
        rows.append([
            f"{high} - {low}",
            str(result.delta_entry),
            str(result.delta_accuracy),
            f"{result.accuracy_high:.3f} / {result.accuracy_low:.3f}",
            verdict.split(":")[0],
        ])
    console.table(
        "entry by condition pair",
        ["contrast", "delta R_z (band mean)", "delta accuracy", "acc hi/lo", "verdict"],
        rows,
    )
    for high, low, verdict in verdicts:
        if not verdict.startswith("ENTRY EFFECT"):
            console.warn(f"{high} vs {low}: {verdict}")

    # The 2x2, when the dataset carries the `supplied` arm. It answers a question the
    # pairwise contrasts above cannot: how much of flexible-minus-control is the
    # operator rather than the latent variable.
    cross = None
    try:
        cross = interaction(trials)
    except ValueError as exc:
        console.detail(f"no 2x2 interaction: {exc}")
    if cross is not None:
        console.table(
            "the 2x2: latent-variable demand against compositional work",
            ["effect", "delta R_z (band mean)"],
            [
                ["operator only (supplied - control)", str(cross.operator_only)],
                ["inference only (report - control)", str(cross.inference_only)],
                ["both (flexible - control)", str(cross.both)],
                ["latent demand (flexible - supplied)", str(cross.latent_demand)],
                ["interaction", str(cross.interaction)],
            ],
        )
        console.detail(cross.verdict())

    common.write(
        "stage1",
        f"{stem}_summary",
        {
            "model": args.model,
            "lens": args.lens,
            "records": args.records,
            "n_trials": len(trials),
            "layers": layers,
            "contrasts": contrasts,
            "interaction": cross.to_dict() if cross is not None else None,
            "layer_profile": {
                str(c): profile
                for c in Condition
                if (profile := layer_profile(trials, c))
            },
        },
        args=args,
    )


if __name__ == "__main__":
    main()
