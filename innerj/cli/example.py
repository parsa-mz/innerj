"""A worked example: what the J-lens literally reads at the query position.

`R_z` is a percentile rank, which is precise and opaque. This dumps, for one
instance in all four arms, what the readout actually contains: the top-k tokens
at each band layer and the gold concept's rank among them.

Usage:
    innerj example --records <jsonl> --instance lang_000000
"""

from __future__ import annotations

from innerj import console
from innerj.cli import common
from innerj.experiments.example import ORDER, read_instance
from innerj.tasks.base import Condition, complete_instances, read_jsonl


def main() -> None:
    parser = common.parser(
        __doc__, needs=("records", "model", "lens", "lens_device", "device", "tag")
    )
    parser.add_argument("--instance", default=None,
                        help="semantic instance id; default is the first complete one")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--stride", type=int, default=3,
                        help="report every Nth band layer")
    parser.set_defaults(tag="example")
    args = parser.parse_args()

    groups = complete_instances(
        list(read_jsonl(args.records)), required=tuple(Condition)
    )
    if not groups:
        raise SystemExit(f"{args.records} has no instance complete in all four arms")
    instance = args.instance or sorted(groups)[0]
    if instance not in groups:
        raise SystemExit(f"{instance} is not complete; have {len(groups)} instances")
    group = groups[instance]
    console.step(f"instance {instance}, {len(group)} arms")

    model, lens = common.load(args)
    # The lens band, subsampled: every layer would be 36 rows of top-k tokens.
    layers = common.readout_layers(model, lens)[:: args.stride]

    any_record = next(iter(group.values()))
    result = {
        "instance": instance,
        "latent_name": any_record.latent_name,
        "latent_value": any_record.latent_value,
        "latent_token_id": any_record.latent_token_id,
        "gold_answer": any_record.gold_answer,
        "candidate_answers": any_record.candidate_answers,
        "context": any_record.context,
        "layers": layers,
        "conditions": {},
    }
    rows = []
    for condition in ORDER:
        if condition not in group:
            continue
        # The tokenizer comes off the model rather than a second
        # from_pretrained: one load, and no chance of the two disagreeing.
        row = read_instance(
            model, lens, model.tokenizer, group[condition], layers=layers,
            top_k=args.top_k,
        )
        result["conditions"][str(condition)] = row
        rows.append([
            str(condition),
            row["answer"],
            str(row["correct"]),
            " ".join(f"{r['rank']}" for r in row["layers"]),
        ])
    console.table(
        f"{any_record.latent_value} rank by layer, {layers[0]}..{layers[-1]}",
        ["condition", "answer", "correct", "rank per layer"],
        rows,
    )

    common.write("example", common.stem(args, instance), result, args=args)


if __name__ == "__main__":
    main()
