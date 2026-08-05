"""Stage 1b: is the latent variable decodable regardless of task demand?

Usage:
    python -m innerj.cli.probe --records <jsonl> --limit 200
"""

from __future__ import annotations

import json

import numpy as np

from innerj import console
from innerj.cli import common
from innerj.experiments.probe import cache_residuals, probe_grid, summarise
from innerj.model import band
from innerj.tasks.base import Condition

#: All four arms: the question is whether z is decodable in every one of them,
#: which is what makes the entry effect a demand effect and not an availability
#: difference. Restricting to the contrast pair would not answer it.
ARMS = tuple(Condition)


def main() -> None:
    parser = common.parser(__doc__, needs=("records", "model", "device", "seed", "tag"))
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=None,
        help="defaults to every third band layer, which is enough for a profile",
    )
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--seeds", type=int, default=5,
                        help="instance splits to average over, from --seed upward")
    parser.add_argument("--recache", action="store_true",
                        help="recompute residuals even if a cache exists")
    parser.set_defaults(tag="probe")
    args = parser.parse_args()

    grouped = common.instances(args, ARMS, limit=args.limit)
    records = [r for group in grouped.values() for r in group.values()]

    stem = common.stem(args)
    cache = common.DATA_ROOT / "probe" / f"{stem}_residuals.npy"
    # Which layers the cached rows correspond to. Without this the axis is a bare
    # index and every layer in the output is mislabelled -- the cache holds
    # L24,L27,... but position 0 is not layer 0.
    cache_layers = cache.with_name(f"{stem}_residuals_layers.json")

    # Every re-analysis -- centring variants, extra seeds, the permuted-label
    # floor -- reuses the same residuals unchanged, and recomputing them costs a
    # 52 GB model load plus ~5 min on a card that is usually busy. So the cache is
    # the default input, not just an output.
    if cache.is_file() and not args.recache:
        residuals = np.load(cache)
        if residuals.shape[0] != len(records):
            raise SystemExit(
                f"{cache.name} holds {residuals.shape[0]} rows but this "
                f"selection has {len(records)} records. Pass --recache, or "
                f"match the --limit the cache was built with."
            )
        if args.layers:
            layers = args.layers
        elif cache_layers.is_file():
            layers = json.loads(cache_layers.read_text())
        else:
            raise SystemExit(
                f"{cache.name} predates the layer sidecar, so its layer "
                f"identities are unknown. Pass --layers to name them, or "
                f"--recache to rebuild."
            )
        if len(layers) != residuals.shape[1]:
            raise SystemExit(
                f"{cache.name} holds {residuals.shape[1]} layers, not "
                f"{len(layers)}. Pass --recache."
            )
        console.step(f"reusing cached residuals: {residuals.shape} from {cache.name}")
    else:
        model, _ = common.load(args, lens=False)
        layers = args.layers or band(model.n_layers)[::3]
        console.step(f"caching residuals at layers {layers}")
        residuals = cache_residuals(model, records, layers)

        # The probe needs no forward passes; free the weights so a big grid fits.
        del model
        import torch

        torch.cuda.empty_cache()

        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, residuals)
        cache_layers.write_text(json.dumps(layers))
        console.wrote(cache)

    summaries = {}
    for centered in (False, True):
        label = "centered" if centered else "raw"
        console.step(f"probe grid, {label} features")
        # Several seeds, because one split of 80 test instances moves by a few
        # points between seeds -- wider than some differences the paper draws.
        # Plus a permuted-label pass, which is the floor for the best-of-layers
        # statistic: a maximum over twelve draws does not sit at 1/n_classes.
        results = []
        for seed in range(args.seed, args.seed + args.seeds):
            for shuffled in (False, True):
                results.extend(
                    probe_grid(
                        residuals,
                        records,
                        layers,
                        train_frac=args.train_frac,
                        seed=seed,
                        device=args.device,
                        center_per_condition=centered,
                        shuffle_labels=shuffled,
                    )
                )
        summary = summarise(results)
        summaries[label] = summary

        floor = summary.get("shuffled_floor", {}).get("joint", {})
        floor_value = floor.get("best_of_layers", {}).get("mean")
        console.detail(
            f"chance = {summary['chance']:.4f} over {summary['n_classes']} "
            f"classes"
            + (f"; permuted-label best-of-layers floor = {floor_value:.4f}"
               if floor_value else "")
        )
        console.table(
            f"{label}: joint probe (one matrix, all arms), "
            f"best of {len(layers)} layers",
            ["arm", "accuracy", "sd over seeds", "x floor"],
            [
                [arm, f"{s['mean']:.4f}", f"{s['sd']:.4f}",
                 f"{s['mean'] / floor_value:.1f}" if floor_value else "n/a"]
                for arm, s in summary["joint_best"].items()
            ],
        )
        console.table(
            f"{label}: per-arm grid, best-layer decoding of z",
            ["arm or transfer", "accuracy", "over chance"],
            [
                [condition, f"{accuracy:.4f}",
                 f"+{accuracy - summary['chance']:.4f}"]
                for condition, accuracy in summary["diagonal_best"].items()
            ]
            + [
                [pair, f"{accuracy:.4f}", f"+{accuracy - summary['chance']:.4f}"]
                for pair, accuracy in sorted(
                    summary["transfer_best"].items(), key=lambda kv: -kv[1]
                )[:6]
            ],
        )

    common.write("probe", stem, summaries, args=args)


if __name__ == "__main__":
    main()
