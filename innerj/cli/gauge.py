"""Gauge check: which diagnostics survive a reparameterisation of the stream.

Run in fp32. In bf16 the construction drifts ~3% over 64 layers from accumulation
alone, which blurs an exact identity into a small residual and proves nothing.

Usage:
    innerj gauge --records <jsonl> --model Qwen/Qwen3.5-9B --dtype float32
"""

from __future__ import annotations

import torch

from innerj import console
from innerj.cli import common
from innerj.experiments.gauge import (
    apply_gauge,
    mean_diagonal,
    read,
    step_gauge,
)
from innerj.model import band, load_lens, load_model, model_slug
from innerj.tasks.base import Condition, read_jsonl


def main() -> None:
    # On common.parser like every other CLI: it is the only place `args` reaches an
    # artifact by a guarded path rather than by hand.
    parser = common.parser(
        __doc__, needs=("records", "model", "lens", "device", "seed", "tag")
    )
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--prompts", type=int, default=8)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--step-layer", type=int, default=20)
    parser.add_argument("--probes", type=int, default=16)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.set_defaults(tag="gauge")
    args = parser.parse_args()

    records = [
        r for r in read_jsonl(args.records) if r.condition is Condition.FLEXIBLE
    ][: args.prompts]
    model = load_model(args.model, device=args.device,
                       dtype=getattr(torch, args.dtype))
    lens = load_lens(args.lens)
    fitted = set(lens.source_layers)
    layers = [layer for layer in band(model.n_layers) if layer in fitted]
    layers = layers[:: max(len(layers) // 12, 1)]
    # One probe layer either side of the step, so the same run carries a moving
    # diagnostic and a stationary one.
    probe_layers = [
        max(layer for layer in layers if layer < args.step_layer),
        min(layer for layer in layers if layer >= args.step_layer),
    ]
    console.detail(
        f"{len(records)} prompts, R_z at {layers}, diagonal at {probe_layers}"
    )

    before = [read(model, lens, r.prompt, r.latent_token_id, layers) for r in records]
    diagonal_before = {
        layer: mean_diagonal(model, records[0].prompt, layer, probes=args.probes,
                             epsilon=args.epsilon, seed=args.seed)
        for layer in probe_layers
    }
    console.detail(f"mean diagonal, original: {diagonal_before}")

    a = step_gauge(model.n_layers, args.step_layer, args.scale)
    handles = apply_gauge(model, a)
    try:
        after = [read(model, lens, r.prompt, r.latent_token_id, layers)
                 for r in records]
        diagonal_after = {
            layer: mean_diagonal(model, records[0].prompt, layer, probes=args.probes,
                                 epsilon=args.epsilon, seed=args.seed)
            for layer in probe_layers
        }
    finally:
        for handle in handles:
            handle.remove()
    console.detail(f"mean diagonal, gauged:   {diagonal_after}")

    logit_change = max(
        float((b["logits"] - x["logits"]).abs().max())
        for b, x in zip(before, after, strict=True)
    )
    logit_scale = max(float(b["logits"].abs().max()) for b in before)
    r_z_change = max(
        abs(b["r_z"][layer] - x["r_z"][layer])
        for b, x in zip(before, after, strict=True)
        for layer in layers
    )
    ratios = {
        layer: diagonal_after[layer] / diagonal_before[layer] for layer in probe_layers
    }
    predicted = {
        layer: a[model.n_layers] / a[layer + 1] for layer in probe_layers
    }

    summary = {
        "n_prompts": len(records),
        "layers": layers,
        "probe_layers": probe_layers,
        "max_abs_logit_change": logit_change,
        "max_abs_logit": logit_scale,
        "max_abs_r_z_change": r_z_change,
        "mean_diagonal_original": diagonal_before,
        "mean_diagonal_gauged": diagonal_after,
        "mean_diagonal_ratio": ratios,
        "predicted_ratio": predicted,
        "lens_mean_diagonal": {
            str(layer): float(lens.jacobians[layer].float().diagonal().mean())
            for layer in layers
        },
    }

    console.detail(f"\nmax |change in model logit| : {logit_change:.3e}"
          f"   (logits reach {logit_scale:.1f})")
    console.detail(f"max |change in R_z|         : {r_z_change:.3e}"
          f"   over {len(layers)} band layers")
    for layer in probe_layers:
        console.detail(f"mean diagonal ratio at L{layer:<3}: {ratios[layer]:.4f}"
              f"   (predicted {predicted[layer]:.4f})")

    # The historical stem, kept verbatim: four artifacts on disk carry it.
    stem = f"{args.tag}_s{args.scale:g}_k{args.step_layer}_{model_slug(args.model)}"
    common.write("gauge", stem, summary, args=args)


if __name__ == "__main__":
    main()
