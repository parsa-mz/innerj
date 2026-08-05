"""Fit a Jacobian lens for the replication model.

Usage:
    python -m innerj.cli.fit_lens --model Qwen/Qwen3.5-9B --n-prompts 120

Two things this guards, because both have silently invalidated a lens before:

**Never cycle the fit corpus.** ``J_l`` is an expectation over the prompt
distribution, so padding a short corpus out to the requested count estimates the
Jacobian of the short corpus at N-times the cost. The tell is two lenses fit at
different settings coming out bitwise identical. We draw distinct passages and
record how many were actually distinct.

**Save in fp32.** The reference ``save()`` defaults to fp16 with no
representability check; entries above 65504 become ``inf`` and every product
downstream becomes ``NaN``. We write fp32 and refuse a non-finite result.
"""

from __future__ import annotations

import torch
from jlens import fit

from innerj import config, console
from innerj.cli import common
from innerj.model import band, load_model

OUT_ROOT = config.DATA_ROOT / "lenses"


def wikitext_passages(n: int, *, min_chars: int = 400) -> list[str]:
    """``n`` distinct wikitext-103 passages, deduplicated.

    Uses the same corpus as the published lenses so the artifact is comparable
    to them.
    """
    from datasets import load_dataset

    dataset = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True
    )
    seen: set[str] = set()
    out: list[str] = []
    for row in dataset:
        text = row["text"].strip()
        if len(text) < min_chars or text.startswith("="):
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= n:
            break
    if len(out) < n:
        raise RuntimeError(f"only {len(out)} distinct passages found, wanted {n}")
    return out


def main() -> None:
    parser = common.parser(__doc__, needs=("device",))
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-prompts", type=int, default=120)
    parser.add_argument("--dim-batch", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=None,
        help="source layers; defaults to the workspace band, which is all we read",
    )
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    print(f"loading {args.model} ...", flush=True)
    model = load_model(args.model, device=args.device)
    layers = args.layers or band(model.n_layers)
    print(
        f"{model.n_layers} layers, fitting {len(layers)} source layers "
        f"L{layers[0]}-L{layers[-1]}",
        flush=True,
    )

    prompts = wikitext_passages(args.n_prompts)
    distinct = len(set(prompts))
    print(f"{len(prompts)} prompts, {distinct} distinct", flush=True)
    if distinct < len(prompts):
        raise SystemExit(
            "the corpus contains duplicates; a cycled corpus estimates the "
            "Jacobian of a smaller corpus at full cost"
        )

    slug = args.model.split("/")[-1]
    tag = args.tag or f"{slug}_wikitext_n{args.n_prompts}"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = OUT_ROOT / f"{tag}.partial"

    lens = fit(
        model,
        prompts,
        source_layers=layers,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        checkpoint_path=str(checkpoint),
        checkpoint_every=5,
    )

    for layer, J in lens.jacobians.items():
        if not torch.isfinite(J).all():
            raise SystemExit(f"J[{layer}] is not finite; refusing to save")

    path = OUT_ROOT / f"{tag}.pt"
    lens.save(str(path), dtype=torch.float32)
    # The resume checkpoint is the same size as the lens itself, so leaving it behind
    # doubles the artifact for no benefit once the real file is on disk.
    checkpoint.unlink(missing_ok=True)
    console.wrote(path)
    print(f"  source layers {lens.source_layers[0]}..{lens.source_layers[-1]}")
    print(f"  n_prompts {lens.n_prompts}, d_model {lens.d_model}")
    diag = {
        layer: float(torch.diagonal(J).mean()) for layer, J in lens.jacobians.items()
    }
    sample = sorted(diag)[:: max(len(diag) // 5, 1)]
    print("  mean diagonal (a sanity check only -- it is gauge-dependent):")
    for layer in sample:
        print(f"    L{layer}: {diag[layer]:.4f}")


if __name__ == "__main__":
    main()
