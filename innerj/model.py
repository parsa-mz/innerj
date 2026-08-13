"""Model and lens loading, with the guards the reference implementation lacks.

The lens algebra comes from the authors' ``jlens``, so our numbers are parity-correct by
construction. This module adds a finiteness check on load -- the reference ``save()``
defaults to fp16 and published Gemma-4 artifacts contain fp16 subnormals -- the
workspace
band mapped onto actual depth, and the position floor, since ``fit()`` skips the first
16
positions but ``apply()`` reads them anyway.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from jlens import JacobianLens, from_hf
from jlens.hf import HFLensModel

#: ``fit()`` never sees positions below this index, so readouts there are out of
#: distribution. Short prompts draw most of their positions from this region.
MIN_FIT_POSITION = 16

#: The source paper locates the workspace in L38-L92 of a 0-100 depth reindex.
BAND_START_FRAC = 0.38
BAND_END_FRAC = 0.92


def band(n_layers: int) -> list[int]:
    """Workspace layer band for ``n_layers``, inclusive.

    Reproduces the depth-verified band for every model in the prior survey: 24 layers ->
    9-22,
    32 -> 12-29, 48 -> 18-44, 60 -> 23-55, 64 -> 24-59. Clamped to the last block, since
    at
    shallow depths the fraction rounds past it.
    """
    if n_layers < 2:
        raise ValueError(f"n_layers={n_layers} is too shallow to have a band")
    lo = round(BAND_START_FRAC * n_layers)
    hi = min(round(BAND_END_FRAC * n_layers), n_layers - 1)
    return list(range(lo, hi + 1))


def model_slug(name: str) -> str:
    """``"Qwen/Qwen3.6-27B"`` -> ``"Qwen3.6-27B"``; artifact stems build on it."""
    return name.split("/")[-1]


@dataclass(frozen=True)
class Target:
    """A (checkpoint, lens) pair. Lens quality varies by family, so a number without its
        exact lens is not interpretable.
    """

    model_name: str
    lens_path: str

    @property
    def slug(self) -> str:
        return model_slug(self.model_name)


#: Primary target. Qwen3.6-27B has no base variant, so it is already
#: instruction-tuned -- which this project needs, since workspace entry under
#: flexible task demands is several times weaker on base checkpoints. Its
#: published lens is the densest artifact available: all 63 source layers at
#: n=1000 fit prompts.
QWEN27B = Target(
    model_name="Qwen/Qwen3.6-27B",
    lens_path=(
        "neuronpedia/jacobian-lens::qwen3.6-27b/jlens/Salesforce-wikitext/"
        "Qwen3.6-27B_jacobian_lens_n1000.pt"
    ),
)


def load_lens(
    path: str, *, device: str | None = None
) -> JacobianLens:
    """Load a fitted lens, refusing anything non-finite.

    ``path`` is a local file or ``"repo_id::filename"``; Jacobians are promoted to fp32.
    ``device`` moves them once at load, because the reference ``transport()`` does
    ``J.to(...)``
    on **every call** -- 955 ms against 376 ms per Stage-1 record. Costs ~6.6 GB.
    """
    if "::" in path:
        repo_id, filename = path.split("::", 1)
        lens = JacobianLens.from_pretrained(repo_id, filename=filename)
    else:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        lens = JacobianLens.load(path)

    for layer, J in lens.jacobians.items():
        if not torch.isfinite(J).all():
            n_bad = int((~torch.isfinite(J)).sum())
            raise ValueError(
                f"{path}: J[{layer}] has {n_bad} non-finite entries. A lens "
                f"saved in fp16 with entries above 65504 overflows to inf, and "
                f"every product downstream becomes NaN."
            )
    if not lens.source_layers:
        raise ValueError(
            f"{path} has no fitted source layers; a readout from it is meaningless"
        )
    if device is not None:
        lens.jacobians = {
            layer: J.to(device) for layer, J in lens.jacobians.items()
        }
    return lens


def load_model(
    model_name: str,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    revision: str | None = None,
) -> HFLensModel:
    """Load a checkpoint and wrap it for lens readout.

    bf16 weights, fp32 readout arithmetic, verified at cosine 0.999967-1.000000 against
    an fp32
    forward. ``device="auto"`` shards across every visible GPU, needed because the gauge
    check
    must run in fp32 (trap 14) and a 31B checkpoint is then ~124 GB.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    placement = (
        {"device_map": "auto"} if device == "auto" else {}
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, revision=revision, **placement
    )
    if device != "auto":
        hf_model = hf_model.to(device)
    return from_hf(hf_model, tokenizer)


def check_positions(positions: list[int], seq_len: int) -> None:
    """Raise if any position is outside the lens's fitted range; negatives resolve
        against ``seq_len`` first, so ``-1`` on a short prompt is caught.
    """
    resolved = [p if p >= 0 else seq_len + p for p in positions]
    if any(not 0 <= p < seq_len for p in resolved):
        raise ValueError(f"positions {positions} out of range for seq_len={seq_len}")
    unfitted = sorted(p for p in resolved if p < MIN_FIT_POSITION)
    if unfitted:
        raise ValueError(
            f"positions {unfitted} are below the fitted floor "
            f"({MIN_FIT_POSITION}); fit() never saw them and readouts there are "
            f"out of distribution. Lengthen the prompt instead."
        )
