"""Model and lens loading, with the guards the reference implementation lacks.

We depend on the authors' own ``jlens`` package for the lens algebra (loading,
``J_l @ h`` transport, unembedding) so our numbers are parity-correct by
construction rather than by test. This module adds what ``jlens`` does not do:

* a finiteness check on load — the reference ``save()`` defaults to fp16 and
  published Gemma-4 artifacts contain entries at fp16's minimum subnormal, so a
  silently non-finite ``J`` is a real failure mode, not a hypothetical one;
* the workspace layer band, mapped from the source paper's 0-100 reindex onto a
  checkpoint's actual depth;
* the position floor. ``fit()`` skips the first 16 positions but ``apply()``
  reads them out anyway, out of distribution and unstable.
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
    """Workspace layer band for a checkpoint of ``n_layers``, inclusive of both
    ends.

    Reproduces the depth-verified band for every model in the prior survey:
    24 layers -> 9-22, 32 -> 12-29, 48 -> 18-44, 60 -> 23-55, 64 -> 24-59.

    The upper end is clamped to the last block: at the shallowest depths the
    fraction rounds past the final layer, and a readout there would index off
    the end of the model.
    """
    if n_layers < 2:
        raise ValueError(f"n_layers={n_layers} is too shallow to have a band")
    lo = round(BAND_START_FRAC * n_layers)
    hi = min(round(BAND_END_FRAC * n_layers), n_layers - 1)
    return list(range(lo, hi + 1))


@dataclass(frozen=True)
class Target:
    """A (checkpoint, lens) pair, resolved and provenance-stamped.

    ``lens_path`` and ``revision`` go into every artifact we write. The source
    paper's causal results are on closed models, and open-weight lens quality
    varies by family, so a number without its exact lens is not interpretable.
    """

    model_name: str
    lens_path: str
    revision: str | None = None

    @property
    def slug(self) -> str:
        return self.model_name.split("/")[-1]


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
    path: str, *, min_layers: int = 1, device: str | None = None
) -> JacobianLens:
    """Load a fitted lens, refusing anything non-finite.

    ``path`` is either a local file or ``"repo_id::filename"`` for a Hub
    artifact. Jacobians are promoted to fp32 regardless of storage dtype.

    ``device`` moves the Jacobians once at load. The reference ``transport()``
    does ``J.to(residual.device)`` on **every call**, so a lens left on the host
    re-copies ~105 MB per layer per readout, which dominates every sweep in this
    package: a Stage-1 record costs 955 ms with the lens on the host against
    376 ms with it resident. The cost is ~6.6 GB of card, which fits alongside a
    27B checkpoint on an 80 GB device. Leave it ``None`` on a smaller card.
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
    if len(lens.source_layers) < min_layers:
        raise ValueError(
            f"{path}: only {len(lens.source_layers)} fitted layers, "
            f"need >= {min_layers}"
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

    bf16 weights with fp32 readout arithmetic: verified on real weights at
    cosine 0.999967-1.000000 against an fp32 forward.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, revision=revision
    ).to(device)
    return from_hf(hf_model, tokenizer)


def check_positions(positions: list[int], seq_len: int) -> None:
    """Raise if any position is outside the lens's fitted range.

    Negative indices are resolved against ``seq_len`` first, so a caller using
    ``-1`` on a short prompt gets caught rather than silently reading an
    unfitted position.
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
