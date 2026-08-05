"""Component-level capture and patching.

This is the instrument for the rest of the project. Stage 1's counterfactual patch
and Stage 2's component screening are the same operation at different granularity:
run a *source* prompt, keep some component's output, then run a *target* prompt
with that output substituted, and see what moves.

Granularity matters for what a result can claim. Patching a whole residual stream
proves only that the information was somewhere upstream. Patching one attention
head's contribution, or one MLP's output, localises it. So three kinds:

* ``resid`` -- the decoder layer's output. Coarse; the upper bound on any effect.
* ``attn``  -- one head's slice of the attention output projection's input, which
  is that head's contribution before it is mixed into the stream.
* ``mlp``   -- one MLP block's output.

Patching is restricted to chosen token positions, defaulting to the query position.
Under a matched legend the *context* is identical across conditions, so context
positions align across arms; only the instruction differs in length. Positions are
therefore always resolved from the end of the sequence, never assumed equal.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

import torch
from jlens.hf import HFLensModel

ComponentKind = Literal["resid", "attn", "mlp"]


@dataclass(frozen=True)
class Component:
    """A patchable site.

    ``head`` selects one attention head's contribution; ``head=None`` on an
    ``attn`` component means the whole attention output, which is the unit the
    coarse sweep uses. ``resid`` and ``mlp`` take no head index.

    Ordered, so callers can deduplicate a component set deterministically. The
    ordering is written out rather than generated: ``order=True`` compares ``head``
    directly and raises on ``int`` against ``None``, which made any set containing
    both a whole attention output and one of its heads unsortable --- a crash that
    only appears when those two are requested in the same run.
    """

    kind: ComponentKind
    layer: int
    head: int | None = None

    def __post_init__(self) -> None:
        if self.kind != "attn" and self.head is not None:
            raise ValueError(f"{self.kind} components take no head index")

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Component):
            return NotImplemented
        # -1 sorts the whole branch before its own heads, which is the order a
        # reader expects and keeps `None` out of the comparison entirely.
        return (self.kind, self.layer, -1 if self.head is None else self.head) < (
            other.kind, other.layer, -1 if other.head is None else other.head
        )

    def __str__(self) -> str:
        if self.kind == "attn":
            return (
                f"attn.L{self.layer}.H{self.head}"
                if self.head is not None
                else f"attn.L{self.layer}.all"
            )
        return f"{self.kind}.L{self.layer}"

    @property
    def slices_a_head(self) -> bool:
        return self.kind == "attn" and self.head is not None


def attention_projection(model: HFLensModel, layer: int) -> tuple[torch.nn.Module, str]:
    """The attention branch's output projection at ``layer``, and its layer type.

    The primary checkpoint is a **hybrid**: ``full_attention_interval=4`` means only
    every fourth layer carries standard attention (``self_attn.o_proj``, 24 heads);
    the other 48 use a gated delta net (``linear_attn.out_proj``, 48 value heads).
    Both project a 6144-wide concatenation into the 5120-wide stream, so the code
    below is uniform once the module is resolved -- but the *number of heads*
    differs, and hard-coding either one would silently misalign three quarters of
    the model.

    Raises on an unrecognised layer rather than guessing, because a wrong module
    path produces plausible numbers instead of an error.
    """
    block = model.layers[layer]
    if hasattr(block, "self_attn") and hasattr(block.self_attn, "o_proj"):
        return block.self_attn.o_proj, "full_attention"
    if hasattr(block, "linear_attn") and hasattr(block.linear_attn, "out_proj"):
        return block.linear_attn.out_proj, "linear_attention"
    raise ValueError(
        f"layer {layer} is a {type(block).__name__} with children "
        f"{[n for n, _ in block.named_children()]}; no known attention output "
        f"projection. Add its layout explicitly."
    )


def layer_type(model: HFLensModel, layer: int) -> str:
    """``"full_attention"`` or ``"linear_attention"`` for ``layer``."""
    declared = getattr(model._hf_model.config.get_text_config(), "layer_types", None)
    if declared is not None and layer < len(declared):
        return str(declared[layer])
    return attention_projection(model, layer)[1]


def n_attention_heads(model: HFLensModel, layer: int) -> int:
    """Head count for ``layer``, which depends on its attention type."""
    text_config = model._hf_model.config.get_text_config()
    if layer_type(model, layer) == "linear_attention":
        return int(
            getattr(text_config, "linear_num_value_heads", None)
            or text_config.num_attention_heads
        )
    return int(text_config.num_attention_heads)


def head_width(model: HFLensModel, layer: int) -> int:
    """Per-head width of the attention output projection's input at ``layer``.

    Read from the module and the layer's own head count, never from ``head_dim``:
    grouped-query and linear-attention layers both have
    ``n_heads * head_dim != d_model``, and on the primary model the projection
    takes 6144 inputs against a 5120-wide stream.
    """
    projection, _ = attention_projection(model, layer)
    n_heads = n_attention_heads(model, layer)
    in_features = projection.in_features
    if in_features % n_heads:
        raise ValueError(
            f"layer {layer}: projection in_features={in_features} is not divisible "
            f"by {n_heads} heads; head slicing would misalign"
        )
    return in_features // n_heads


def _resolve(positions: list[int], seq_len: int) -> list[int]:
    """Resolve negative indices. -1 is the query position in every arm."""
    out = [p if p >= 0 else seq_len + p for p in positions]
    bad = [p for p in out if not 0 <= p < seq_len]
    if bad:
        raise ValueError(f"positions {bad} out of range for seq_len={seq_len}")
    return out


@contextmanager
def _hooks(handles: list) -> Iterator[None]:
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def capture(
    model: HFLensModel,
    prompt: str,
    components: list[Component],
    *,
    positions: list[int] | None = None,
    max_seq_len: int = 512,
) -> dict[Component, torch.Tensor]:
    """Run ``prompt`` and keep each component's output at ``positions``.

    Returns ``{component: Tensor[n_positions, width]}``, detached and on the
    component's own device.
    """
    positions = positions or [-1]
    input_ids = model.encode(prompt, max_length=max_seq_len)
    resolved = _resolve(positions, int(input_ids.shape[1]))
    captured: dict[Component, torch.Tensor] = {}
    handles = []

    def make_hook(component: Component, is_pre: bool):
        def hook(_module, args, output=None):
            tensor = args[0] if is_pre else output
            if isinstance(tensor, tuple):
                tensor = tensor[0]
            rows = tensor[0, resolved]
            if component.slices_a_head:
                width = head_width(model, component.layer)
                start = component.head * width
                rows = rows[:, start : start + width]
            captured[component] = rows.detach().clone()

        return hook

    for component in components:
        block = model.layers[component.layer]
        if component.kind == "resid":
            handles.append(block.register_forward_hook(make_hook(component, False)))
        elif component.kind == "mlp":
            handles.append(block.mlp.register_forward_hook(make_hook(component, False)))
        else:
            projection, _ = attention_projection(model, component.layer)
            handles.append(
                projection.register_forward_pre_hook(make_hook(component, True))
            )

    with _hooks(handles):
        model.forward(input_ids)

    missing = [c for c in components if c not in captured]
    if missing:
        raise RuntimeError(
            f"no activation captured for {[str(c) for c in missing]}; the hook "
            f"never fired, so the module path is wrong for this architecture"
        )
    return captured


@torch.no_grad()
def run_patched(
    model: HFLensModel,
    prompt: str,
    patches: dict[Component, torch.Tensor],
    *,
    positions: list[int] | None = None,
    max_seq_len: int = 512,
    record_layers: list[int] | None = None,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Run ``prompt`` with each component's output replaced at ``positions``.

    Returns ``(logits_at_positions, {layer: residual_at_positions})`` so a caller
    gets both the behavioural and the readout consequence from one forward pass.
    An empty ``patches`` dict gives the clean run, which is the baseline every
    effect is measured against.
    """
    positions = positions or [-1]
    input_ids = model.encode(prompt, max_length=max_seq_len)
    seq_len = int(input_ids.shape[1])
    resolved = _resolve(positions, seq_len)
    recorded: dict[int, torch.Tensor] = {}
    handles = []

    def make_patch_hook(component: Component, replacement: torch.Tensor, is_pre: bool):
        def hook(_module, args, output=None):
            tensor = args[0] if is_pre else output
            was_tuple = isinstance(tensor, tuple)
            rest = tensor[1:] if was_tuple else ()
            if was_tuple:
                tensor = tensor[0]
            tensor = tensor.clone()
            value = replacement.to(tensor.device, tensor.dtype)
            if component.slices_a_head:
                width = head_width(model, component.layer)
                start = component.head * width
                tensor[0, resolved, start : start + width] = value
            else:
                tensor[0, resolved] = value
            if is_pre:
                return (tensor, *args[1:])
            return (tensor, *rest) if was_tuple else tensor

        return hook

    for component, replacement in patches.items():
        block = model.layers[component.layer]
        if component.kind == "resid":
            handles.append(
                block.register_forward_hook(
                    make_patch_hook(component, replacement, False)
                )
            )
        elif component.kind == "mlp":
            handles.append(
                block.mlp.register_forward_hook(
                    make_patch_hook(component, replacement, False)
                )
            )
        else:
            projection, _ = attention_projection(model, component.layer)
            handles.append(
                projection.register_forward_pre_hook(
                    make_patch_hook(component, replacement, True)
                )
            )

    # The readout needs the stream after the last block, so it is always recorded.
    final_layer = model.n_layers - 1
    for layer in sorted(set(record_layers or []) | {final_layer}):

        def recorder(_module, _args, output, layer=layer):
            tensor = output[0] if isinstance(output, tuple) else output
            recorded[layer] = tensor[0, resolved].detach().float()

        handles.append(model.layers[layer].register_forward_hook(recorder))

    with _hooks(handles):
        model.forward(input_ids)

    logits = model.unembed(recorded[final_layer]).float().cpu()
    return logits, recorded
