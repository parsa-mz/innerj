"""Does the gather head's attention *route* change with task demand?

Every attention result elsewhere in this package patches the **output** of
``o_proj``. That answers "what does this component contribute", and it is silent on
the question a reviewer will ask about the gather: is the route opened on demand, or
is it always open with different content on it? A component whose contribution moves
with task demand is, functionally, a gate -- unless the route is identical across arms
and only what travels along it differs.

This module measures the route directly. For one clean forward pass per record it
records the attention pattern at the gather layer, and reduces it to a single number
per head: **how much of the query positions' attention mass lands on passage tokens**.
No patching, no generation, no lens.

Both outcomes answer the question, which is why this is worth a card:

* mass is demand-modulated -> the gather is itself gated, and the gate is the
  attention pattern. That is a routing gate, not an unmasker, and it is a positive
  result rather than a concession.
* mass is flat -> the route is open in both arms and only its content differs. That is
  a stronger negative than we can currently state.

Three things this cannot do, all stated rather than hidden. Only full-attention layers
expose a pattern: the hybrid's gated delta-net layers have no attention matrix to
read, so a linear-attention gather (which is what the tracking family uses) is out of
scope here. Attention mass is not information flow -- a head can attend heavily to a
span and write nothing useful from it. And most importantly, **the arms' instructions
are not the same length** (12 tokens for control, 14 for flexible on the language
family), so the query span covers a different mix of tokens in each arm and the
absolute contrast is confounded. That is why every measurement here is taken at more
than one layer: the interpretable quantity is not "does L39 move" --- on a first pass
all 24 heads moved, which is a property of the prompt pair, not of the gather --- but
"does L39 move **more than a layer that does not gather**".
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass

import numpy as np
import torch
from jlens.hf import HFLensModel

from innerj import console
from innerj.analysis.stats import Estimate, paired_bootstrap
from innerj.model import check_positions
from innerj.patch import layer_type
from innerj.positions import context_length
from innerj.tasks.base import Record


@dataclass
class AttentionObservation:
    """One record, one head: where the query positions look."""

    instance: str
    condition: str
    layer: int
    head: int
    #: Which positions the mass is measured *from*, e.g. ``"query:last12"``.
    positions: str
    #: Fraction of those positions' attention mass landing on context tokens.
    passage_mass: float
    #: The same, restricted to the final query token, as a sanity companion.
    passage_mass_last: float
    n_context: int
    n_prompt: int
    #: Tokens between the end of the context and the start of the measured span.
    #: Zero means the span begins exactly at the instruction. The instruction
    #: length differs by arm (12-14 tokens here), so at a fixed ``last_n`` the
    #: arms do not measure quite the same span -- see :func:`pool`.
    gap_to_context: int


@dataclass
class AttentionResult:
    """One head's flexible-minus-control difference in passage mass."""

    layer: int
    head: int
    positions: str
    delta_passage_mass: Estimate
    #: The same contrast restricted to the **final** token. The span measure reads
    #: a slightly different set of tokens in each arm, because the instruction
    #: differs in length; the last token is the same token in both arms by
    #: construction, so this column is the confound-free companion. They should
    #: agree in sign, and a disagreement means the span asymmetry is doing work.
    delta_passage_mass_last: Estimate
    mean_flexible: float
    mean_control: float
    #: Mean number of instruction tokens the span covers, per arm. Equal spans
    #: would make these identical.
    span_flexible: float
    span_control: float
    n: int

    def to_dict(self) -> dict:
        return asdict(self)


@contextmanager
def _attention_weights(model: HFLensModel, layer: int):
    """Yield a dict that fills with ``layer``'s attention probabilities.

    ``output_attentions=True`` on the whole model would allocate a
    ``[batch, heads, seq, seq]`` tensor for **every** layer and silently switch the
    implementation away from sdpa. We want one layer, so the module is asked directly
    and eager attention is forced only for the duration.
    """
    block = model.layers[layer]
    if not hasattr(block, "self_attn"):
        raise ValueError(
            f"layer {layer} is {layer_type(model, layer)}; a gated delta net has no "
            f"attention matrix to record. Pick a full-attention layer."
        )
    attention = block.self_attn
    captured: dict[str, torch.Tensor] = {}
    previous = getattr(attention.config, "_attn_implementation", None)

    def hook(_module, args, kwargs, output):
        # HF returns (hidden, weights) when the module is asked for weights; the
        # second element is None under sdpa, which is exactly the silent-failure
        # mode this guard exists for.
        weights = output[1] if isinstance(output, tuple) and len(output) > 1 else None
        if weights is None:
            raise RuntimeError(
                f"layer {layer} returned no attention weights. The implementation is "
                f"{getattr(attention.config, '_attn_implementation', '?')}; sdpa and "
                f"flash both discard the matrix."
            )
        captured["w"] = weights.detach()

    if previous is not None:
        attention.config._attn_implementation = "eager"
    handle = attention.register_forward_hook(hook, with_kwargs=True)
    try:
        yield captured
    finally:
        handle.remove()
        if previous is not None:
            attention.config._attn_implementation = previous


@torch.no_grad()
def attention_mass(
    model: HFLensModel,
    records: list[Record],
    *,
    layer: int,
    last_n: int = 12,
    position_label: str = "query:last12",
    max_seq_len: int = 512,
) -> list[AttentionObservation]:
    """Passage attention mass at ``layer``, per head, for each record.

    The denominator is the full row, which sums to 1 under a causal softmax, so the
    reported number is directly "what fraction of what this position looked at was
    passage". The context boundary is located by
    :func:`innerj.positions.context_length`, which verifies the context is a token
    prefix rather than assuming it.
    """
    observations: list[AttentionObservation] = []
    for record in console.track(records, "recording attention"):
        input_ids = model.encode(record.prompt, max_length=max_seq_len)
        seq_len = int(input_ids.shape[1])
        query = list(range(seq_len - last_n, seq_len))
        check_positions(query, seq_len)
        n_context = context_length(model, record, max_seq_len=max_seq_len)
        # Context tokens are 0..n_context-1 and the query span starts at
        # seq_len - last_n, so equality means exactly adjacent, not overlapping.
        # On this family they usually *are* adjacent: the instruction is 12-14
        # tokens, so last_n=12 lands on the instruction and nothing else.
        if n_context > seq_len - last_n:
            raise ValueError(
                f"{record.id}: the context runs to token {n_context} but the query "
                f"span starts at {seq_len - last_n}; they overlap, so passage mass "
                f"would include the query itself."
            )

        with _attention_weights(model, layer) as captured:
            model.forward(input_ids)
        weights = captured["w"][0].float()  # [heads, seq, seq]

        rows = weights[:, query, :]                     # [heads, last_n, seq]
        mass = rows[:, :, :n_context].sum(-1).mean(-1)  # [heads]
        last = weights[:, -1, :n_context].sum(-1)       # [heads]
        for head in range(weights.shape[0]):
            observations.append(
                AttentionObservation(
                    instance=record.semantic_instance_id,
                    condition=str(record.condition),
                    layer=layer,
                    head=head,
                    positions=position_label,
                    passage_mass=float(mass[head]),
                    passage_mass_last=float(last[head]),
                    n_context=n_context,
                    n_prompt=seq_len,
                    gap_to_context=(seq_len - last_n) - n_context,
                )
            )
    return observations


def pool(
    observations: list[AttentionObservation],
    *,
    high: str = "flexible",
    low: str = "control",
    seed: int = 0,
) -> list[AttentionResult]:
    """Paired ``high`` minus ``low`` passage mass, per head.

    Paired within the semantic instance, because the arms share a passage and the
    absolute mass is dominated by how long that passage is.

    Two measures per head, and both are reported because they fail differently. The
    span measure averages over the last ``k`` positions, which on this family covers
    the instruction --- and the instruction is 12 tokens in control and 14 in
    flexible, so the two arms average over slightly different tokens. The last-token
    measure has no such asymmetry: the final token is ``Answer:`` in every arm. If
    the two disagree, the span asymmetry is carrying the result and neither should be
    reported without the other.
    """
    by_head: dict[tuple[int, int, str], dict[str, dict[str, tuple]]] = {}
    for o in observations:
        key = (o.layer, o.head, o.positions)
        by_head.setdefault(key, {}).setdefault(o.instance, {})[o.condition] = (
            o.passage_mass, o.passage_mass_last, o.gap_to_context,
        )

    results = []
    for (layer, head, positions), instances in sorted(by_head.items()):
        paired = [
            (v[high], v[low]) for v in instances.values() if high in v and low in v
        ]
        if not paired:
            continue
        a = np.array([p[0][0] for p in paired])
        b = np.array([p[1][0] for p in paired])
        a_last = np.array([p[0][1] for p in paired])
        b_last = np.array([p[1][1] for p in paired])
        results.append(
            AttentionResult(
                layer=layer,
                head=head,
                positions=positions,
                delta_passage_mass=paired_bootstrap(a, b, seed=seed),
                delta_passage_mass_last=paired_bootstrap(a_last, b_last, seed=seed),
                mean_flexible=float(a.mean()),
                mean_control=float(b.mean()),
                span_flexible=float(np.mean([p[0][2] for p in paired])),
                span_control=float(np.mean([p[1][2] for p in paired])),
                n=len(paired),
            )
        )
    return results


def verdict(
    results: list[AttentionResult],
    *,
    gather_layer: int,
    control_layers: list[int],
) -> str:
    """What the pooled heads say about the routing-gate hypothesis.

    The comparison that matters is **gather layer against control layers**, not the
    gather layer against zero. Because the arms' instructions differ in length, the
    query span covers different tokens in each arm and every layer's passage mass
    shifts; on a first pass all 24 heads at L39 moved, which says nothing about the
    gather. A routing gate predicts the gather layer moves *more* than layers that do
    not gather.

    Four outcomes, and three of them are negative results we would report as such.
    """
    if not results:
        return "NO DATA"
    gather = [r for r in results if r.layer == gather_layer]
    controls = [r for r in results if r.layer in set(control_layers)]
    if not gather:
        return f"NO DATA at L{gather_layer}"

    top = max(gather, key=lambda r: abs(r.delta_passage_mass.point))
    span, last = top.delta_passage_mass, top.delta_passage_mass_last

    if not any(r.delta_passage_mass.excludes_zero for r in gather):
        widest = max(r.delta_passage_mass.hi - r.delta_passage_mass.lo
                     for r in gather)
        return (
            f"ROUTE FLAT: no head of {len(gather)} at L{gather_layer} changes its "
            f"passage attention mass with demand (widest interval {widest:.4f}). The "
            f"route is open in both arms; what differs is what travels on it."
        )
    if span.point * last.point <= 0 or not last.excludes_zero:
        return (
            f"REFUSED: L{gather_layer} H{top.head}'s span measure is {span} but its "
            f"last-token measure is {last}. The span covers "
            f"{top.span_flexible:.1f} instruction tokens in the high arm against "
            f"{top.span_control:.1f} in the low arm, so the arms do not average over "
            f"the same positions and this contrast is not interpretable."
        )
    if not controls:
        return (
            f"UNCONTROLLED: L{gather_layer} H{top.head} moves by {span}, but no "
            f"control layer was measured, so this cannot be distinguished from a "
            f"prompt-pair effect present at every layer."
        )

    # Before comparing layers at all, check whether the measure discriminates. If
    # every head at every layer shifts significantly, and the ranking of layers
    # depends on whether one summarises by median or by maximum, then the measure
    # is dominated by something global to the prompt pair and no layer comparison
    # it supports means anything. Saying which layer "wins" under a statistic
    # chosen after seeing the data is exactly the error this project keeps making.
    by_layer = {}
    for r in results:
        by_layer.setdefault(r.layer, []).append(abs(r.delta_passage_mass.point))
    significant = sum(1 for r in results if r.delta_passage_mass.excludes_zero)
    by_median = sorted(by_layer, key=lambda k: -float(np.median(by_layer[k])))
    by_max = sorted(by_layer, key=lambda k: -max(by_layer[k]))
    if significant == len(results) and by_median[0] != by_max[0]:
        return (
            f"UNINFORMATIVE: every one of {len(results)} heads across "
            f"{len(by_layer)} layers shifts significantly, and the layer ranking "
            f"inverts with the summary statistic (by median L{by_median[0]} leads, "
            f"by maximum L{by_max[0]} does). The contrast is dominated by a "
            f"difference between the two prompts that is present at every depth, so "
            f"it cannot say whether the gather's route in particular is "
            f"demand-dependent. A design that equalises the instruction across arms "
            f"is needed before this measure means anything."
        )

    best_control = max(controls, key=lambda r: abs(r.delta_passage_mass.point))
    control_median = float(
        np.median([abs(r.delta_passage_mass.point) for r in controls])
    )
    if abs(span.point) <= abs(best_control.delta_passage_mass.point):
        return (
            f"NOT SPECIFIC: L{gather_layer}'s largest head moves by {span.point:+.4f} "
            f"while L{best_control.layer} H{best_control.head}, which does not "
            f"gather, moves by {best_control.delta_passage_mass.point:+.4f}. The "
            f"shift is a property of the prompt pair, not of the gather: the route "
            f"is no more demand-dependent at the gather layer than elsewhere."
        )
    ratio = abs(span.point) / max(control_median, 1e-9)

    # Does the span measure need the last-token cross-check at all? The check
    # exists to catch a span whose tokens differ between arms. When the arms'
    # spans start at the same offset from the context (``gap_to_context`` equal),
    # they cover the same positions and there is nothing for it to catch --- the
    # span measure is itself confound-free, and a one-token window is simply a
    # smaller, weaker view of it. The paper's own 1-vs-12 position comparison says
    # that window is too small to carry the effect at all: `resid.L39` gives
    # +0.0250 [-0.0250,+0.0813] at the query position alone against +0.3375 at
    # twelve. So we only demand agreement where the span is misaligned.
    aligned = all(
        abs(r.span_flexible - r.span_control) < 1e-9 for r in results
    )
    last_median = {}
    for r in results:
        last_median.setdefault(r.layer, []).append(abs(r.delta_passage_mass_last.point))
    last_gather = float(np.median(last_median[gather_layer]))
    last_best_control = max(
        float(np.median(v)) for k, v in last_median.items() if k != gather_layer
    )
    if not aligned and last_gather <= last_best_control:
        return (
            f"ROUTE MODULATED, SPAN ONLY: over the {positions_of(results)} span "
            f"L{gather_layer} leads on both median and maximum --- H{top.head} moves "
            f"by {span}, {ratio:.1f}x the median control shift ({control_median:+.4f}) "
            f"--- but the arms' spans are misaligned here, and on the single final "
            f"token, which is identical across arms, L{gather_layer}'s median shift "
            f"({last_gather:.4f}) does not exceed the best control layer's "
            f"({last_best_control:.4f}). Regenerate with matched instruction lengths "
            f"before reading this as a fact about the layer."
        )
    last_leader = max(last_median, key=lambda k: float(np.median(last_median[k])))
    trailing = (
        f" The final token alone favours L{last_leader}, but a one-token window does"
        f" not carry this effect in the behavioural data either."
        if last_gather <= last_best_control
        else f" L{gather_layer} leads on the final-token measure as well."
    )
    return (
        f"ROUTE MODULATED: L{gather_layer} H{top.head} moves by {span} over the "
        f"{positions_of(results)} span, {ratio:.1f}x the median control-layer shift "
        f"({control_median:+.4f} over {len(controls)} heads at "
        f"{', '.join(f'L{x}' for x in control_layers)}), with the span "
        f"{'token-identical' if aligned else 'misaligned'} across arms. The gather is "
        f"itself demand-dependent, which is a routing gate rather than an unmasking "
        f"gate.{trailing}"
    )


def positions_of(results: list[AttentionResult]) -> str:
    """The position label these results were measured over, for verdict text."""
    labels = {r.positions for r in results}
    return labels.pop() if len(labels) == 1 else "mixed"
