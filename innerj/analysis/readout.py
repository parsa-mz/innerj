"""Dependent variables for workspace entry.

Three measures of one concept, computed together by :func:`concept_scores` so no
experiment
can record one and lose the others:

* ``R_z`` (:func:`percentile_rank`) -- what every published number was measured on. It
  **saturates**: ranks 1 and 25 differ by 0.0001 on 248,320 tokens, and 92% of
  flexible-arm cells at L40--L44 read above 0.999.
* ``-log10(rank)`` (:func:`neg_log_rank`) -- the non-saturating companion, placing the
  depth peak ~15 layers deeper (the profiles correlate at only ``r=0.581``).
* ``M_z`` (:func:`logprob_margin`) -- the margin against rival candidates.

All three are read after the model's own scale-free final norm, so all three are
invariant
to a function-preserving rescale. Raw coefficient magnitude, distance to identity and
unnormalised lens norms are not, and must not support a claim. ``j_access`` is absent:
with
support ``k~16`` it is a step function of rank, so a run of zeros carries no trend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch


def single_token_id(tokenizer: Any, word: str, *, continuation: bool = True) -> int:
    """Token id for ``word``, requiring exactly one token.

    ``continuation=True`` scores the mid-sentence form (``" Spanish"``), a *different
    id* from
    the bare form: worth 0/36 against 10/36 on a real checkpoint. Failing labels are
    excluded
    from the dataset, never worked around.
    """
    form = f" {word}" if continuation else word
    ids = tokenizer.encode(form, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(
            f"{form!r} is {len(ids)} tokens for this tokenizer, not 1 "
            f"(ids={ids}). Exclude the label."
        )
    return int(ids[0])


def concept_rank(logits: torch.Tensor, token_id: int) -> int:
    """0-indexed rank of ``token_id``: how many tokens outscore it. 0 is top-1."""
    if logits.ndim != 1:
        raise ValueError(
            f"expected a 1-D logit vector, got shape {tuple(logits.shape)}"
        )
    return int((logits > logits[token_id]).sum())


def percentile_rank(logits: torch.Tensor, token_id: int) -> float:
    """``R_z``: 1.0 when the concept is top-1, 0.0 when last. Saturates -- always pair
        with :func:`neg_log_rank`.
    """
    vocab = logits.shape[0]
    return 1.0 - concept_rank(logits, token_id) / (vocab - 1)


def neg_log_rank(logits: torch.Tensor, token_id: int) -> float:
    """``-log10(rank)``, 1-indexed: the non-saturating companion to ``R_z``.

    Even spacing in orders of magnitude is the point, the interesting region being the
    top few
    hundred of a quarter-million tokens. Unbounded, and not comparable across vocabulary
    sizes.
    """
    return -math.log10(concept_rank(logits, token_id) + 1)


def logprob_margin(
    logits: torch.Tensor, token_id: int, control_ids: list[int]
) -> float:
    """``M_z``: log-prob of the concept minus the mean over matched controls.

    An overcomplete readout assigns *some* mass to almost anything, so the contrast
    carries the
    measurement; controls must be matched on frequency and part of speech.
    """
    if not control_ids:
        raise ValueError("logprob_margin needs at least one matched control")
    logp = torch.log_softmax(logits.float(), dim=-1)
    return float(logp[token_id] - logp[control_ids].mean())


@dataclass(frozen=True)
class ConceptScores:
    """All three measures from one logit vector, so an experiment cannot record one and
        lose the others -- recovering a missing one costs a full GPU re-run.
    """

    rank: int
    r_z: float
    neg_log_rank: float
    m_z: float


def concept_scores(
    logits: torch.Tensor, token_id: int, control_ids: list[int]
) -> ConceptScores:
    """Rank, ``R_z``, ``-log10(rank)`` and ``M_z`` in one pass, sharing one rank."""
    rank = concept_rank(logits, token_id)
    vocab = logits.shape[0]
    return ConceptScores(
        rank=rank,
        r_z=1.0 - rank / (vocab - 1),
        neg_log_rank=-math.log10(rank + 1),
        m_z=logprob_margin(logits, token_id, control_ids),
    )


def single_token_subset(tokenizer: Any, words: list[str]) -> list[str]:
    """Keep the words that are one token in continuation form (trap 6: never assume)."""
    out = []
    for word in words:
        try:
            single_token_id(tokenizer, word)
        except ValueError:
            continue
        out.append(word)
    return out


def transported_logits(
    model: Any, lens: Any, residuals: dict[int, torch.Tensor], layer: int, *, row: int
) -> torch.Tensor:
    """Vocabulary logits from a recorded residual, through the lens at ``layer``.

    ``row`` is keyword-only with no default (trap 7): for a one-position patch ``[0]``
    and
    ``[-1]`` agree, but on a ``last12`` span ``[0]`` is twelve positions before the
    query and
    the number still looks fine.
    """
    return model.unembed(lens.transport(residuals[layer], layer)).float()[row]


def band_scores(
    model: Any,
    lens: Any,
    residuals: dict[int, torch.Tensor],
    layers: list[int],
    token_id: int,
    control_ids: list[int],
    *,
    row: int,
) -> tuple[float, float, float]:
    """Mean ``(R_z, -log10(rank), M_z)`` of ``token_id`` over ``layers``.

    Reuses residuals from a pass already run, which is what makes a several-hundred-cell
    sweep
    affordable.
    """
    scores = [
        concept_scores(
            transported_logits(model, lens, residuals, layer, row=row),
            token_id,
            control_ids,
        )
        for layer in layers
    ]
    return (
        float(sum(s.r_z for s in scores) / len(scores)),
        float(sum(s.neg_log_rank for s in scores) / len(scores)),
        float(sum(s.m_z for s in scores) / len(scores)),
    )


def forced_choice(
    logits: torch.Tensor, candidate_ids: list[int]
) -> tuple[int, float]:
    r"""Argmax over the task's candidates, with the winner's margin over the best rival.

    Open-vocabulary argmax measures formatting: the top token is often ``'\n\n'`` with
    gold at
    rank 1, reading 0.144 accuracy where forced choice reads 0.955.
    """
    if not candidate_ids:
        raise ValueError("forced_choice needs a candidate set")
    scores = logits[candidate_ids].float()
    order = torch.argsort(scores, descending=True)
    best = candidate_ids[int(order[0])]
    if len(candidate_ids) == 1:
        margin = 0.0
    else:
        margin = float(scores[order[0]] - scores[order[1]])
    return best, margin


@dataclass
class LayerReadout:
    """Workspace-entry measures for one concept at one layer and position."""

    layer: int
    position: int
    rank: int
    r_z: float
    m_z: float
    neg_log_rank: float


@dataclass
class EntryReadout:
    """Per-layer readouts for one trial, plus band summaries.

    ``band_mean_r_z`` is primary; ``band_r_z`` (the maximum) is secondary, since a max
    over ~36
    layers reads near 1.0 whenever any one ranks the concept well, compressing +0.64 to
    +0.02.
    """

    gold_token_id: int
    layers: list[LayerReadout] = field(default_factory=list)

    @property
    def band_mean_r_z(self) -> float:
        """Mean ``R_z`` across the band: the primary summary DV."""
        if not self.layers:
            return float("nan")
        return sum(lr.r_z for lr in self.layers) / len(self.layers)

    @property
    def band_mean_m_z(self) -> float:
        if not self.layers:
            return float("nan")
        return sum(lr.m_z for lr in self.layers) / len(self.layers)

    @property
    def band_mean_neg_log_rank(self) -> float:
        """Mean ``-log10(rank)`` across the band; moves where ``R_z`` is pinned near
        1."""
        if not self.layers:
            return float("nan")
        return sum(lr.neg_log_rank for lr in self.layers) / len(self.layers)

    @property
    def band_r_z(self) -> float:
        """Maximum ``R_z`` across the band. Secondary: saturates. See above."""
        return max((lr.r_z for lr in self.layers), default=float("nan"))

    @property
    def band_m_z(self) -> float:
        return max((lr.m_z for lr in self.layers), default=float("nan"))

    @property
    def best_layer(self) -> int | None:
        if not self.layers:
            return None
        return max(self.layers, key=lambda lr: lr.r_z).layer

    def n_layers_above(self, threshold: float) -> int:
        """How many band layers hold the concept above ``threshold`` of ``R_z``."""
        return sum(1 for lr in self.layers if lr.r_z > threshold)


def read_entry(
    lens_logits: dict[int, torch.Tensor],
    *,
    gold_token_id: int,
    control_ids: list[int],
    position_index: int = 0,
    position: int | None = None,
) -> EntryReadout:
    """Assemble an :class:`EntryReadout` from ``jlens``-style lens logits.

    Args:
        lens_logits: ``{layer: Tensor[n_positions, vocab]}``, as returned by
            ``JacobianLens.apply``.
        gold_token_id: The concept whose entry is being measured.
        control_ids: Matched control tokens for ``M_z``.
        position_index: Which row of the ``n_positions`` axis to read.
        position: The absolute sequence position, recorded for provenance.
    """
    out = EntryReadout(gold_token_id=gold_token_id)
    for layer in sorted(lens_logits):
        row = lens_logits[layer][position_index]
        scores = concept_scores(row, gold_token_id, control_ids)
        out.layers.append(
            LayerReadout(
                layer=layer,
                position=position if position is not None else position_index,
                rank=scores.rank,
                r_z=scores.r_z,
                m_z=scores.m_z,
                neg_log_rank=scores.neg_log_rank,
            )
        )
    return out
