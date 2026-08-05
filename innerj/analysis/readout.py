"""Dependent variables for workspace entry.

Three measures of the same concept, computed together by :func:`concept_scores`
so that no experiment can record one and lose the others:

* ``R_z`` (:func:`percentile_rank`) -- comparable to the source paper, and the
  quantity every published number here was measured on. It **saturates**: 92% of
  flexible-arm cells at L40--L44 read above 0.999, which flattens a real
  mid-band peak.
* ``-log10(rank)`` (:func:`neg_log_rank`) -- the non-saturating companion. It
  moves freely where ``R_z`` is pinned at the ceiling, and it places the depth
  peak about 15 layers deeper than ``R_z`` does (the two profiles correlate at
  only ``r=0.581``). A conclusion that holds under one and not the other is a
  fact about the metric.
* ``M_z`` (:func:`logprob_margin`) -- the contrastive margin against the rival
  concepts, i.e. ``lambda_z`` minus the mean over the other candidates. Graded,
  unbounded, and unaffected by ceiling effects in either direction.

Every one is invariant to a function-preserving rescaling of the residual stream
``h_l -> C_l h_l``, because each is read *after* the model's own final norm,
which is scale-free. That rules out the class of diagnostics that turn out to
measure a coordinate choice rather than a property of the model: raw coefficient
magnitude, raw distance to the identity, and unnormalised lens norms never appear
here, and must not support a claim.

The sparse nonnegative pursuit coefficient (``j_access``) is deliberately absent:
with a support of ``k~16`` it is a *step function of rank*, reading exactly 0.000
whenever the concept ranks below the support size, so a run of zeros carries no
trend and any correlation against it is zero by construction. It belongs in a
secondary column, added when there is a rank effect to corroborate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch


def single_token_id(tokenizer: Any, word: str, *, continuation: bool = True) -> int:
    """Token id for ``word``, requiring it to be exactly one token.

    ``continuation=True`` scores the mid-sentence form (``" Spanish"``), which
    is what a model emits after a prompt ending in ``"Answer:"``. The bare form
    is a *different id* on any BPE vocabulary, and scoring it instead is worth
    the difference between 0/36 and 10/36 on a real checkpoint.

    Raises:
        ValueError: If the form is not single-token for this tokenizer. Labels
            that fail are excluded from the dataset rather than worked around.
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
    """``R_z``: 1.0 when the concept is top-1, 0.0 when last.

    A percentile rather than a raw rank so that the quantity is comparable
    across checkpoints with different vocabulary sizes.

    **This measure saturates**, and the saturation is not benign. With a
    248,320-token vocabulary, ranks 1 and 25 differ by 0.0001, so once a concept
    is anywhere near the top the score is pinned: 92% of flexible-arm cells at
    L40--L44 read above 0.999. A depth profile built from it therefore flattens
    exactly where the effect is largest. Pair it with :func:`neg_log_rank`, which
    resolves that region, and treat any conclusion that holds under one and not
    the other as a fact about the metric rather than about the model.
    """
    vocab = logits.shape[0]
    return 1.0 - concept_rank(logits, token_id) / (vocab - 1)


def neg_log_rank(logits: torch.Tensor, token_id: int) -> float:
    """``-log10(rank)`` on a 1-indexed rank: the non-saturating companion to ``R_z``.

    Rank is 1-indexed here so that top-1 maps to ``0.0`` and the logarithm is
    always defined; larger is better, matching ``R_z``'s direction so the two can
    be read off the same axis sign. Unlike ``R_z`` the spacing is even in orders
    of magnitude, so rank 1 -> 2 and rank 1000 -> 2000 move it equally --- which
    is the whole point, since the interesting region for a concept entering the
    workspace is the top few hundred of a quarter-million-token vocabulary, and
    ``R_z`` compresses all of it into its last thousandth.

    It is *not* bounded, and it is not comparable across vocabulary sizes the way
    ``R_z`` is. Both are recorded; neither replaces the other.
    """
    return -math.log10(concept_rank(logits, token_id) + 1)


def logprob_margin(
    logits: torch.Tensor, token_id: int, control_ids: list[int]
) -> float:
    """``M_z``: log-prob of the concept minus the mean over matched controls.

    Controls carry the work here. An overcomplete readout assigns *some* mass to
    almost anything, so an absolute score is uninterpretable; the contrast
    against frequency- and part-of-speech-matched tokens is what makes it a
    measurement. Pass controls that are matched, not merely arbitrary.
    """
    if not control_ids:
        raise ValueError("logprob_margin needs at least one matched control")
    logp = torch.log_softmax(logits.float(), dim=-1)
    return float(logp[token_id] - logp[control_ids].mean())


@dataclass(frozen=True)
class ConceptScores:
    """All three concept measures from one logit vector.

    Exists so that an experiment cannot record one metric and lose the others.
    Every patching artifact written before this class stored ``R_z`` alone, which
    meant the saturation question could not be answered without re-running the
    model --- hours of GPU per sweep to recover a number that was free at
    measurement time. Emit the whole struct.
    """

    rank: int
    r_z: float
    neg_log_rank: float
    m_z: float


def concept_scores(
    logits: torch.Tensor, token_id: int, control_ids: list[int]
) -> ConceptScores:
    """Rank, ``R_z``, ``-log10(rank)`` and ``M_z`` in one pass.

    The rank is computed once and the two rank-derived measures share it, so this
    costs no more than :func:`percentile_rank` alone plus one log-softmax.
    """
    rank = concept_rank(logits, token_id)
    vocab = logits.shape[0]
    return ConceptScores(
        rank=rank,
        r_z=1.0 - rank / (vocab - 1),
        neg_log_rank=-math.log10(rank + 1),
        m_z=logprob_margin(logits, token_id, control_ids),
    )


def forced_choice(
    logits: torch.Tensor, candidate_ids: list[int]
) -> tuple[int, float]:
    """Argmax restricted to the task's candidate set, and the winner's margin.

    Open-vocabulary argmax measures output formatting, not knowledge: on the
    primary checkpoint the top token is often ``'\\n\\n'`` with the gold answer
    at rank 1, which reads as 0.144 accuracy where forced choice over the same
    trials reads 0.955. Behavioural accuracy is always forced choice; the
    open-vocabulary number is kept only as a diagnostic.

    Returns:
        ``(winning_id, margin)`` where margin is the winner's logit minus the
        best rival's. Margin is 0.0 for a single candidate.
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

    ``band_mean_r_z`` is the primary summary. ``band_r_z`` (the maximum) is kept
    as a secondary because it **saturates**: a maximum over ~36 band layers reads
    near 1.0 whenever any single layer ranks the concept well, which compresses a
    real per-layer effect of +0.64 down to +0.02. That is a property of a
    saturating order statistic over many draws, not a property of the model, and
    it is the reason the mean leads.
    """

    gold_token_id: int
    layers: list[LayerReadout] = field(default_factory=list)

    @property
    def band_mean_r_z(self) -> float:
        """Mean ``R_z`` across the band -- the primary summary DV."""
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
        """Mean ``-log10(rank)`` across the band: the non-saturating companion.

        Where ``band_mean_r_z`` is pinned near 1.0 this still moves, which is why
        both are reported. They disagree about *where* in depth the concept is
        most visible, and that disagreement is a result, not a nuisance.
        """
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
        """How many band layers hold the concept above ``threshold`` of ``R_z``.

        A breadth measure: a concept present across many layers is more securely
        in the workspace than one that spikes at a single depth.
        """
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
