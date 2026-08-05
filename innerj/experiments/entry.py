"""Stage 1: separate latent availability from workspace entry.

The claim this stage has to establish is a dissociation, not an effect:

    ``z`` is present and usable in *both* the automatic and the flexible
    condition, but strongly represented in J-space only in the flexible one.

Without it, "the write gate" is indistinguishable from "the model knows ``z``",
and everything downstream measures the wrong thing. So the stage reports three
quantities per instance and the contrast between conditions for each:

* ``R_z`` at the query position, across the workspace band -- workspace entry;
* forced-choice accuracy -- behavioural availability;
* the model's own next-token distribution -- a sanity channel that never touches
  the lens, so it cannot inherit a lens artifact.

Measurement is pinned to the query position, never a passive sentence-final
read. A workspace is for holding things until they are asked for; reading it
where nothing asks understates persistence badly enough that a prior project
retracted its own headline over it.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from jlens import JacobianLens
from jlens.hf import HFLensModel

from innerj import console
from innerj.analysis.readout import EntryReadout, forced_choice, read_entry
from innerj.analysis.stats import Estimate, paired_bootstrap
from innerj.model import check_positions
from innerj.tasks.base import Condition, Record


@dataclass
class Trial:
    """One record measured: entry at the query position, plus behaviour."""

    record_id: str
    semantic_instance_id: str
    family: str
    condition: str
    query_position: int
    seq_len: int
    band_mean_r_z: float
    band_mean_m_z: float
    band_r_z: float
    band_m_z: float
    n_candidates: int
    best_layer: int | None
    n_layers_above_99: int
    correct: bool
    fc_margin: float
    open_vocab_top: int
    per_layer: dict[int, float]

    @classmethod
    def build(
        cls,
        record: Record,
        readout: EntryReadout,
        *,
        position: int,
        seq_len: int,
        correct: bool,
        fc_margin: float,
        open_vocab_top: int,
    ) -> Trial:
        return cls(
            record_id=record.id,
            semantic_instance_id=record.semantic_instance_id,
            family=record.family,
            condition=str(record.condition),
            query_position=position,
            seq_len=seq_len,
            band_mean_r_z=readout.band_mean_r_z,
            band_mean_m_z=readout.band_mean_m_z,
            band_r_z=readout.band_r_z,
            band_m_z=readout.band_m_z,
            n_candidates=len(record.candidate_token_ids),
            best_layer=readout.best_layer,
            # 0.99 of the vocabulary is still ~2500 tokens on a 248k vocabulary,
            # so this is a breadth measure, not a top-1 count.
            n_layers_above_99=readout.n_layers_above(0.99),
            correct=correct,
            fc_margin=fc_margin,
            open_vocab_top=open_vocab_top,
            per_layer={lr.layer: lr.r_z for lr in readout.layers},
        )


@torch.no_grad()
def measure(
    model: HFLensModel,
    lens: JacobianLens,
    record: Record,
    *,
    layers: list[int],
    max_seq_len: int = 512,
) -> Trial:
    """Measure one record at its query position.

    The query position is the final prompt token -- the point at which the model
    has been asked and must answer. Entry is read there; behaviour is scored on
    the same forward pass, so the two cannot drift apart.
    """
    prompt = record.prompt
    input_ids = model.encode(prompt, max_length=max_seq_len)
    seq_len = int(input_ids.shape[1])
    query = seq_len - 1
    check_positions([query], seq_len)

    lens_logits, model_logits, _ = lens.apply(
        model, prompt, layers=layers, positions=[query], max_seq_len=max_seq_len
    )
    readout = read_entry(
        lens_logits,
        gold_token_id=record.latent_token_id,
        control_ids=record.control_token_ids,
        position_index=0,
        position=query,
    )

    row = model_logits[0]
    winner, margin = forced_choice(row, record.candidate_token_ids)
    gold_index = record.candidate_answers.index(record.gold_answer)
    gold_id = record.candidate_token_ids[gold_index]

    return Trial.build(
        record,
        readout,
        position=query,
        seq_len=seq_len,
        correct=bool(winner == gold_id),
        fc_margin=margin if winner == gold_id else -margin,
        open_vocab_top=int(row.argmax()),
    )


def run(
    model: HFLensModel,
    lens: JacobianLens,
    records: list[Record],
    *,
    layers: list[int],
    max_seq_len: int = 512,
) -> list[Trial]:
    """Measure every record, printing as it goes.

    Progress logging is permanent, not scaffolding: a long silent run against
    its own timeout is indistinguishable from a hang.
    """
    trials: list[Trial] = []
    for record in console.track(records, "processing"):
        trials.append(
            measure(model, lens, record, layers=layers, max_seq_len=max_seq_len)
        )
    return trials


@dataclass
class Dissociation:
    """The Stage-1 result for one condition pair.

    ``delta_entry`` is the headline: how much more of the concept is in the
    workspace when the task demands flexible reuse. It is computed on the *mean*
    over band layers, because the maximum saturates.

    ``delta_accuracy`` is the guard. If behaviour moves as much as entry, the two
    conditions differ in difficulty and the contrast is confounded rather than
    informative -- which is why the format-matched control arm exists.
    """

    family: str
    high: str
    low: str
    delta_entry: Estimate
    delta_margin: Estimate
    delta_entry_max: Estimate
    delta_accuracy: Estimate
    accuracy_high: float
    accuracy_low: float
    entry_high: float
    entry_low: float
    chance_high: float
    chance_low: float

    #: An entry contrast is not interpretable once the arms differ this much in
    #: accuracy: the comparison is then between a task the model can do and one it
    #: cannot, which is a difficulty confound rather than a demand effect.
    MAX_ACCURACY_GAP = 0.15

    def verdict(self) -> str:
        """A dissociation needs entry to move *and* both arms to be performable.

        The chance comparison is the check that matters and the one this function
        originally lacked. An arm at or below its own chance floor is not doing the
        task, so its readout is a readout of failure -- and because the floor
        depends on the candidate-set size, comparing accuracy against a fixed
        constant cannot detect it. A tracking family passed an earlier version of
        this guard with a report arm at 0.125 against a chance of 0.250.
        """
        if min(self.accuracy_high, self.accuracy_low) > 0.995:
            return "INVALID: both arms at ceiling, no behavioural variance"
        for name, accuracy, chance in (
            ("high", self.accuracy_high, self.chance_high),
            ("low", self.accuracy_low, self.chance_low),
        ):
            if accuracy <= chance:
                return (
                    f"INVALID: the {name} arm is at or below chance "
                    f"({accuracy:.3f} vs {chance:.3f}); it is not performing the "
                    f"task, so its readout is a readout of failure"
                )
        if abs(self.delta_accuracy.point) > self.MAX_ACCURACY_GAP:
            return (
                f"CONFOUNDED: arms differ by {self.delta_accuracy.point:+.3f} in "
                f"accuracy, so the entry contrast is a difficulty difference"
            )
        if not self.delta_entry.excludes_zero:
            return "NO ENTRY EFFECT: delta_entry interval includes zero"
        if self.delta_entry.point < 0:
            return "REVERSED: entry is higher in the low-demand condition"
        return "ENTRY EFFECT"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict()
        return d


def dissociation(
    trials: list[Trial],
    *,
    high: Condition = Condition.FLEXIBLE,
    low: Condition = Condition.AUTOMATIC,
    seed: int = 0,
) -> Dissociation:
    """Paired contrast between two conditions over shared semantic instances."""
    by_instance: dict[str, dict[str, Trial]] = defaultdict(dict)
    for t in trials:
        by_instance[t.semantic_instance_id][t.condition] = t

    paired = [
        (v[str(high)], v[str(low)])
        for v in by_instance.values()
        if str(high) in v and str(low) in v
    ]
    if not paired:
        raise ValueError(
            f"no instance has both {high} and {low}; an unpaired contrast is "
            f"not the quantity this stage claims to measure"
        )

    hi = [p[0] for p in paired]
    lo = [p[1] for p in paired]

    def arr(ts: list[Trial], field_name: str) -> np.ndarray:
        return np.array([getattr(t, field_name) for t in ts], dtype=float)

    return Dissociation(
        family=hi[0].family,
        high=str(high),
        low=str(low),
        delta_entry=paired_bootstrap(
            arr(hi, "band_mean_r_z"), arr(lo, "band_mean_r_z"), seed=seed
        ),
        delta_margin=paired_bootstrap(
            arr(hi, "band_mean_m_z"), arr(lo, "band_mean_m_z"), seed=seed
        ),
        delta_entry_max=paired_bootstrap(
            arr(hi, "band_r_z"), arr(lo, "band_r_z"), seed=seed
        ),
        delta_accuracy=paired_bootstrap(
            arr(hi, "correct"), arr(lo, "correct"), seed=seed
        ),
        accuracy_high=float(arr(hi, "correct").mean()),
        accuracy_low=float(arr(lo, "correct").mean()),
        entry_high=float(arr(hi, "band_mean_r_z").mean()),
        entry_low=float(arr(lo, "band_mean_r_z").mean()),
        chance_high=float((1.0 / arr(hi, "n_candidates")).mean()),
        chance_low=float((1.0 / arr(lo, "n_candidates")).mean()),
    )


@dataclass
class Interaction:
    """The 2x2 that separates latent-variable demand from compositional work.

    ``flexible`` differs from ``control`` in two ways at once: it must infer ``z``
    *and* apply a prompted operator to it. Matching prompt format and accuracy does
    not match computation, so the flexible-minus-control contrast is not by itself a
    measure of latent-variable demand. Crossing the two factors separates them:

    ==================  ===============  =================
    ``z`` inferred?     no operator      operator
    ==================  ===============  =================
    no                  ``control``      ``supplied``
    yes                 ``report``       ``flexible``
    ==================  ===============  =================

    ``operator_only`` is what an arm that applies the operator to a value *given in
    the prompt* achieves without inferring anything --- the size of the confound.
    ``latent_demand`` holds the operator fixed and varies only the inference, so it
    is the clean estimate. ``point`` is the interaction itself, bootstrapped as one
    paired quantity per instance rather than assembled from four point estimates,
    which would discard the pairing.
    """

    latent_demand: Estimate
    operator_only: Estimate
    inference_only: Estimate
    both: Estimate
    interaction: Estimate
    accuracy: dict[str, float]
    n: int

    @property
    def confound_share(self) -> float:
        """Fraction of the both-vs-neither effect that the operator alone reproduces.

        The number that says how much of a published ``flexible - control`` effect
        was never about the latent variable.
        """
        if self.both.point == 0:
            return float("nan")
        return self.operator_only.point / self.both.point

    def verdict(self) -> str:
        if not self.latent_demand.excludes_zero:
            return (
                f"NOT LATENT-VARIABLE DEMAND: holding the operator fixed, inference "
                f"adds {self.latent_demand}, which does not clear zero. The "
                f"both-vs-neither effect is compositional work."
            )
        ceiling = [a for a, v in self.accuracy.items() if v >= 0.999]
        note = (
            f" Caution: {', '.join(ceiling)} at accuracy 1.000, so that arm has no "
            f"behavioural variance and part of the contrast may be difficulty."
            if ceiling else ""
        )
        return (
            f"LATENT-VARIABLE DEMAND: with the operator held fixed, inference adds "
            f"{self.latent_demand}; the operator alone reproduces "
            f"{self.confound_share:.0%} of the both-vs-neither effect "
            f"({self.operator_only.point:+.4f} of {self.both.point:+.4f}), so that "
            f"contrast is not by itself a latent-variable measure.{note}"
        )

    def to_dict(self) -> dict:
        out = asdict(self)
        out["confound_share"] = self.confound_share
        out["verdict"] = self.verdict()
        return out


def interaction(
    trials: list[Trial],
    *,
    field_name: str = "band_mean_r_z",
    neither: Condition = Condition.CONTROL,
    operator: Condition = Condition.SUPPLIED,
    inference: Condition = Condition.REPORT,
    both: Condition = Condition.FLEXIBLE,
    seed: int = 0,
) -> Interaction:
    """Run the 2x2 above over instances present in all four arms."""
    by_instance: dict[str, dict[str, Trial]] = defaultdict(dict)
    for t in trials:
        by_instance[t.semantic_instance_id][t.condition] = t

    arms = {"neither": neither, "operator": operator,
            "inference": inference, "both": both}
    names = {k: str(v) for k, v in arms.items()}
    rows = [v for v in by_instance.values() if set(names.values()) <= set(v)]
    if not rows:
        raise ValueError(
            f"no instance covers all four arms {sorted(names.values())}; the "
            f"interaction is not defined without the full 2x2"
        )

    def col(key: str) -> np.ndarray:
        return np.array(
            [getattr(v[names[key]], field_name) for v in rows], dtype=float
        )

    n, o, i, b = col("neither"), col("operator"), col("inference"), col("both")
    # One paired quantity per instance, so the interval respects the pairing that
    # four separately pooled contrasts would throw away.
    per_instance = (b - o) - (i - n)
    return Interaction(
        latent_demand=paired_bootstrap(b, o, seed=seed),
        operator_only=paired_bootstrap(o, n, seed=seed),
        inference_only=paired_bootstrap(i, n, seed=seed),
        both=paired_bootstrap(b, n, seed=seed),
        interaction=paired_bootstrap(per_instance, np.zeros_like(per_instance),
                                     seed=seed),
        accuracy={
            names[k]: float(np.mean([v[names[k]].correct for v in rows]))
            for k in arms
        },
        n=len(rows),
    )


def layer_profile(trials: list[Trial], condition: Condition) -> dict[int, float]:
    """Mean ``R_z`` per layer for one condition -- where in depth entry happens."""
    sums: dict[int, list[float]] = defaultdict(list)
    for t in trials:
        if t.condition != str(condition):
            continue
        for layer, r_z in t.per_layer.items():
            sums[int(layer)].append(r_z)
    return {layer: float(np.mean(v)) for layer, v in sorted(sums.items())}


def save_trials(trials: list[Trial], path: str | Path) -> int:
    """Persist raw trials. Every reported number must be recomputable from these."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for t in trials:
            fh.write(json.dumps(asdict(t)) + "\n")
    return len(trials)

