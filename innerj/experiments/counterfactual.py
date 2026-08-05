"""Stage 2c: does the transported value change the *answer*?

A rank shift in a lens readout is evidence about a representation. It is not
evidence that the model uses it. The strongest available test turns the patch into
a behavioural prediction with a specific wrong answer attached.

Both target and donor are **flexible** records, so each carries its own lookup
table. Patch a component from a donor holding language `D` into a target holding
language `T`, then ask which symbol the model emits:

* ``T -> table[T]`` -- unaffected, the patch carried nothing;
* ``T -> table[D]`` -- the target's *own* table applied to the **donor's**
  language. The patch transported `D`, and the model then ran the operator the
  target's prompt defined.

That second outcome is the one that matters. `table[D]` is a symbol the donor's
prompt never contained (the tables are independently randomised), and the donor's
language never appears in the target's context, so nothing but transported content
can produce it. Generic damage produces neither -- it produces the remaining
distractor symbols or noise.

This is the difference between "the workspace representation changed" and "the
model read the changed representation and acted on it."

**Both halves are recorded per trial.** An earlier version scored only the
forced-choice argmax, which made "0 of 80 answer flips" the whole behavioural
story --- and a thresholded outcome cannot distinguish "the patch moved nothing"
from "the patch moved the donor's symbol substantially but not past the gold
symbol". The continuous columns (``donor_logit_*``, ``donor_prob_*``,
``donor_vs_other_margin_*``) settle that, and when a lens is supplied the donor
concept's readout shift is recorded on **the same trial** as the behaviour, so the
readout-versus-use dissociation becomes a within-trial fact rather than a
comparison of two differently sized samples.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from jlens import JacobianLens
from jlens.hf import HFLensModel

from innerj import console
from innerj.analysis.readout import concept_scores, forced_choice
from innerj.analysis.stats import Estimate, paired_bootstrap
from innerj.patch import Component, capture, run_patched
from innerj.positions import PositionFn, build, describe
from innerj.tasks.base import Record


def _restricted(
    logits: torch.Tensor, candidate_ids: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Logits and probabilities over the candidate set alone.

    The task is forced choice, so the probability that matters is normalised over
    the candidates rather than the full vocabulary --- on this checkpoint the
    open-vocabulary top token is often ``'\\n\\n'``, which would dominate any
    unrestricted probability and measure formatting instead of the answer.
    """
    scores = logits[candidate_ids].float()
    return scores, torch.softmax(scores, dim=-1)


@dataclass
class Behaviour:
    """Continuous read of one forward pass over the candidate set."""

    donor_logit: float
    gold_logit: float
    other_logit: float
    donor_prob: float
    donor_vs_other_margin: float

    @classmethod
    def score(
        cls,
        logits: torch.Tensor,
        candidate_ids: list[int],
        answers: list[str],
        *,
        gold_symbol: str,
        donor_symbol: str,
    ) -> Behaviour:
        scores, probs = _restricted(logits, candidate_ids)
        index = {answer: i for i, answer in enumerate(answers)}
        donor, gold = index[donor_symbol], index[gold_symbol]
        others = [
            i for answer, i in index.items()
            if answer not in (gold_symbol, donor_symbol)
        ]
        other_logit = float(scores[others].mean()) if others else float("nan")
        return cls(
            donor_logit=float(scores[donor]),
            gold_logit=float(scores[gold]),
            other_logit=other_logit,
            donor_prob=float(probs[donor]),
            # The graded companion to "did the argmax flip". Transport should raise
            # the donor's symbol *specifically*; destruction raises every non-gold
            # symbol together, which leaves this flat.
            donor_vs_other_margin=float(scores[donor]) - other_logit,
        )


@dataclass
class CounterfactualObservation:
    """One mismatched flexible-into-flexible patch, scored on the answer."""

    component: str
    positions: str
    target_instance: str
    donor_instance: str
    target_value: str
    donor_value: str
    gold_symbol: str
    donor_symbol: str
    clean_answer: str
    patched_answer: str
    clean_is_gold: bool
    patched_is_gold: bool
    patched_is_donor_symbol: bool
    clean_is_donor_symbol: bool
    patched_is_other: bool
    clean_is_other: bool
    n_other: int
    #: Continuous behaviour over the candidate set. A thresholded flip rate cannot
    #: separate "moved nothing" from "moved a lot but not past the gold symbol".
    donor_logit_clean: float
    donor_logit_patched: float
    gold_logit_clean: float
    gold_logit_patched: float
    other_logit_clean: float
    other_logit_patched: float
    donor_prob_clean: float
    donor_prob_patched: float
    donor_vs_other_margin_clean: float
    donor_vs_other_margin_patched: float
    #: The donor concept's J-lens readout on **this** trial, when a lens was
    #: supplied. ``None`` keeps the lens-free runs lens-free: the headline
    #: behavioural result is deliberately measured without touching the lens, and
    #: that independence is worth preserving.
    donor_rz_clean: float | None = None
    donor_rz_patched: float | None = None
    donor_logrank_clean: float | None = None
    donor_logrank_patched: float | None = None
    donor_mz_clean: float | None = None
    donor_mz_patched: float | None = None


def buildable_pairs(
    flexible: dict[str, Record], *, seed: int = 0
) -> list[tuple[Record, Record, str, str]]:
    """Pair targets with donors whose language maps to a *different* symbol.

    Requires the donor's language to appear in the target's table -- otherwise
    ``table[D]`` does not exist and the counterfactual has no predicted answer. It
    also requires ``table[D] != table[T]``, or the prediction is indistinguishable
    from no effect.
    """
    instances = sorted(flexible)
    rng = np.random.default_rng(seed)
    out: list[tuple[Record, Record, str, str]] = []
    for instance in instances:
        target = flexible[instance]
        table = target.meta.get("table") or {}
        gold_symbol = table.get(target.latent_value)
        if gold_symbol is None:
            continue
        candidates = [
            i
            for i in instances
            if (donor_value := flexible[i].latent_value) != target.latent_value
            and donor_value in table
            and table[donor_value] != gold_symbol
        ]
        if not candidates:
            continue
        donor = flexible[candidates[rng.integers(len(candidates))]]
        out.append((donor, target, gold_symbol, table[donor.latent_value]))
    if not out:
        raise ValueError(
            "no target has a donor whose language is in its table with a different "
            "symbol; the counterfactual cannot be built from this dataset"
        )
    return out


@torch.no_grad()
def counterfactual(
    model: HFLensModel,
    pairs: list[tuple[Record, Record, str, str]],
    groups: list[tuple[str, list[Component]]],
    *,
    positions: PositionFn | None = None,
    max_seq_len: int = 512,
    lens: JacobianLens | None = None,
    read_layers: list[int] | None = None,
) -> list[CounterfactualObservation]:
    """Patch each component and record which symbol the model chooses.

    ``groups`` are labelled component sets, each patched **together**. A group of one
    is a single-component patch; larger groups test whether a pathway that no single
    component carries is nonetheless carried by a small set of them.

    ``positions`` is resolved **per record**, so donor and target use different
    absolute indices for the same semantic location -- necessary because they are
    different instances with different passage lengths. Defaults to the query
    position alone.

    The position choice is an experimental axis, not a detail. Patching only the
    query region leaves the operator no downstream computation in which to consume a
    transported value, and patching only there also misses the passage, where the
    latent variable is constructed in the first place.

    Supplying ``lens`` and ``read_layers`` additionally records the donor concept's
    J-lens readout on the same trial, which is what makes the readout-versus-use
    comparison a within-trial one. Both or neither; the lens-free path is the
    default because the headline behavioural result is stronger for never having
    touched the lens.
    """
    if (lens is None) != (read_layers is None):
        raise ValueError(
            "pass both lens and read_layers or neither; a lens with no readout "
            "layers records nothing and layers with no lens cannot be read"
        )
    positions = positions or build("query", 1)
    # One capture per pair covers every group, since groups share components.
    components = sorted({c for _, group in groups for c in group})
    observations: list[CounterfactualObservation] = []
    for donor, target, gold_symbol, donor_symbol in console.track(pairs, "processing"):
        donor_positions = positions(model, donor)
        target_positions = positions(model, target)
        if len(donor_positions) != len(target_positions):
            raise ValueError(
                f"{donor.id} gives {len(donor_positions)} positions but {target.id} "
                f"gives {len(target_positions)}; the captured activation would not "
                f"fit the site it is written into"
            )
        donor_acts = capture(
            model, donor.prompt, components, positions=donor_positions,
            max_seq_len=max_seq_len,
        )
        # The clean run records the *same* positions as the patched one. It used to
        # record only the default final position, which left `clean_res` with one
        # row against the patched run's twelve --- so a fixed index read the final
        # token in one and `seq_len - 12` in the other, and the readout delta was
        # measuring the difference between two different positions. Every row below
        # is indexed `[-1]`, the final token, in both runs.
        clean_logits, clean_res = run_patched(
            model, target.prompt, {}, positions=target_positions,
            max_seq_len=max_seq_len, record_layers=read_layers,
        )
        clean_id, _ = forced_choice(clean_logits[-1], target.candidate_token_ids)
        answer_of = dict(
            zip(target.candidate_token_ids, target.candidate_answers, strict=True)
        )
        clean_answer = answer_of[clean_id]
        clean_behaviour = Behaviour.score(
            clean_logits[-1], target.candidate_token_ids, target.candidate_answers,
            gold_symbol=gold_symbol, donor_symbol=donor_symbol,
        )
        # The donor's language and its rivals, for the concept readout. Rivals
        # exclude the donor itself so the margin is contrastive.
        concepts = {target.latent_token_id, *target.control_token_ids}
        donor_rivals = sorted(concepts - {donor.latent_token_id})

        def readout(residuals, rivals=donor_rivals, token=donor.latent_token_id):
            """Mean (R_z, log-rank, M_z) of the donor concept over read_layers."""
            if lens is None or residuals is None:
                return (None, None, None)
            scores = [
                concept_scores(
                    model.unembed(lens.transport(residuals[layer], layer)).float()[-1],
                    token,
                    rivals,
                )
                for layer in read_layers
            ]
            return (
                float(np.mean([s.r_z for s in scores])),
                float(np.mean([s.neg_log_rank for s in scores])),
                float(np.mean([s.m_z for s in scores])),
            )

        clean_read = readout(clean_res)

        for label, group in groups:
            logits, patched_res = run_patched(
                model,
                target.prompt,
                {c: donor_acts[c] for c in group},
                positions=target_positions,
                max_seq_len=max_seq_len,
                record_layers=read_layers,
            )
            patched_id, _ = forced_choice(logits[-1], target.candidate_token_ids)
            patched_answer = answer_of[patched_id]
            others = set(target.candidate_answers) - {gold_symbol, donor_symbol}
            patched_behaviour = Behaviour.score(
                logits[-1], target.candidate_token_ids, target.candidate_answers,
                gold_symbol=gold_symbol, donor_symbol=donor_symbol,
            )
            patched_read = readout(patched_res)
            observations.append(
                CounterfactualObservation(
                    component=label,
                    positions=describe(positions),
                    target_instance=target.semantic_instance_id,
                    donor_instance=donor.semantic_instance_id,
                    target_value=target.latent_value,
                    donor_value=donor.latent_value,
                    gold_symbol=gold_symbol,
                    donor_symbol=donor_symbol,
                    clean_answer=clean_answer,
                    patched_answer=patched_answer,
                    clean_is_gold=clean_answer == gold_symbol,
                    patched_is_gold=patched_answer == gold_symbol,
                    patched_is_donor_symbol=patched_answer == donor_symbol,
                    clean_is_donor_symbol=clean_answer == donor_symbol,
                    patched_is_other=patched_answer in others,
                    clean_is_other=clean_answer in others,
                    n_other=len(others),
                    donor_logit_clean=clean_behaviour.donor_logit,
                    donor_logit_patched=patched_behaviour.donor_logit,
                    gold_logit_clean=clean_behaviour.gold_logit,
                    gold_logit_patched=patched_behaviour.gold_logit,
                    other_logit_clean=clean_behaviour.other_logit,
                    other_logit_patched=patched_behaviour.other_logit,
                    donor_prob_clean=clean_behaviour.donor_prob,
                    donor_prob_patched=patched_behaviour.donor_prob,
                    donor_vs_other_margin_clean=(
                        clean_behaviour.donor_vs_other_margin
                    ),
                    donor_vs_other_margin_patched=(
                        patched_behaviour.donor_vs_other_margin
                    ),
                    donor_rz_clean=clean_read[0],
                    donor_rz_patched=patched_read[0],
                    donor_logrank_clean=clean_read[1],
                    donor_logrank_patched=patched_read[1],
                    donor_mz_clean=clean_read[2],
                    donor_mz_patched=patched_read[2],
                )
            )
    return observations


@dataclass
class CounterfactualResult:
    """Behavioural consequence of transporting the donor's latent value."""

    component: str
    positions: str
    delta_gold: Estimate
    delta_donor_symbol: Estimate
    delta_donor_vs_other: Estimate
    clean_accuracy: float
    patched_accuracy: float
    donor_symbol_rate: float
    other_symbol_rate: float
    n: int
    #: Continuous behaviour. ``delta_donor_symbol`` above is a flip rate, and a flip
    #: rate of zero is compatible with a large logit movement that never overtakes
    #: the gold symbol. These separate "inert" from "moved but not decisive".
    delta_donor_logit: Estimate | None = None
    delta_donor_prob: Estimate | None = None
    delta_donor_vs_other_margin: Estimate | None = None
    #: Readout shift on the same trials, when a lens was supplied, and the
    #: trial-level correlation between it and the behavioural movement. This is the
    #: dissociation measured *within* trial: a component can move its own readout
    #: a long way and predict nothing about what the model then does.
    delta_readout_logrank: Estimate | None = None
    readout_behaviour_r: float | None = None
    readout_behaviour_n: int | None = None

    def verdict(self) -> str:
        """The donor's symbol must beat the *unrelated distractors*, not zero.

        Comparing against the clean baseline is not enough and produces confident
        false positives. A patch that merely destroys the computation drives the
        answer toward uniform over the candidate set, which lifts the donor-symbol
        rate from ~0 to ~1/n purely as an artifact -- observed at passage positions,
        where six components each read +0.20 to +0.23 against a 0.25 chance floor
        while accuracy fell to 0.18-0.28.

        Each target's table holds the gold symbol, the donor's symbol, and unrelated
        distractors. Randomisation raises all non-gold symbols equally; transport
        raises only the donor's. So the contrast against the matched per-distractor
        rate is the test.
        """
        if not self.delta_donor_vs_other.excludes_zero:
            if self.delta_donor_symbol.excludes_zero:
                return (
                    "DESTRUCTION: donor's symbol rises no more than unrelated "
                    "distractors, so the answer is being randomised"
                )
            # A flip rate that does not move is not the same as a patch that does
            # nothing. If the continuous margin moved, the honest statement is that
            # the patch had a graded influence too small to change the decision --
            # not that it was inert. Reporting the latter overstated one result here
            # for weeks.
            margin = self.delta_donor_vs_other_margin
            if margin is not None and margin.excludes_zero and margin.point > 0:
                return (
                    f"SUBTHRESHOLD: the donor's symbol gains {margin} in logit "
                    f"margin over the distractors without overtaking the gold "
                    f"symbol, so the patch has a graded influence that never "
                    f"changes the decision"
                )
            return "NO COUNTERFACTUAL: donor-symbol rate does not move"
        if self.delta_donor_vs_other.point < 0:
            return "REVERSED: donor's symbol is suppressed relative to distractors"
        return (
            "COUNTERFACTUAL ANSWER: the model applies its own table to the "
            "donor's value"
        )

    def to_dict(self) -> dict:
        out = asdict(self)
        out["verdict"] = self.verdict()
        return out


def pool_counterfactual(
    observations: list[CounterfactualObservation], *, seed: int = 0
) -> list[CounterfactualResult]:
    """Pool by component, with paired intervals over targets."""
    by_component: dict[tuple[str, str], list[CounterfactualObservation]] = {}
    for observation in observations:
        by_component.setdefault(
            (observation.component, observation.positions), []
        ).append(observation)

    results = []
    for (component, position_label), group in by_component.items():
        gold_patched = np.array([o.patched_is_gold for o in group], dtype=float)
        gold_clean = np.array([o.clean_is_gold for o in group], dtype=float)
        donor_patched = np.array(
            [o.patched_is_donor_symbol for o in group], dtype=float
        )
        donor_clean = np.array([o.clean_is_donor_symbol for o in group], dtype=float)
        # Per-distractor rate, so it is directly comparable to a single symbol's
        # rate rather than to the pooled mass over several.
        other_patched = np.array(
            [o.patched_is_other / max(o.n_other, 1) for o in group], dtype=float
        )

        def delta(field: str, rows: list = group) -> Estimate:
            return paired_bootstrap(
                np.array([getattr(o, f"{field}_patched") for o in rows], dtype=float),
                np.array([getattr(o, f"{field}_clean") for o in rows], dtype=float),
                seed=seed,
            )

        # Readout is optional, so the correlation is too. Pearson over trials
        # between the donor concept's readout shift and its behavioural shift: the
        # dissociation this project has already been wrong about once, now measured
        # within trial instead of across two differently sized samples.
        lensed = [o for o in group if o.donor_logrank_patched is not None]
        readout_delta = correlation = None
        if len(lensed) >= 3:
            read_shift = np.array(
                [o.donor_logrank_patched - o.donor_logrank_clean for o in lensed]
            )
            act_shift = np.array(
                [
                    o.donor_vs_other_margin_patched - o.donor_vs_other_margin_clean
                    for o in lensed
                ]
            )
            readout_delta = paired_bootstrap(
                np.array([o.donor_logrank_patched for o in lensed]),
                np.array([o.donor_logrank_clean for o in lensed]),
                seed=seed,
            )
            # A constant column has no correlation; returning 0.0 there would read
            # as "measured and null" rather than "not measurable".
            if read_shift.std() > 0 and act_shift.std() > 0:
                correlation = float(np.corrcoef(read_shift, act_shift)[0, 1])

        results.append(
            CounterfactualResult(
                component=component,
                positions=position_label,
                delta_gold=paired_bootstrap(gold_patched, gold_clean, seed=seed),
                delta_donor_symbol=paired_bootstrap(
                    donor_patched, donor_clean, seed=seed
                ),
                delta_donor_vs_other=paired_bootstrap(
                    donor_patched, other_patched, seed=seed
                ),
                clean_accuracy=float(gold_clean.mean()),
                patched_accuracy=float(gold_patched.mean()),
                donor_symbol_rate=float(donor_patched.mean()),
                other_symbol_rate=float(other_patched.mean()),
                n=len(group),
                delta_donor_logit=delta("donor_logit"),
                delta_donor_prob=delta("donor_prob"),
                delta_donor_vs_other_margin=delta("donor_vs_other_margin"),
                delta_readout_logrank=readout_delta,
                readout_behaviour_r=correlation,
                readout_behaviour_n=len(lensed) or None,
            )
        )
    return sorted(results, key=lambda r: -r.delta_donor_vs_other.point)
