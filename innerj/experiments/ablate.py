"""Stage 4: necessity. Is the gather required for flexible use?

Transport shows a component *can* carry the latent value. Necessity is the other
direction: remove it and see whether the behaviour that needs $z$ breaks while
behaviour that does not need $z$ survives.

The comparison that makes this informative is **across conditions, not across
components**. Ablating anything degrades a model somewhat, so "flexible accuracy
fell" is not evidence on its own. What would be evidence is a *dissociation*:
flexible and report accuracy fall while the format-matched control --- which needs
$z$ for nothing --- holds. The control arm is the whole design, exactly as in Stage 1.

Two ablation modes, because they answer slightly different questions and zero
ablation alone is known to be misleading:

* ``zero`` --- write zeros over the component's output. Simple, but it moves the
  activation off-distribution, so a large effect may be shock rather than lost
  information.
* ``mean`` --- write the component's mean output, estimated over other instances of
  the same condition. Removes the *instance-specific* signal while keeping the
  activation in distribution. This is the one to lead with.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from jlens.hf import HFLensModel

from innerj import console
from innerj.analysis.readout import forced_choice
from innerj.analysis.stats import Estimate, paired_bootstrap
from innerj.patch import Component, capture, run_patched
from innerj.tasks.base import Condition, Record


@torch.no_grad()
def activation_sums(
    model: HFLensModel,
    records: list[Record],
    components: list[Component],
    *,
    positions: list[int] | None = None,
    max_seq_len: int = 512,
) -> tuple[dict[Component, torch.Tensor], dict[str, dict[Component, torch.Tensor]]]:
    """Summed component outputs over ``records``, and each record's own value.

    Returning the sum rather than the mean is what makes a **leave-one-out** mean
    free: the mean excluding record ``i`` is ``(sum - x_i) / (n - 1)``, so no second
    pass is needed. That matters because the ablation baseline was estimated from
    the first 32 records of an arm and then evaluated on all of them, so every
    record in the calibration set was partly ablated towards itself.
    """
    positions = positions or [-1]
    total: dict[Component, torch.Tensor] = {}
    per_record: dict[str, dict[Component, torch.Tensor]] = {}
    for record in records:
        captured = capture(
            model, record.prompt, components, positions=positions,
            max_seq_len=max_seq_len,
        )
        per_record[record.id] = {c: v.float() for c, v in captured.items()}
        for component, value in captured.items():
            acc = total.get(component)
            total[component] = value.float() if acc is None else acc + value.float()
    return total, per_record


def mean_activation(
    model: HFLensModel,
    records: list[Record],
    components: list[Component],
    *,
    positions: list[int] | None = None,
    max_seq_len: int = 512,
) -> dict[Component, torch.Tensor]:
    """Mean output of each component over ``records``, for mean ablation.

    Estimated within a condition. Pooling conditions would mix in the very
    difference the experiment is about.
    """
    total, _ = activation_sums(
        model, records, components, positions=positions, max_seq_len=max_seq_len
    )
    return {c: v / len(records) for c, v in total.items()}


@dataclass
class AblationObservation:
    """One record, one component, one ablation mode."""

    component: str
    #: Which positions were ablated, e.g. ``"query:last12"``. See the note on
    #: :class:`~innerj.experiments.mediate.MediationObservation`.
    positions: str
    mode: str
    condition: str
    instance: str
    correct_clean: bool
    correct_ablated: bool
    margin_clean: float
    margin_ablated: float


@dataclass
class AblationResult:
    """Accuracy cost of removing a component, within one condition."""

    component: str
    #: Which positions were ablated, e.g. ``"query:last12"``. See the note on
    #: :class:`~innerj.experiments.mediate.MediationObservation`.
    positions: str
    mode: str
    condition: str
    delta_accuracy: Estimate
    delta_margin: Estimate
    accuracy_clean: float
    accuracy_ablated: float
    n: int

    def to_dict(self) -> dict:
        return asdict(self)


@torch.no_grad()
def ablate(
    model: HFLensModel,
    records: list[Record],
    components: list[Component],
    *,
    mode: str = "mean",
    positions: list[int] | None = None,
    position_label: str = "query:last1",
    max_seq_len: int = 512,
    calibration_n: int = 32,
) -> list[AblationObservation]:
    """Ablate each component on each record and score forced choice.

    ``calibration_n`` caps how many records per condition estimate the mean-ablation
    baseline. The mean is stable well before the full arm, and each record's own
    contribution is removed from the baseline used to ablate it.
    """
    if mode not in ("zero", "mean"):
        raise ValueError(f"unknown ablation mode {mode!r}")
    positions = positions or [-1]

    by_condition: dict[str, list[Record]] = {}
    for record in records:
        by_condition.setdefault(str(record.condition), []).append(record)

    # Sums rather than means, so the baseline for each record can exclude that
    # record. The old code averaged the first 32 of an arm and then evaluated on
    # every record including those 32, which ablates a calibration record partly
    # towards itself and shrinks its own measured effect.
    sums: dict[str, dict[Component, torch.Tensor]] = {}
    counts: dict[str, int] = {}
    own: dict[str, dict[Component, torch.Tensor]] = {}
    if mode == "mean":
        for condition, group in by_condition.items():
            sample = group[:calibration_n]
            sums[condition], captured = activation_sums(
                model, sample, components, positions=positions,
                max_seq_len=max_seq_len,
            )
            counts[condition] = len(sample)
            own.update(captured)
            console.detail(
                f"mean activations for {condition} over {len(sample)} records, "
                f"leave-one-out"
            )

    def baseline(record: Record, component: Component) -> torch.Tensor:
        """The condition mean for ``component``, excluding ``record`` itself."""
        condition = str(record.condition)
        total, n = sums[condition][component], counts[condition]
        mine = own.get(record.id, {}).get(component)
        if mine is None:
            return total / n
        if n <= 1:
            raise ValueError(
                f"{record.id} is the only calibration record for {condition}; a "
                f"leave-one-out mean is undefined and ablating it towards itself "
                f"would be a no-op reported as an effect"
            )
        return (total - mine) / (n - 1)

    observations: list[AblationObservation] = []
    for record in console.track(records, "ablating"):
        condition = str(record.condition)
        clean_logits, _ = run_patched(
            model, record.prompt, {}, positions=positions, max_seq_len=max_seq_len
        )
        gold_index = record.candidate_answers.index(record.gold_answer)
        gold_id = record.candidate_token_ids[gold_index]
        clean_id, clean_margin = forced_choice(
            clean_logits[-1], record.candidate_token_ids
        )

        for component in components:
            if mode == "zero":
                shape = capture(
                    model, record.prompt, [component], positions=positions,
                    max_seq_len=max_seq_len,
                )[component]
                replacement = torch.zeros_like(shape)
            else:
                replacement = baseline(record, component)
            logits, _ = run_patched(
                model, record.prompt, {component: replacement},
                positions=positions, max_seq_len=max_seq_len,
            )
            ablated_id, ablated_margin = forced_choice(
                logits[-1], record.candidate_token_ids
            )
            observations.append(
                AblationObservation(
                    component=str(component),
                    positions=position_label,
                    mode=mode,
                    condition=condition,
                    instance=record.semantic_instance_id,
                    correct_clean=bool(clean_id == gold_id),
                    correct_ablated=bool(ablated_id == gold_id),
                    margin_clean=clean_margin if clean_id == gold_id else -clean_margin,
                    margin_ablated=(
                        ablated_margin if ablated_id == gold_id else -ablated_margin
                    ),
                )
            )
    return observations


def pool(
    observations: list[AblationObservation], *, seed: int = 0
) -> list[AblationResult]:
    """Pool per (component, positions, mode, condition), paired over instances.

    ``positions`` is part of the key, not just carried through. Pooling two
    position modes together is trap 16: on the counterfactual artifact it made a
    null at passage positions look significantly positive.
    """
    grouped: dict[tuple[str, str, str, str], list[AblationObservation]] = {}
    for o in observations:
        grouped.setdefault(
            (o.component, o.positions, o.mode, o.condition), []
        ).append(o)

    results = []
    for (component, positions, mode, condition), group in grouped.items():
        clean = np.array([o.correct_clean for o in group], dtype=float)
        ablated = np.array([o.correct_ablated for o in group], dtype=float)
        results.append(
            AblationResult(
                component=component,
                positions=positions,
                mode=mode,
                condition=condition,
                delta_accuracy=paired_bootstrap(ablated, clean, seed=seed),
                delta_margin=paired_bootstrap(
                    np.array([o.margin_ablated for o in group]),
                    np.array([o.margin_clean for o in group]),
                    seed=seed,
                ),
                accuracy_clean=float(clean.mean()),
                accuracy_ablated=float(ablated.mean()),
                n=len(group),
            )
        )
    return sorted(
        results, key=lambda r: (r.component, r.positions, r.mode, r.condition)
    )


def dissociation(
    results: list[AblationResult],
    *,
    needs_z: tuple[Condition, ...] = (Condition.FLEXIBLE, Condition.REPORT),
    baseline: Condition = Condition.CONTROL,
    mode: str = "mean",
) -> dict[str, dict]:
    """Per component: does ablation cost the $z$-dependent arms more than control?

    ``mode`` selects which ablation the summary describes. It is not optional: zero
    and mean ablation answer different questions, and mixing them in one summary is
    how a stored artifact came to report a zero-ablation triple for a component whose
    prose quoted mean ablation.

    A component necessary for flexible use should show a larger accuracy drop in
    the arms that need $z$ than in the format-matched control. A component whose
    ablation costs every arm equally is generally damaging, which is not the same
    claim and must not be reported as if it were.
    """
    # Keyed on mode as well as condition. Without it a run carrying both `zero` and
    # `mean` results silently summarised whichever was written last, so
    # `D_necessity`'s stored selectivity for `resid.L39` reported the zero-ablation
    # triple while the paper quoted mean ablation beside it. Same failure shape as
    # pooling two position modes: the factor is in the data and not in the key.
    by_component: dict[tuple[str, str], dict[str, AblationResult]] = {}
    for r in results:
        if r.mode != mode:
            continue
        by_component.setdefault((r.component, r.mode), {})[r.condition] = r

    out: dict[str, dict] = {}
    for (component, _mode), arms in by_component.items():
        control = arms.get(str(baseline))
        if control is None:
            continue
        entry: dict[str, object] = {
            "control_delta": control.delta_accuracy.point,
        }
        selective = []
        for condition in needs_z:
            arm = arms.get(str(condition))
            if arm is None:
                continue
            gap = arm.delta_accuracy.point - control.delta_accuracy.point
            entry[f"{condition}_delta"] = arm.delta_accuracy.point
            entry[f"{condition}_minus_control"] = gap
            selective.append(gap)
        if selective:
            entry["selectivity"] = float(np.mean(selective))
            entry["verdict"] = (
                "SELECTIVE: costs the z-dependent arms more than control"
                if min(selective) < -0.02
                else "NON-SELECTIVE: no larger cost where z is needed"
            )
        out[component] = entry
    return out
