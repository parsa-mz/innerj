"""Stage 2b: is the effect content-specific, or does patching there just help?

The coarse sweep found that many components move the readout when patched at the
query position, several of them *downward*. So "largest positive effect" does not
yet mean "carries the latent variable". A component could raise the gold concept's
rank by generally sharpening the readout, with no relation to which language the
donor was holding.

This is the test that separates the two. Patch a component from a donor whose
latent value is **different** from the target's, and read out *two* concepts:

* the **target's** gold language -- a generic-sharpening component still raises it;
* the **donor's** language -- only a component actually carrying the latent value
  raises this one.

A write component should transport the donor's content: donor-language rank up,
and ideally target-language rank down. A generic amplifier moves the target's rank
up and leaves the donor's untouched. The two predictions are opposite, so one
experiment decides it.

This is the "same component, wrong latent variable" control, and without it the
whole screen is compatible with an interpretability illusion where a subspace
intervention activates a pathway that has nothing to do with the model's own
mechanism.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from jlens import JacobianLens
from jlens.hf import HFLensModel

from innerj import console
from innerj.analysis.readout import percentile_rank
from innerj.analysis.stats import Estimate, paired_bootstrap
from innerj.patch import Component, capture, run_patched
from innerj.tasks.base import Record


@dataclass
class SpecificityObservation:
    """One (component, target, mismatched donor) trial, read on both concepts."""

    component: str
    target_instance: str
    donor_instance: str
    target_value: str
    donor_value: str
    target_entry_clean: float
    target_entry_patched: float
    donor_entry_clean: float
    donor_entry_patched: float


@dataclass
class SpecificityResult:
    """Does patching this component transport the donor's latent value?"""

    component: str
    delta_target: Estimate
    delta_donor: Estimate
    n: int

    @property
    def asymmetry(self) -> float:
        """How much more the donor's concept rises than the target's own.

        This, not the sign pattern, is the informative quantity. The donor's
        language is **absent from the target's context** while the target's own
        language is present, so any nonspecific sharpening of the readout should
        favour the *target*. A ratio well above 1 therefore cannot come from
        sharpening; it requires the patch to have carried content.

        Returns ``inf`` when the target's concept does not rise at all.
        """
        if self.delta_target.point <= 0:
            return float("inf") if self.delta_donor.point > 0 else 0.0
        return self.delta_donor.point / self.delta_target.point

    def verdict(self, *, min_asymmetry: float = 2.0) -> str:
        """Classify the pattern.

        ``min_asymmetry`` is a reporting threshold, not a test: the interval on
        each delta is what carries significance.
        """
        donor_up = self.delta_donor.excludes_zero and self.delta_donor.point > 0
        both_down = self.delta_donor.point < 0 and self.delta_target.point < 0
        if both_down:
            return "DISRUPTION: both concepts fall"
        if not donor_up:
            return "NO TRANSPORT: donor's concept does not rise"
        if self.asymmetry == float("inf"):
            return (
                "CONTENT-SPECIFIC: donor's concept rises while the target's own "
                "does not"
            )
        if self.asymmetry >= min_asymmetry:
            return (
                f"CONTENT-SPECIFIC: donor's concept rises {self.asymmetry:.1f}x "
                f"more than the target's own"
            )
        return "AMBIGUOUS: both rise comparably, consistent with sharpening"

    def to_dict(self) -> dict:
        out = asdict(self)
        out["asymmetry"] = self.asymmetry
        out["verdict"] = self.verdict()
        return out


def mismatched_pairs(
    flexible: dict[str, Record], automatic: dict[str, Record], *, seed: int = 0
) -> list[tuple[Record, Record]]:
    """Pair each automatic target with a flexible donor of a *different* language.

    Instances are matched on everything except the latent value, so a difference in
    outcome cannot come from passage length, template or operator family.
    """
    instances = sorted(set(flexible) & set(automatic))
    rng = np.random.default_rng(seed)
    out: list[tuple[Record, Record]] = []
    for instance in instances:
        target = automatic[instance]
        candidates = [
            i
            for i in instances
            if flexible[i].latent_value != target.latent_value
        ]
        if not candidates:
            continue
        donor = flexible[candidates[rng.integers(len(candidates))]]
        out.append((donor, target))
    if not out:
        raise ValueError(
            "no instance has a donor with a different latent value; a mismatched "
            "control cannot be built from this dataset"
        )
    return out


def _entry(
    lens: JacobianLens,
    model: HFLensModel,
    residuals: dict[int, torch.Tensor],
    layers: list[int],
    token_id: int,
) -> float:
    scores = []
    for layer in layers:
        logits = model.unembed(lens.transport(residuals[layer], layer)).float()[0]
        scores.append(percentile_rank(logits, token_id))
    return float(np.mean(scores))


@torch.no_grad()
def specificity(
    model: HFLensModel,
    lens: JacobianLens,
    pairs: list[tuple[Record, Record]],
    components: list[Component],
    *,
    read_layers: list[int],
    max_seq_len: int = 512,
) -> list[SpecificityObservation]:
    """Run the mismatched-donor patch and read out both concepts."""
    observations: list[SpecificityObservation] = []
    for donor, target in console.track(pairs, "processing"):
        if donor.latent_token_id == target.latent_token_id:
            raise ValueError(
                f"{donor.id} and {target.id} share a latent value; this control "
                f"requires them to differ"
            )
        donor_acts = capture(model, donor.prompt, components, max_seq_len=max_seq_len)
        _, clean = run_patched(
            model,
            target.prompt,
            {},
            max_seq_len=max_seq_len,
            record_layers=read_layers,
        )
        target_clean = _entry(lens, model, clean, read_layers, target.latent_token_id)
        donor_clean = _entry(lens, model, clean, read_layers, donor.latent_token_id)

        for component in components:
            _, residuals = run_patched(
                model,
                target.prompt,
                {component: donor_acts[component]},
                max_seq_len=max_seq_len,
                record_layers=read_layers,
            )
            observations.append(
                SpecificityObservation(
                    component=str(component),
                    target_instance=target.semantic_instance_id,
                    donor_instance=donor.semantic_instance_id,
                    target_value=target.latent_value,
                    donor_value=donor.latent_value,
                    target_entry_clean=target_clean,
                    target_entry_patched=_entry(
                        lens, model, residuals, read_layers, target.latent_token_id
                    ),
                    donor_entry_clean=donor_clean,
                    donor_entry_patched=_entry(
                        lens, model, residuals, read_layers, donor.latent_token_id
                    ),
                )
            )
    return observations


def pool_specificity(
    observations: list[SpecificityObservation], *, seed: int = 0
) -> list[SpecificityResult]:
    """Pool by component, with paired intervals over targets."""
    by_component: dict[str, list[SpecificityObservation]] = {}
    for observation in observations:
        by_component.setdefault(observation.component, []).append(observation)

    results = []
    for component, group in by_component.items():
        results.append(
            SpecificityResult(
                component=component,
                delta_target=paired_bootstrap(
                    np.array([o.target_entry_patched for o in group]),
                    np.array([o.target_entry_clean for o in group]),
                    seed=seed,
                ),
                delta_donor=paired_bootstrap(
                    np.array([o.donor_entry_patched for o in group]),
                    np.array([o.donor_entry_clean for o in group]),
                    seed=seed,
                ),
                n=len(group),
            )
        )
    return sorted(results, key=lambda r: -r.delta_donor.point)
