"""Stage 2b: is the effect content-specific, or does patching there just help?

Many components move the readout when patched, several *downward*, so "largest positive
effect" does not yet mean "carries the latent variable". Patch from a donor whose latent
value **differs** from the target's and read out two concepts: the **target's** gold
language, which a generic sharpener still raises, and the **donor's**, which only a
component carrying the latent value raises. The predictions are opposite, so one
experiment
decides it.

Without this control the whole screen is compatible with an interpretability illusion
where
a subspace intervention activates a pathway unrelated to the model's mechanism.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from jlens import JacobianLens
from jlens.hf import HFLensModel

from innerj import console
from innerj.analysis.readout import percentile_rank, transported_logits
from innerj.analysis.stats import Estimate, paired_bootstrap
from innerj.patch import Component, capture, run_patched
from innerj.positions import READ_AT, PositionFn, describe, labelled
from innerj.tasks.base import Record


@dataclass
class SpecificityObservation:
    """One (component, target, mismatched donor) trial, read on both concepts."""

    component: str
    #: Which positions were patched. Trap 16: an observations file that omits an
    #: experimental factor cannot be re-analysed by it, and this experiment gained a
    #: position axis (the repair test's span-matched and passage-inclusive arms)
    #: after the class was first written.
    positions: str
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
    positions: str
    delta_target: Estimate
    delta_donor: Estimate
    n: int

    @property
    def asymmetry(self) -> float:
        """How much more the donor's concept rises than the target's own.

                The informative quantity, not the sign pattern: the donor's language is
                **absent
                from the target's context** while the target's own is present, so
                nonspecific
                sharpening should favour the *target*. ``inf`` when the target's does
                not rise.
        """
        if self.delta_target.point <= 0:
            return float("inf") if self.delta_donor.point > 0 else 0.0
        return self.delta_donor.point / self.delta_target.point

    def verdict(self, *, min_asymmetry: float = 2.0) -> str:
        """Classify the pattern; ``min_asymmetry`` is a reporting threshold, not a
        test."""
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
    """Pair each automatic target with a flexible donor of a *different* language,
        matched on everything else, so an outcome difference cannot come from length or
        template.
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
    """Mean ``R_z`` of ``token_id`` over ``layers``, at the answer position.

    ``row=-1``, not ``[0]``: identical for a one-position patch and not once a span is
    patched
    (trap 7). ``R_z`` alone, since this experiment has no matched control set for
    ``M_z``.
    """
    return float(
        np.mean([
            percentile_rank(
                transported_logits(model, lens, residuals, layer, row=-1), token_id
            )
            for layer in layers
        ])
    )


@torch.no_grad()
def specificity(
    model: HFLensModel,
    lens: JacobianLens,
    pairs: list[tuple[Record, Record]],
    components: list[Component],
    *,
    read_layers: list[int],
    positions: PositionFn | None = None,
    max_seq_len: int = 512,
) -> list[SpecificityObservation]:
    """Run the mismatched-donor patch and read out both concepts; ``positions`` is
        resolved per record and defaults to the final token, as the published results
        used.
    """
    positions = positions or labelled(lambda _model, _record: [-1], "query:last1")
    observations: list[SpecificityObservation] = []
    position_label = describe(positions)
    for donor, target in console.track(pairs, "processing"):
        if donor.latent_token_id == target.latent_token_id:
            raise ValueError(
                f"{donor.id} and {target.id} share a latent value; this control "
                f"requires them to differ"
            )
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
        _, clean = run_patched(
            model,
            target.prompt,
            {},
            positions=target_positions,
            read_positions=READ_AT,
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
                positions=target_positions,
                read_positions=READ_AT,
                max_seq_len=max_seq_len,
                record_layers=read_layers,
            )
            observations.append(
                SpecificityObservation(
                    component=str(component),
                    positions=position_label,
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
    """Pool by component **and position mode**, paired over targets.

    Pooling two position modes into one row is the collapse that made a null read as
    significantly positive elsewhere; here it would halve the difference being measured.
    """
    by_cell: dict[tuple[str, str], list[SpecificityObservation]] = {}
    for observation in observations:
        by_cell.setdefault(
            (observation.component, observation.positions), []
        ).append(observation)

    results = []
    for (component, position_label), group in by_cell.items():
        results.append(
            SpecificityResult(
                component=component,
                positions=position_label,
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
