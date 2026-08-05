"""Stage 2: which components carry the difference between automatic and flexible?

For a matched pair of runs over the same semantic instance, take one component's
output from the *flexible* run and substitute it into the *automatic* run. A
component that carries the write decision should move workspace entry toward the
flexible level; most components should do nothing.

Both directions are run, because a real write component has to show both:

* ``flex_to_auto`` -- injecting it into the automatic run *raises* entry;
* ``auto_to_flex`` -- injecting the automatic value into the flexible run
  *suppresses* entry.

A component that only does the first is as consistent with "any perturbation here
nudges the readout" as with a write mechanism.

**Readout layers sit strictly above every screened component.** Patching at layer
L cannot affect a readout below L, so averaging ``R_z`` over the whole band would
reward shallow components purely for having more readout layers downstream of
them. That would be an artifact of the aggregation, not a fact about the circuit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from jlens import JacobianLens
from jlens.hf import HFLensModel

from innerj import console
from innerj.analysis.readout import concept_scores, forced_choice
from innerj.analysis.stats import Estimate, paired_bootstrap, ratio_with_gap
from innerj.patch import Component, capture, layer_type, run_patched
from innerj.tasks.base import Record


def coarse_components(layers: list[int]) -> list[Component]:
    """One attention block and one MLP per layer: the cheap first pass."""
    return [Component(kind, layer) for layer in layers for kind in ("attn", "mlp")]


def head_components(
    model: HFLensModel, layers: list[int], *, include_mlp: bool = True
) -> list[Component]:
    """Every individual head in ``layers``, plus each layer's MLP.

    Head count is read per layer, not passed in: on a hybrid checkpoint a
    full-attention layer has 24 heads and a linear-attention layer has 48, so one
    global count would either skip heads or index past the projection.
    """
    from innerj.patch import n_attention_heads

    out: list[Component] = []
    for layer in layers:
        n_heads = n_attention_heads(model, layer)
        out.extend(Component("attn", layer, head=h) for h in range(n_heads))
        if include_mlp:
            out.append(Component("mlp", layer))
    return out


def readout_layers(
    screened: list[int], n_layers: int, *, n_readout: int = 6
) -> list[int]:
    """Layers strictly above every screened one, so all components are comparable.

    Raises if the screened range leaves no room: a readout at or below a patched
    layer measures nothing the patch could have caused.
    """
    first = max(screened) + 1
    available = list(range(first, n_layers))
    if len(available) < n_readout:
        raise ValueError(
            f"screening up to L{max(screened)} leaves only {len(available)} layers "
            f"above it, need {n_readout}. Screen shallower or read fewer layers."
        )
    return available[:n_readout]


def _entry(
    lens: JacobianLens,
    model: HFLensModel,
    residuals: dict[int, torch.Tensor],
    layers: list[int],
    gold_token_id: int,
    control_ids: list[int],
) -> tuple[float, float, float]:
    """Mean ``(R_z, -log10(rank), M_z)`` over ``layers`` from recorded residuals.

    Reuses the residuals from the patched forward pass rather than running again,
    which is what makes a 300-component sweep affordable.

    All three metrics come back together because recovering a missing one costs a
    full re-run of the sweep on GPU. Earlier versions returned ``R_z`` alone, and
    the saturation question --- whether the transport window is an artifact of a
    metric that is pinned above 0.999 in 92% of the cells it is read from ---
    could not then be answered from any artifact on disk.
    """
    scores = [
        concept_scores(
            model.unembed(lens.transport(residuals[layer], layer)).float()[0],
            gold_token_id,
            control_ids,
        )
        for layer in layers
    ]
    return (
        float(np.mean([s.r_z for s in scores])),
        float(np.mean([s.neg_log_rank for s in scores])),
        float(np.mean([s.m_z for s in scores])),
    )


@dataclass
class Observation:
    """One component, one instance, one direction."""

    component: str
    kind: str
    layer: int
    head: int | None
    direction: str
    layer_type: str
    instance: str
    entry_clean: float
    entry_patched: float
    entry_reference: float
    margin_clean: float
    margin_patched: float
    #: ``-log10(rank)`` companions to the ``entry_*`` percentile ranks above.
    #: ``entry_*`` saturates in the band this sweep reads from, so a null here
    #: and a null there are different claims.
    logrank_clean: float
    logrank_patched: float
    logrank_reference: float
    #: Contrastive concept margin ``M_z`` against the rival candidates. Graded and
    #: unbounded, so it survives both ceiling and floor.
    mz_clean: float
    mz_patched: float
    mz_reference: float


@dataclass
class ScreenResult:
    """Pooled effect of one component in one direction, over instances."""

    component: str
    kind: str
    layer: int
    head: int | None
    direction: str
    layer_type: str
    delta_entry: Estimate
    delta_margin: Estimate
    #: Same contrast under the non-saturating rank measure and the contrastive
    #: margin. Reported alongside ``delta_entry`` rather than instead of it: a
    #: component significant under one and not the others is a finding about the
    #: metric, and must not be quoted as a bare transport effect.
    delta_logrank: Estimate
    delta_mz: Estimate
    recovery: float
    recovery_gap: float
    n: int

    def to_dict(self) -> dict:
        return asdict(self)


@torch.no_grad()
def screen(
    model: HFLensModel,
    lens: JacobianLens,
    pairs: list[tuple[Record, Record]],
    components: list[Component],
    *,
    read_layers: list[int],
    direction: str = "flex_to_auto",
    max_seq_len: int = 512,
) -> list[Observation]:
    """Patch every component from source into target, for every pair.

    ``pairs`` are ``(flexible, automatic)`` records of the same instance.
    ``direction`` picks which is the donor. One donor capture and one clean run per
    pair, then one forward pass per component.
    """
    if direction not in ("flex_to_auto", "auto_to_flex"):
        raise ValueError(f"unknown direction {direction!r}")

    types = {c.layer: layer_type(model, c.layer) for c in components}
    observations: list[Observation] = []
    for flexible, automatic in console.track(pairs, "screening"):
        donor, target = (
            (flexible, automatic)
            if direction == "flex_to_auto"
            else (automatic, flexible)
        )
        gold = target.latent_token_id
        # Rival concept names, for the contrastive margin. Matched by construction:
        # same category, same frequency band, same part of speech.
        controls = target.control_token_ids
        if donor.latent_token_id != gold:
            raise ValueError(
                f"{donor.id} and {target.id} disagree on the latent token; they are "
                f"not the same semantic instance"
            )

        donor_acts = capture(
            model, donor.prompt, components, max_seq_len=max_seq_len
        )
        clean_logits, clean_res = run_patched(
            model, target.prompt, {}, max_seq_len=max_seq_len, record_layers=read_layers
        )
        _, donor_res = run_patched(
            model, donor.prompt, {}, max_seq_len=max_seq_len, record_layers=read_layers
        )
        entry_clean = _entry(lens, model, clean_res, read_layers, gold, controls)
        entry_reference = _entry(lens, model, donor_res, read_layers, gold, controls)
        _, margin_clean = forced_choice(clean_logits[0], target.candidate_token_ids)

        for component in components:
            logits, residuals = run_patched(
                model,
                target.prompt,
                {component: donor_acts[component]},
                max_seq_len=max_seq_len,
                record_layers=read_layers,
            )
            _, margin = forced_choice(logits[0], target.candidate_token_ids)
            patched = _entry(lens, model, residuals, read_layers, gold, controls)
            observations.append(
                Observation(
                    component=str(component),
                    kind=component.kind,
                    layer=component.layer,
                    head=component.head,
                    direction=direction,
                    layer_type=types[component.layer],
                    instance=target.semantic_instance_id,
                    entry_clean=entry_clean[0],
                    entry_patched=patched[0],
                    entry_reference=entry_reference[0],
                    margin_clean=margin_clean,
                    margin_patched=margin,
                    logrank_clean=entry_clean[1],
                    logrank_patched=patched[1],
                    logrank_reference=entry_reference[1],
                    mz_clean=entry_clean[2],
                    mz_patched=patched[2],
                    mz_reference=entry_reference[2],
                )
            )
    console.detail(f"{len(observations)} observations")
    return observations


def pool(observations: list[Observation], *, seed: int = 0) -> list[ScreenResult]:
    """Pool per-component effects with paired intervals over instances.

    ``recovery`` is the fraction of the automatic-to-flexible entry gap the patch
    closes. Its absolute gap travels with it: when the reference gap is near zero
    the ratio is meaningless and is returned as ``nan`` rather than as a large
    number.
    """
    by_key: dict[tuple[str, str], list[Observation]] = {}
    for observation in observations:
        by_key.setdefault((observation.component, observation.direction), []).append(
            observation
        )

    results: list[ScreenResult] = []
    for (component, direction), group in by_key.items():
        patched = np.array([o.entry_patched for o in group])
        clean = np.array([o.entry_clean for o in group])
        reference = np.array([o.entry_reference for o in group])
        delta_entry = paired_bootstrap(patched, clean, seed=seed)
        delta_margin = paired_bootstrap(
            np.array([o.margin_patched for o in group]),
            np.array([o.margin_clean for o in group]),
            seed=seed,
        )
        delta_logrank = paired_bootstrap(
            np.array([o.logrank_patched for o in group]),
            np.array([o.logrank_clean for o in group]),
            seed=seed,
        )
        delta_mz = paired_bootstrap(
            np.array([o.mz_patched for o in group]),
            np.array([o.mz_clean for o in group]),
            seed=seed,
        )
        recovery, gap = ratio_with_gap(
            float((patched - clean).mean()), float((reference - clean).mean())
        )
        first = group[0]
        results.append(
            ScreenResult(
                component=component,
                kind=first.kind,
                layer=first.layer,
                head=first.head,
                direction=direction,
                layer_type=first.layer_type,
                delta_entry=delta_entry,
                delta_margin=delta_margin,
                delta_logrank=delta_logrank,
                delta_mz=delta_mz,
                recovery=recovery,
                recovery_gap=gap,
                n=len(group),
            )
        )
    return sorted(results, key=lambda r: -abs(r.delta_entry.point))


def survivors(
    results: list[ScreenResult], *, alpha: float = 0.05, metric: str = "delta_entry"
) -> list[ScreenResult]:
    """Components whose effect clears zero, FDR-corrected across the sweep.

    A bootstrap interval per component is not enough when hundreds are tested: the
    top of an uncorrected ranking is mostly noise. The interval is converted to an
    approximate two-sided p-value for the correction, and the interval itself is
    still what gets reported.

    ``metric`` selects which estimate is corrected --- ``delta_entry`` (``R_z``),
    ``delta_logrank`` or ``delta_mz``. Correction is applied *within* a metric,
    because the family being corrected over is the sweep's components, not the
    metrics; running it across all three pooled would treat three views of one
    measurement as three independent tests.
    """
    from scipy.stats import norm

    from innerj.analysis.stats import benjamini_hochberg

    if not results:
        return []
    if not hasattr(results[0], metric):
        raise ValueError(
            f"{metric!r} is not an estimate on ScreenResult; expected one of "
            f"delta_entry, delta_logrank, delta_mz, delta_margin"
        )
    pvalues = []
    for r in results:
        estimate = getattr(r, metric)
        half_width = (estimate.hi - estimate.lo) / 2
        standard_error = max(half_width / 1.959964, 1e-12)
        pvalues.append(2 * norm.sf(abs(estimate.point) / standard_error))
    keep = benjamini_hochberg(np.array(pvalues), alpha=alpha)
    return [r for r, k in zip(results, keep, strict=True) if k]
