"""The 2D sweep: patch layer x readout layer.

A one-dimensional sweep cannot separate two things, and conflating them already cost
this
project a wrong conclusion. Read at a **fixed late** layer and a shallow patch is
measured
after many layers of possible overwriting while a deep one is measured after few -- that
grades *survival*, and reporting it as write strength produced an "onset at L39" that
was
really a repair boundary. Read **close above** the patch and every layer is measured at
matched distance, which grades installation.

So readout distance is a second axis. This records the residual at *every* layer above a
patch in one pass: a row decaying with distance was installed then repaired, flat and
high
means retained, flat at zero means never installed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from jlens import JacobianLens
from jlens.hf import HFLensModel

from innerj import console
from innerj.analysis.readout import (
    ConceptScores,
    concept_scores,
    transported_logits,
)
from innerj.analysis.stats import Estimate, paired_bootstrap
from innerj.patch import Component, capture, run_patched
from innerj.positions import READ_AT
from innerj.tasks.base import Record


@dataclass
class Cell:
    """One (component, readout layer) cell, pooled over pairs.

    ``delta_donor`` is the ``R_z`` every published number used and it **saturates**
    inside this
    band, so the ``_logrank`` and ``_mz`` companions are what decide whether a window
    boundary is
    a property of the model or of the measure.
    """

    component: str
    kind: str
    patch_layer: int
    read_layer: int
    distance: int
    delta_donor: Estimate
    delta_target: Estimate
    delta_donor_logrank: Estimate
    delta_target_logrank: Estimate
    delta_donor_mz: Estimate
    delta_target_mz: Estimate
    n: int
    #: Which target instances this cell pooled over, in the order they were
    #: measured. Without it a pooled cell cannot be re-analysed by instance, so a
    #: discovery/confirmation split is impossible after the fact -- which is exactly
    #: what happened to the head-level sweep, whose split had to be reported as
    #: unavailable rather than computed.
    instances: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@torch.no_grad()
def sweep(
    model: HFLensModel,
    lens: JacobianLens,
    pairs: list[tuple[Record, Record]],
    components: list[Component],
    *,
    max_read_layer: int | None = None,
    positions: list[int] | None = None,
    max_seq_len: int = 512,
) -> tuple[list[Cell], list[dict]]:
    """For every component, the donor's concept rank at every layer above it.

    Returns the pooled cells **and** the per-observation rows. The rows are why: a
    pooled cell
    cannot be re-split, and when the tracking sweep went from 50 to 140 pairs, whether a
    head's
    emergence came from the added pairs or the donor reassignment was unanswerable from
    the
    results file. Every factor is carried per row, including ``positions``.
    """
    positions = positions or [-1]
    max_read = model.n_layers - 1 if max_read_layer is None else max_read_layer
    fitted = set(lens.source_layers)

    # {(component, read_layer): [12-tuple]} -- see `scores` for the layout.
    cells: dict[tuple[str, int], list[tuple[float, ...]]] = {}
    # Parallel to `cells`, so a pooled estimate stays traceable to its instances.
    cell_instances: dict[tuple[str, int], list[str]] = {}
    # One row per (pair, component, read_layer), written beside the summary.
    observations: list[dict] = []
    meta: dict[str, Component] = {str(c): c for c in components}

    def scores(
        residuals: dict[int, torch.Tensor],
        layers: list[int],
        token_id: int,
        rivals: list[int],
    ) -> dict[int, ConceptScores]:
        """All three concept measures at every readout layer, from one pass.

                ``R_z`` alone was recorded originally, so the grid could not be
                re-analysed under a
                non-saturating measure without repeating the sweep on GPU. The rank is
                shared, so the
                extra columns are nearly free.
        """
        return {
            layer: concept_scores(
                transported_logits(model, lens, residuals, layer, row=-1),
                token_id,
                rivals,
            )
            for layer in layers
        }

    for donor, target in console.track(pairs, "sweeping"):
        if donor.latent_token_id == target.latent_token_id:
            raise ValueError(
                f"{donor.id} and {target.id} share a latent value; this sweep needs "
                f"them to differ"
            )
        # The instance's candidate concept set. The contrastive margin for a
        # concept is measured against the *other* candidates, so each of the two
        # concepts tracked here gets its own rival set -- matched by construction,
        # and symmetric between donor and target rather than privileging either.
        concepts = {target.latent_token_id, *target.control_token_ids}
        donor_rivals = sorted(concepts - {donor.latent_token_id})
        target_rivals = sorted(concepts - {target.latent_token_id})
        if not donor_rivals or not target_rivals:
            raise ValueError(
                f"{target.id} has no rival concepts for the contrastive margin; the "
                f"candidate set is {sorted(concepts)}"
            )
        read_all = sorted(
            layer
            for layer in fitted
            if min(c.layer for c in components) < layer <= max_read
        )
        donor_acts = capture(
            model, donor.prompt, components, positions=positions,
            max_seq_len=max_seq_len,
        )
        _, clean = run_patched(
            model, target.prompt, {}, positions=positions,
            read_positions=READ_AT,
            max_seq_len=max_seq_len, record_layers=read_all,
        )
        donor_clean = scores(clean, read_all, donor.latent_token_id, donor_rivals)
        target_clean = scores(clean, read_all, target.latent_token_id, target_rivals)

        for component in components:
            above = [layer for layer in read_all if layer > component.layer]
            if not above:
                continue
            _, patched = run_patched(
                model, target.prompt, {component: donor_acts[component]},
                positions=positions, read_positions=READ_AT,
                max_seq_len=max_seq_len, record_layers=above,
            )
            donor_patched = scores(patched, above, donor.latent_token_id, donor_rivals)
            target_patched = scores(
                patched, above, target.latent_token_id, target_rivals
            )
            for layer in above:
                dp, dc = donor_patched[layer], donor_clean[layer]
                tp, tc = target_patched[layer], target_clean[layer]
                cell_instances.setdefault((str(component), layer), []).append(
                    target.semantic_instance_id
                )
                cells.setdefault((str(component), layer), []).append(
                    (
                        dp.r_z, dc.r_z, tp.r_z, tc.r_z,
                        dp.neg_log_rank, dc.neg_log_rank,
                        tp.neg_log_rank, tc.neg_log_rank,
                        dp.m_z, dc.m_z, tp.m_z, tc.m_z,
                    )
                )
                observations.append(
                    {
                        "component": str(component),
                        "kind": component.kind,
                        "patch_layer": component.layer,
                        "head": component.head,
                        "read_layer": layer,
                        "distance": layer - component.layer,
                        "positions": list(positions),
                        "semantic_instance_id": target.semantic_instance_id,
                        "target_id": target.id,
                        "donor_id": donor.id,
                        "target_latent_token_id": target.latent_token_id,
                        "donor_latent_token_id": donor.latent_token_id,
                        "donor_patched_r_z": dp.r_z,
                        "donor_clean_r_z": dc.r_z,
                        "target_patched_r_z": tp.r_z,
                        "target_clean_r_z": tc.r_z,
                        "donor_patched_logrank": dp.neg_log_rank,
                        "donor_clean_logrank": dc.neg_log_rank,
                        "target_patched_logrank": tp.neg_log_rank,
                        "target_clean_logrank": tc.neg_log_rank,
                        "donor_patched_m_z": dp.m_z,
                        "donor_clean_m_z": dc.m_z,
                        "target_patched_m_z": tp.m_z,
                        "target_clean_m_z": tc.m_z,
                    }
                )

    out: list[Cell] = []
    for (name, layer), rows in cells.items():
        arr = np.array(rows, dtype=float)
        component = meta[name]
        out.append(
            Cell(
                component=name,
                kind=component.kind,
                patch_layer=component.layer,
                read_layer=layer,
                distance=layer - component.layer,
                delta_donor=paired_bootstrap(arr[:, 0], arr[:, 1]),
                delta_target=paired_bootstrap(arr[:, 2], arr[:, 3]),
                delta_donor_logrank=paired_bootstrap(arr[:, 4], arr[:, 5]),
                delta_target_logrank=paired_bootstrap(arr[:, 6], arr[:, 7]),
                delta_donor_mz=paired_bootstrap(arr[:, 8], arr[:, 9]),
                delta_target_mz=paired_bootstrap(arr[:, 10], arr[:, 11]),
                n=len(rows),
                instances=cell_instances[(name, layer)],
            )
        )
    ordered = sorted(out, key=lambda c: (c.patch_layer, c.kind, c.read_layer))
    return ordered, observations


def at_distance(cells: list[Cell], distance: int, *, tolerance: int = 1) -> list[Cell]:
    """Cells read roughly ``distance`` layers above their patch -- what makes patch
    layers
        comparable, which a fixed readout layer does not.
    """
    return [c for c in cells if abs(c.distance - distance) <= tolerance]


def decay_profile(cells: list[Cell], component: str) -> list[tuple[int, float]]:
    """``(distance, delta_donor)`` for one component -- installed then repaired?"""
    rows = [
        (c.distance, c.delta_donor.point)
        for c in cells
        if c.component == component
    ]
    return sorted(rows)
