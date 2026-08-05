"""Stage 6: is the behavioural effect mediated by J-space?

This is the experiment that can undercut the paper, which is why it is worth running.

Patching the residual stream at L39 from a mismatched donor changes the answer to the
predicted counterfactual ~44% of the time. But a residual patch carries *everything*
the donor computed, not only the latent value. So the behavioural effect might be
carried by the J-space representation of the donor's language --- the thing this paper
is about --- or by some other component of the donor's stream that happens to travel
with it. A readout showing the language arrived does not settle it: §9 already found a
component that moves the readout and changes no answers.

The test removes the candidate mediator and asks whether the effect survives:

* **full** --- patch the donor's residual unchanged. The reference effect.
* **ablated** --- patch it with the donor language's J-direction **projected out**, so
  everything else the donor computed is preserved and only that direction is removed.
* **random** --- patch it with a *random* direction of matched norm projected out. The
  control that separates "removing this direction matters" from "removing any
  direction of this size matters".

If the effect falls under **ablated** but not **random**, it is J-mediated. If it
survives both, the behavioural result rides on something other than the J-space
representation and the framing has to change.

**Which direction, exactly.** The readout is ``lambda_z(h) = W_U[z] . N(J_l h)``
with ``N`` the model's final norm, so the direction that raises token ``z``'s lens
logit is the *gradient*

    grad_h lambda_z = J_l^T . DN(J_l h)^T . W_U[z].

``J_l^T W_U[z]`` alone drops ``DN``, and is therefore the **static** pre-normaliser
lens vector rather than the exact local logit-gradient direction. For RMSNorm the
two differ in two specific ways, both implemented and both testable
(:func:`readout_direction`): the unembedding row is weighted by the norm's gain,
``u = W_U[z] * g``, and the component of ``u`` along the activation itself is
removed. Up to a positive scale the static vector is the gradient with the radial
component left in and the gain omitted --- close, but not the same vector, and §7's
causal claim is about targeting the concept direction.

Three derivations are available, which is the robustness experiment the difference
deserves rather than a correction to be conceded:

* ``static`` --- ``J^T W_U[z]``, what every published number here used;
* ``gradient`` --- the exact normalised-logit gradient above;
* ``margin`` --- the exact gradient of the *contrastive* score
  ``lambda_z - mean_{c != z} lambda_c``, which is what a concept score should mean.
  Because the readout is linear in the post-norm vector, this is the same formula
  with ``W_U[z]`` replaced by ``W_U[z] - mean_{c != z} W_U[c]``.

**And which control.** An isotropic random direction of matched *vector* norm is an
easy control: it removes far less of the activation than the gold direction does, so
the comparison partly measures how much was removed. The controls here are
therefore graded by what they hold fixed --- see :data:`CONTROLS`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from jlens import JacobianLens
from jlens.hf import HFLensModel

from innerj import console
from innerj.analysis.readout import forced_choice
from innerj.analysis.stats import Estimate, paired_bootstrap
from innerj.patch import Component, capture, run_patched
from innerj.tasks.base import Record

#: Legacy three-mode labels, kept because published artifacts carry them.
MODES = ("full", "ablated", "random")

#: How the concept direction is derived. See the module docstring.
DERIVATIONS = ("static", "gradient", "margin")

#: What gets projected out, ordered by how much work the control does.
#:
#: * ``gold`` --- the donor concept's own direction. The candidate mediator.
#: * ``wrong`` --- another concept's direction, drawn from the same set of language
#:   directions, so it is matched on everything except identity. The sharpest null:
#:   the 20 concept directions have mean pairwise cosine 0.50, so if the effect
#:   comes from removing "a broadly language-aligned high-norm direction" rather
#:   than from removing *this* language, ``wrong`` reproduces it.
#: * ``orthogonal`` --- the gold direction with its projection onto the span of the
#:   other concept directions removed, i.e. the part of it that is *only* this
#:   concept. Removing it strips this language without stripping 65--73% of the
#:   others along with it.
#: * ``norm_matched`` --- a random direction, but the removal is rescaled so it
#:   takes away the **same activation norm** the gold direction does. This is the
#:   control that separates "what was removed" from "how much was removed", which
#:   a unit-vector-matched random direction cannot.
#: * ``random`` --- isotropic unit direction, the original control. Kept for
#:   comparability with the published numbers, not because it is sufficient.
#: * ``wrong_all`` --- every concept in the family's vocabulary except the donor's,
#:   one row each, so the wrong-concept null is a *distribution* rather than a single
#:   draw. Needs ``concept_ids``.
CONTROLS = ("gold", "wrong", "wrong_all", "orthogonal", "norm_matched", "random")

#: Where the direction is removed from.
#:
#: * ``absolute`` --- from the donor's residual, as published.
#: * ``delta`` --- from the donor-minus-target difference, writing
#:   ``h_t + (I - P_v)(h_d - h_t)``. This preserves everything donor and target
#:   share and excises only the component the substitution introduces, so it is the
#:   more surgical test of the same hypothesis.
PROJECTIONS = ("absolute", "delta")


def _final_norm_gain(model: HFLensModel) -> tuple[torch.Tensor, float]:
    """The final norm's gain vector and epsilon, for the norm Jacobian.

    Raises rather than guessing: a norm without a plain gain (a bias term, or a
    ``layer_scalar``) makes the analytic Jacobian below wrong, and silently
    returning ones would produce a plausible direction that is not the gradient.
    """
    norm = model._final_norm
    weight = getattr(norm, "weight", None)
    if weight is None:
        raise ValueError(
            f"{type(norm).__name__} has no `weight`; the analytic norm Jacobian "
            f"here assumes RMSNorm with a gain and no bias"
        )
    if getattr(norm, "bias", None) is not None:
        raise ValueError(
            f"{type(norm).__name__} has a bias, so it is not the scale-free norm "
            f"this derivation and the gauge argument both assume"
        )
    eps = getattr(norm, "variance_epsilon", None)
    if eps is None:
        eps = getattr(norm, "eps", 1e-6)
    return weight.detach().float(), float(eps)


def _logit_weight(
    model: HFLensModel, token_id: int, rival_ids: list[int] | None
) -> torch.Tensor:
    """``W_U[z]``, or the contrastive ``W_U[z] - mean_{c != z} W_U[c]``.

    The readout is linear in the post-norm vector, so a contrastive score over
    concepts collapses to a single weight vector before the norm is differentiated.
    That is why the margin gradient costs no more than the logit gradient.
    """
    unembed = model._lm_head.weight  # [vocab, d_model]
    w = unembed[token_id].detach().float()
    if not rival_ids:
        return w
    return w - unembed[rival_ids].detach().float().mean(0)


def readout_direction(
    model: HFLensModel,
    lens: JacobianLens,
    layer: int,
    token_id: int,
    *,
    kind: str = "static",
    activation: torch.Tensor | None = None,
    rival_ids: list[int] | None = None,
) -> torch.Tensor:
    """Unit residual direction at ``layer`` that raises ``token_id``'s lens score.

    ``kind`` selects the derivation --- see :data:`DERIVATIONS` and the module
    docstring. ``static`` needs no activation and returns ``J^T W_U[z]``; the other
    two differentiate through the final norm and are therefore *activation
    dependent*, so ``activation`` (the residual at the patch site, shape ``[d]`` or
    ``[n_positions, d]``) is required. With ``[n_positions, d]`` the result is one
    unit direction per position, because the gradient genuinely differs between
    them.

    All arithmetic is fp32. In bf16 the norm Jacobian's radial subtraction loses
    most of its precision, which is the same reason the gauge check upstream runs
    in fp32.
    """
    if kind not in DERIVATIONS:
        raise ValueError(f"unknown derivation {kind!r}; expected one of {DERIVATIONS}")
    if kind == "margin" and not rival_ids:
        raise ValueError(
            "the margin derivation needs rival concept ids; without them it is the "
            "plain logit gradient under a different name"
        )
    jacobian = lens.jacobians[layer].float()
    weight = _logit_weight(
        model, token_id, rival_ids if kind == "margin" else None
    ).to(jacobian.device)

    if kind == "static":
        direction = jacobian.T @ weight
    else:
        if activation is None:
            raise ValueError(
                f"the {kind!r} derivation is activation-dependent; pass the residual "
                f"at the patch site"
            )
        gain, eps = _final_norm_gain(model)
        gain = gain.to(jacobian.device)
        h = activation.detach().float().to(jacobian.device)
        squeeze = h.ndim == 1
        h = h.unsqueeze(0) if squeeze else h
        x = h @ jacobian.T                                   # [n, d], the transported
        d_model = x.shape[-1]
        mean_square = x.pow(2).mean(-1, keepdim=True) + eps
        r = mean_square.sqrt()                               # [n, 1]
        u = weight * gain                                    # gain-weighted row
        # DN(x)^T u = (1/r)[u - x <u,x> / (d r^2)] for RMSNorm.
        radial = (x * u).sum(-1, keepdim=True) / (d_model * mean_square)
        through_norm = (u.unsqueeze(0) - x * radial) / r      # [n, d]
        direction = through_norm @ jacobian                   # J^T applied on the right
        direction = direction.squeeze(0) if squeeze else direction

    norms = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
    if bool((norms == 0).any()):
        raise ValueError(
            f"the {kind!r} direction for token {token_id} at L{layer} is the zero "
            f"vector; it cannot be projected out"
        )
    return direction / norms


def j_direction(
    model: HFLensModel, lens: JacobianLens, layer: int, token_id: int
) -> torch.Tensor:
    """The static direction, as every published number here used.

    Kept as a name of its own because it is what the paper's §7 quotes, and
    because renaming it would make the artifacts harder to trace. Prefer
    :func:`readout_direction` for new work, which says which derivation it means.
    """
    return readout_direction(model, lens, layer, token_id, kind="static")


def orthogonalise(
    direction: torch.Tensor, others: torch.Tensor
) -> torch.Tensor:
    """``direction`` with its projection onto the span of ``others`` removed.

    ``others`` is ``[k, d]``. The concept directions here have mean pairwise cosine
    0.50, so removing the gold one also strips most of every rival; what survives
    this is the part that belongs to this concept alone.

    Raises if almost nothing survives: a residual of near-zero norm means the
    concept has no independent direction at this layer, and normalising it would
    turn numerical noise into a confident-looking intervention.
    """
    if others.ndim != 2:
        raise ValueError(f"expected others as [k, d], got {tuple(others.shape)}")
    basis = others.float().to(direction.device)
    # Least-squares coefficients against a possibly ill-conditioned basis.
    solution = torch.linalg.lstsq(basis.T, direction.float().unsqueeze(-1)).solution
    residual = direction.float() - (basis.T @ solution).squeeze(-1)
    surviving = float(torch.linalg.vector_norm(residual))
    if surviving < 1e-3:
        raise ValueError(
            f"orthogonalising against {basis.shape[0]} rival directions leaves norm "
            f"{surviving:.2e}; this concept has no direction independent of the "
            f"others at this layer, so the control is not measurable"
        )
    return residual / surviving


def remove(
    value: torch.Tensor, direction: torch.Tensor, *, scale: float = 1.0
) -> torch.Tensor:
    """Subtract ``scale`` times ``value``'s component along ``direction``.

    ``scale=1`` is a full projection; intermediate values give the dose--response
    curve, which is what shows an effect grows with how much of the direction is
    removed rather than switching on at full removal.

    ``direction`` may be one unit vector or one per row of ``value``.
    """
    d = direction.to(value.device, value.dtype)
    if d.ndim == 2 and value.ndim == 2:
        coefficient = (value * d).sum(-1, keepdim=True)
        return value - scale * coefficient * d
    coefficient = value @ d
    if value.ndim == 2:
        return value - scale * torch.outer(coefficient, d)
    return value - scale * coefficient * d


def removed_norm(value: torch.Tensor, direction: torch.Tensor) -> float:
    """How much of ``value``'s norm a full projection along ``direction`` removes.

    The quantity the ``norm_matched`` control equalises. Matching *vector* norms
    between a gold and a random direction does not match this, because the
    activation is not isotropic: the gold direction is aligned with the stream and
    a random one mostly is not.
    """
    d = direction.to(value.device, value.dtype)
    if d.ndim == 2 and value.ndim == 2:
        coefficient = (value * d).sum(-1)
    else:
        coefficient = value @ d
    return float(torch.linalg.vector_norm(coefficient.float()))


def project_out(value: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Remove ``direction`` (unit) from ``value``, leaving everything else intact."""
    return remove(value, direction, scale=1.0)


@dataclass
class MediationObservation:
    component: str
    #: Which positions were patched, e.g. ``"query:last12"``. Recorded per row
    #: because an observations file that omits an experimental factor cannot be
    #: re-analysed by it -- splitting on row order is the only recovery, and we
    #: have had to do that once already.
    positions: str
    mode: str
    target_instance: str
    donor_instance: str
    donor_value: str
    gold_symbol: str
    donor_symbol: str
    answer: str
    is_gold: bool
    is_donor_symbol: bool
    is_other: bool
    n_other: int
    #: Which control direction was removed (:data:`CONTROLS`), how it was derived
    #: (:data:`DERIVATIONS`), and what fraction of it was taken out. All three are
    #: experimental factors, so all three are recorded per row rather than encoded
    #: in the ``mode`` string alone.
    control: str = "gold"
    derivation: str = "static"
    dose: float = 1.0
    #: How much activation norm the removal actually took away. The quantity that
    #: makes a random-direction control interpretable: without it, a null control
    #: cannot be distinguished from a control that removed almost nothing.
    removed_norm: float = float("nan")
    #: Norm of the donor activation being modified, so ``removed_norm`` can be read
    #: as a fraction. A removal of 0.4 means nothing without this.
    activation_norm: float = float("nan")
    #: The rival concept token whose direction was removed under ``control="wrong"``
    #: or ``"wrong_all"``. A token id rather than a name because the rival is drawn
    #: from the instance's control tokens, where only ids are carried.
    wrong_token_id: int | None = None
    #: ``absolute`` or ``delta`` --- see :data:`PROJECTIONS`.
    projection: str = "absolute"


@dataclass
class MediationResult:
    component: str
    #: Which positions were patched, e.g. ``"query:last12"``. See the note on
    #: :class:`MediationObservation`.
    positions: str
    mode: str
    donor_symbol_rate: float
    other_symbol_rate: float
    accuracy: float
    delta_vs_other: Estimate
    n: int
    #: The factors this cell varied. Carried on the result as well as the
    #: observation so a pooled artifact can be re-analysed by them without
    #: re-parsing the composite ``mode`` label.
    control: str = "gold"
    derivation: str = "static"
    dose: float = 1.0
    projection: str = "absolute"
    #: Mean activation norm actually removed. A control that removed almost
    #: nothing and a control that removed a lot and did nothing are different
    #: findings, and only this column tells them apart.
    mean_removed_norm: float = float("nan")

    def to_dict(self) -> dict:
        return asdict(self)


@torch.no_grad()
def mediate(
    model: HFLensModel,
    lens: JacobianLens,
    pairs: list[tuple[Record, Record, str, str]],
    component: Component,
    *,
    positions: list[int],
    position_label: str,
    seed: int = 0,
    max_seq_len: int = 512,
    derivations: tuple[str, ...] = ("static",),
    controls: tuple[str, ...] = ("gold", "random"),
    doses: tuple[float, ...] = (1.0,),
    projections: tuple[str, ...] = ("absolute",),
    concept_ids: list[int] | None = None,
) -> list[MediationObservation]:
    """Run the ``full`` reference plus every (derivation, control, dose) cell.

    ``full`` is measured once per pair --- it does not depend on any of the three
    axes --- and is labelled ``mode="full"`` so the published artifacts stay
    comparable. Every other row carries its factors explicitly.

    The number of forward passes per pair is
    ``1 + len(derivations) * len(controls) * len(doses)``, so widen one axis at a
    time rather than taking the full product by default.
    """
    for name, values, allowed in (
        ("derivations", derivations, DERIVATIONS),
        ("controls", controls, CONTROLS),
        ("projections", projections, PROJECTIONS),
    ):
        unknown = set(values) - set(allowed)
        if unknown:
            raise ValueError(f"unknown {name}: {sorted(unknown)}; expected {allowed}")
    if any(not 0.0 <= d <= 1.0 for d in doses):
        raise ValueError(f"doses are fractions of the direction to remove: {doses}")
    if "wrong_all" in controls and not concept_ids:
        raise ValueError(
            "the 'wrong_all' control sweeps the family's whole concept vocabulary and "
            "needs concept_ids; without them it would silently fall back to the three "
            "rivals in one instance's candidate set, which is the weaker control it "
            "exists to replace"
        )

    generator = torch.Generator(device="cpu").manual_seed(seed)
    observations: list[MediationObservation] = []

    for donor, target, gold_symbol, donor_symbol in console.track(
        pairs, "mediating"
    ):
        donor_act = capture(
            model, donor.prompt, [component], positions=positions,
            max_seq_len=max_seq_len,
        )[component]
        reference = donor_act.float()

        answer_of = dict(
            zip(target.candidate_token_ids, target.candidate_answers, strict=True)
        )
        others = set(target.candidate_answers) - {gold_symbol, donor_symbol}

        # The rival concepts, for the wrong-concept and orthogonal controls and for
        # the contrastive margin. Drawn from the instance's own candidate set, so
        # they are matched on category and frequency band by construction.
        rival_ids = [
            i
            for i in [target.latent_token_id, *target.control_token_ids]
            if i != donor.latent_token_id
        ]
        # One rival per pair, drawn with the run's generator so the wrong-concept
        # control is not always the same language across the sweep.
        wrong_id = (
            rival_ids[int(torch.randint(len(rival_ids), (1,), generator=generator))]
            if rival_ids
            else None
        )

        # Bound explicitly rather than captured: these are loop variables, and a
        # closure over them is the kind of hazard that bites the moment anyone
        # defers the call.
        shared = dict(
            component=str(component),
            positions=position_label,
            target_instance=target.semantic_instance_id,
            donor_instance=donor.semantic_instance_id,
            donor_value=donor.latent_value,
            gold_symbol=gold_symbol,
            donor_symbol=donor_symbol,
            n_other=len(others),
        )

        def record(
            answer: str,
            *,
            _shared: dict = shared,
            _others: set = others,
            _gold: str = gold_symbol,
            _donor: str = donor_symbol,
            **factors,
        ) -> None:
            observations.append(
                MediationObservation(
                    answer=answer,
                    is_gold=answer == _gold,
                    is_donor_symbol=answer == _donor,
                    is_other=answer in _others,
                    **_shared,
                    **factors,
                )
            )

        def run(
            value: torch.Tensor,
            *,
            _record: Record = target,
            _answers: dict = answer_of,
        ) -> str:
            logits, _ = run_patched(
                model, _record.prompt, {component: value},
                positions=positions, max_seq_len=max_seq_len,
            )
            winner, _ = forced_choice(logits[-1], _record.candidate_token_ids)
            return _answers[winner]

        # The reference: the donor's residual unchanged.
        activation_norm = float(torch.linalg.vector_norm(reference))
        record(run(donor_act), mode="full", control="none", derivation="none",
               dose=0.0, removed_norm=0.0, activation_norm=activation_norm)

        # The target's own activation at the same site, for the delta projection.
        # Captured only when needed: it costs one forward pass per pair.
        target_act = None
        if "delta" in projections:
            target_act = capture(
                model, target.prompt, [component], positions=positions,
                max_seq_len=max_seq_len,
            )[component].float()

        for derivation in derivations:
            gold_direction = readout_direction(
                model, lens, component.layer, donor.latent_token_id,
                kind=derivation, activation=reference, rival_ids=rival_ids,
            )
            gold_removed = removed_norm(reference, gold_direction)

            for control in controls:
                # One control can expand into several rows: `wrong_all` sweeps the
                # whole concept vocabulary so the null is a distribution rather than
                # a single draw.
                if control == "wrong_all":
                    wrong_set = [i for i in concept_ids if i != donor.latent_token_id]
                elif control == "wrong":
                    if wrong_id is None:
                        continue
                    wrong_set = [wrong_id]
                else:
                    wrong_set = [None]

                for used_wrong in wrong_set:
                    if control in ("wrong", "wrong_all"):
                        direction = readout_direction(
                            model, lens, component.layer, used_wrong,
                            kind=derivation, activation=reference,
                            rival_ids=rival_ids,
                        )
                    elif control == "orthogonal":
                        rivals = torch.stack([
                            readout_direction(
                                model, lens, component.layer, i, kind="static"
                            )
                            for i in rival_ids
                        ])
                        base = gold_direction
                        # A per-position gradient direction has no single vector to
                        # orthogonalise; use the mean direction, renormalised.
                        if base.ndim == 2:
                            base = base.mean(0)
                            base = base / torch.linalg.vector_norm(base)
                        direction = orthogonalise(base, rivals)
                    elif control in ("random", "norm_matched"):
                        direction = torch.randn(
                            gold_direction.shape[-1], generator=generator
                        ).to(gold_direction.device)
                        direction = direction / torch.linalg.vector_norm(direction)
                    else:
                        direction = gold_direction

                    # Rescale so the removal takes the same activation norm as gold.
                    # Without this the random control removes 6.7x less and the
                    # comparison partly measures how much was removed, not what.
                    scale = 1.0
                    if control == "norm_matched":
                        this = removed_norm(reference, direction)
                        if this <= 0:
                            continue
                        scale = gold_removed / this

                    for projection in projections:
                        # `absolute` removes the direction from the donor's residual.
                        # `delta` removes it only from what the substitution
                        # introduces, h_t + (I - P_v)(h_d - h_t), preserving
                        # everything donor and target already share.
                        base_value = (
                            reference if projection == "absolute"
                            else reference - target_act
                        )
                        for dose in doses:
                            stripped = remove(
                                base_value, direction, scale=scale * dose
                            )
                            value = (
                                stripped if projection == "absolute"
                                else target_act + stripped
                            )
                            record(
                                run(value),
                                mode=(
                                    f"{control}:{derivation}:{projection}"
                                    f"@{dose:.2f}"
                                ),
                                control=control,
                                derivation=derivation,
                                dose=dose,
                                projection=projection,
                                removed_norm=float(
                                    torch.linalg.vector_norm(reference - value)
                                ),
                                activation_norm=activation_norm,
                                wrong_token_id=used_wrong,
                            )
    return observations


def pool(
    observations: list[MediationObservation], *, seed: int = 0
) -> list[MediationResult]:
    """Pool per (component, positions, mode), distractor-controlled as everywhere.

    ``positions`` is part of the key. Pooling two position modes together is trap
    16: on the counterfactual artifact it made a null at passage positions look
    significantly positive.
    """
    # The wrong concept is part of the key: `wrong_all` emits one row per rival and
    # pooling them together would average a distribution into a single number, which
    # is exactly the weaker control it replaces.
    grouped: dict[tuple, list[MediationObservation]] = {}
    for o in observations:
        grouped.setdefault(
            (o.component, o.positions, o.mode, o.wrong_token_id), []
        ).append(o)

    results = []
    for (component, positions, mode, _wrong), group in grouped.items():
        donor = np.array([o.is_donor_symbol for o in group], dtype=float)
        other = np.array(
            [o.is_other / max(o.n_other, 1) for o in group], dtype=float
        )
        results.append(
            MediationResult(
                component=component,
                positions=positions,
                mode=mode,
                donor_symbol_rate=float(donor.mean()),
                other_symbol_rate=float(other.mean()),
                accuracy=float(np.mean([o.is_gold for o in group])),
                delta_vs_other=paired_bootstrap(donor, other, seed=seed),
                n=len(group),
                control=group[0].control,
                derivation=group[0].derivation,
                dose=group[0].dose,
                projection=group[0].projection,
                mean_removed_norm=float(
                    np.mean([o.removed_norm for o in group])
                ),
            )
        )
    # "full" first, then by control, derivation and dose. The mode string is a
    # composite label now, so it cannot carry the ordering by itself.
    control_order = {c: i for i, c in enumerate(("none", *CONTROLS))}
    return sorted(
        results,
        key=lambda r: (
            r.component,
            r.positions,
            control_order.get(r.control, len(control_order)),
            r.derivation,
            r.projection,
            r.dose,
            r.mode,
        ),
    )


def wrong_concept_spread(
    results: list[MediationResult], *, component: str, projection: str = "absolute"
) -> dict:
    """Summarise the ``wrong_all`` null across the concept vocabulary.

    One pooled row per rival concept, so this reports how the *worst* wrong concept
    does, not just the average. A gold effect that the best rival also reproduces
    would not be concept-specific, and only the maximum can say that.
    """
    rows = [
        r for r in results
        if r.component == component
        and r.control == "wrong_all"
        and r.projection == projection
        and r.dose == 1.0
    ]
    if not rows:
        return {}
    rates = np.array([r.donor_symbol_rate for r in rows])
    reference = next(
        (r.donor_symbol_rate for r in results
         if r.component == component and r.control == "none"),
        float("nan"),
    )
    losses = reference - rates
    return {
        "n_rivals": len(rows),
        "reference_rate": float(reference),
        "mean_loss": float(losses.mean()),
        "max_loss": float(losses.max()),
        "min_loss": float(losses.min()),
        "sd_loss": float(losses.std(ddof=1)) if len(rows) > 1 else 0.0,
    }


def verdict(
    observations: list[MediationObservation],
    *,
    seed: int = 0,
    derivation: str = "static",
    projection: str = "absolute",
) -> dict[str, str]:
    """Per component: is the behavioural effect carried by the concept direction?

    The comparison is **paired per instance against a control direction**, not
    against a fixed fraction of the effect. An earlier version thresholded the
    proportional loss at 50% and therefore labelled a component with a 42% loss
    "NOT J-MEDIATED" while its matched random control cost exactly 0% ---
    substantively the same finding as one with a 56% loss. That is the same mistake
    as comparing an answer rate to a constant instead of to chance, which this
    project has now made three times.

    **Which control decides it.** An isotropic random direction is the weakest of
    them: it removes far less of the activation than the gold direction does, so
    passing against it partly measures how much was removed. The verdict is
    therefore taken against the *strongest control present*, in the order
    ``wrong`` > ``norm_matched`` > ``random``, and it names which one it used. A
    result that clears ``random`` and fails ``wrong`` is reported as failing.

    Only full-dose rows enter the verdict; intermediate doses are the dose--response
    curve, read separately.
    """
    def key(o: MediationObservation) -> str | None:
        if o.control == "none":
            return "full"
        # One derivation at a time. Without this filter several derivations of the
        # same control collapse onto one dict key and the last one written silently
        # becomes the verdict.
        if o.derivation != derivation or o.dose != 1.0:
            return None
        if o.control != "none" and o.projection != projection:
            return None
        # `wrong_all` is deliberately excluded: it emits one row per rival concept,
        # and every sensible way of collapsing those to a single per-instance number
        # is either a selected statistic (the worst rival) or hides the spread (the
        # mean). Its value is the distribution, which `pool` keeps and
        # :func:`wrong_concept_spread` summarises.
        if o.control == "wrong_all":
            return None
        return o.control

    grouped: dict[str, dict[str, dict[str, MediationObservation]]] = {}
    for o in observations:
        label = key(o)
        if label is None:
            continue
        grouped.setdefault(o.component, {}).setdefault(o.target_instance, {})[
            label
        ] = o

    # Strongest first: a wrong-concept null is the sharpest evidence available,
    # norm-matched is next, and an isotropic random direction is the weakest.
    PREFERENCE = ("wrong", "norm_matched", "random")
    out: dict[str, str] = {}
    for component, instances in grouped.items():
        available = set.intersection(*(set(v) for v in instances.values()))
        if "full" not in available or "gold" not in available:
            continue
        control = next((c for c in PREFERENCE if c in available), None)
        rows = [
            v for v in instances.values()
            if {"full", "gold"} <= set(v) and (control is None or control in v)
        ]
        if not rows:
            continue
        full = np.array([r["full"].is_donor_symbol for r in rows], dtype=float)
        gold = np.array([r["gold"].is_donor_symbol for r in rows], dtype=float)
        loss_j = paired_bootstrap(full, gold, seed=seed)

        if full.mean() <= 0.05:
            out[component] = "NO EFFECT TO MEDIATE: the full patch does little"
            continue
        if control is None:
            out[component] = (
                f"UNCONTROLLED: removing the concept direction costs {loss_j}, but no "
                f"control direction was run, so this cannot be separated from the cost "
                f"of removing any direction of that size"
            )
            continue
        control_rows = np.array(
            [r[control].is_donor_symbol for r in rows], dtype=float
        )
        loss_control = paired_bootstrap(full, control_rows, seed=seed)
        removed_gold = float(np.mean([r["gold"].removed_norm for r in rows]))
        removed_control = float(np.mean([r[control].removed_norm for r in rows]))

        # A projection that removes nothing cannot test anything, and calling that
        # "NOT MEDIATED" would report a degenerate intervention as a null result.
        # This is not hypothetical: under the exact `gradient` and `margin`
        # derivations the removal is *identically* zero, because RMSNorm is
        # scale-free -- the readout does not change along the activation's own
        # direction, so the exact gradient has no component there, and
        # `J^T DN(x)^T u` is therefore orthogonal to `h` by construction. Ablating
        # along the exact logit-gradient direction is vacuous at the very point the
        # gradient was taken. The static vector is the one with overlap to remove.
        activation = float(np.mean([r["gold"].activation_norm for r in rows]))
        fraction = removed_gold / activation if activation else 0.0
        if fraction < 1e-4:
            out[component] = (
                f"NOT MEASURABLE under the {derivation!r} derivation: projecting the "
                f"concept direction out removes {fraction:.2e} of the activation's "
                f"norm, so the intervention is vacuous and its null says nothing. "
                f"For the exact gradient this is structural rather than a bug: the "
                f"final norm is scale-free, so the direction that raises the readout "
                f"carries no component along the activation itself, and there is "
                f"nothing for a projection to take away."
            )
            continue
        ratio = removed_gold / removed_control if removed_control else float("inf")
        aside = (
            f" It removed {ratio:.1f}x as much activation norm as the control "
            f"({fraction:.1%} of the activation against "
            f"{removed_control / activation:.1%}), so part of the contrast is how "
            f"much was removed."
            if ratio > 1.5
            else (
                f" Both removals took comparable activation norm ({ratio:.2f}x, "
                f"{fraction:.1%} of the activation)."
            )
        )

        if loss_j.excludes_zero and not loss_control.excludes_zero:
            share = loss_j.point / full.mean() if full.mean() else float("nan")
            out[component] = (
                f"MEDIATED vs {control}: removing the concept direction costs "
                f"{loss_j} ({share:.0%} of the effect) while the {control} control "
                f"costs {loss_control}.{aside}"
            )
        elif loss_j.excludes_zero and loss_control.excludes_zero:
            out[component] = (
                f"AMBIGUOUS vs {control}: the concept direction costs {loss_j} but so "
                f"does the {control} control ({loss_control}); the projection itself "
                f"does some work.{aside}"
            )
        else:
            out[component] = (
                f"NOT MEDIATED: removing the concept direction costs {loss_j}, which "
                f"does not clear zero; the effect rides on something else.{aside}"
            )
    return out
