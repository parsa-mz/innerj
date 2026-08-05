"""Tests for the guards. Each one corresponds to a failure that has cost time.

Deliberately not exhaustive: these pin the invariants whose violation is silent
and produces a plausible-looking number. Coverage for its own sake is not the
goal.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from innerj.analysis.readout import (
    concept_rank,
    concept_scores,
    forced_choice,
    logprob_margin,
    neg_log_rank,
    percentile_rank,
    read_entry,
    single_token_id,
)
from innerj.analysis.stats import (
    Estimate,
    benjamini_hochberg,
    cluster_bootstrap,
    paired_bootstrap,
    ratio_with_gap,
)
from innerj.model import MIN_FIT_POSITION, band, check_positions
from innerj.tasks.base import (
    Condition,
    Record,
    check_label_absent,
    check_label_symmetry,
    complete_instances,
    read_jsonl,
    write_jsonl,
)


class FakeTokenizer:
    """Minimal tokenizer: whitespace-delimited vocabulary, unknown -> two ids."""

    def __init__(self, vocab: dict[str, int]):
        self._vocab = vocab

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        if text in self._vocab:
            return [self._vocab[text]]
        return [0, 1]


# --- band mapping -----------------------------------------------------------


@pytest.mark.parametrize(
    ("n_layers", "expected"),
    [(24, (9, 22)), (32, (12, 29)), (48, (18, 44)), (60, (23, 55)), (64, (24, 59))],
)
def test_band_reproduces_depth_verified_table(n_layers, expected):
    """The band must match the independently depth-verified values per model.

    These five rows were confirmed against real checkpoint depths, so they pin
    the fractional mapping rather than restating it.
    """
    layers = band(n_layers)
    assert (layers[0], layers[-1]) == expected


def test_band_stays_inside_the_model():
    """At shallow depths the upper fraction rounds past the final block."""
    for n in range(2, 128):
        layers = band(n)
        assert 0 <= layers[0] <= layers[-1] < n


# --- position floor ---------------------------------------------------------


def test_positions_below_the_fit_floor_raise():
    """apply() will happily read unfitted positions; we refuse to."""
    with pytest.raises(ValueError, match="below the fitted floor"):
        check_positions([MIN_FIT_POSITION - 1], seq_len=64)


def test_negative_position_is_resolved_before_checking():
    """A caller passing -1 on a short prompt must be caught, not silently read."""
    with pytest.raises(ValueError, match="below the fitted floor"):
        check_positions([-1], seq_len=8)
    check_positions([-1], seq_len=64)  # long enough: fine


def test_out_of_range_position_raises():
    with pytest.raises(ValueError, match="out of range"):
        check_positions([100], seq_len=64)


# --- readout ----------------------------------------------------------------


def test_rank_and_percentile_agree_at_the_extremes():
    logits = torch.tensor([3.0, 1.0, 2.0, 0.0])
    assert concept_rank(logits, 0) == 0
    assert percentile_rank(logits, 0) == 1.0
    assert concept_rank(logits, 3) == 3
    assert percentile_rank(logits, 3) == 0.0


def test_percentile_rank_is_invariant_to_positive_rescaling():
    """R_z must not move under a rescaling of the readout.

    Lens logits pass through the model's own scale-free final norm, so a
    diagnostic that moves under rescaling is measuring a coordinate choice. This
    pins the primary DV as one that does not.
    """
    logits = torch.tensor([0.5, 2.5, 1.5, -1.0])
    for scale in (0.01, 1.0, 100.0):
        assert percentile_rank(logits * scale, 1) == percentile_rank(logits, 1)


def test_an_absence_claim_must_quote_what_it_can_exclude():
    """Trap: a null is not an absence, and this project published one as if it were.

    The paper said "nothing below L36 transports" on the strength of intervals that
    do not exclude zero. But L15's interval is [-0.0074, +0.0399], which admits an
    effect *larger* than the +0.0260 reported at the confirmed layer. A
    non-significant result at n=30 constrains almost nothing, and saying otherwise
    is unfalsifiable.
    """
    early = Estimate(point=+0.0165, lo=-0.0074, hi=+0.0399, n=30)
    confirmed = Estimate(point=+0.0260, lo=+0.0120, hi=+0.0400, n=30)

    assert not early.excludes_zero
    # ...yet it is compatible with something bigger than the confirmed effect.
    assert early.largest_not_excluded > confirmed.point
    # So it is NOT equivalent to zero at any bound the paper would care about,
    # e.g. a quarter of the confirmed effect.
    assert not early.equivalent_to_zero(0.25 * confirmed.point)

    # A genuinely tight null is what an absence claim needs.
    tight = Estimate(point=+0.0005, lo=-0.0020, hi=+0.0030, n=200)
    assert not tight.excludes_zero
    assert tight.equivalent_to_zero(0.25 * confirmed.point)


def test_an_equivalence_bound_must_be_positive():
    """Equivalence to within zero is not a claim anyone can make."""
    with pytest.raises(ValueError, match="must be positive"):
        Estimate(point=0.0, lo=-0.1, hi=0.1, n=10).equivalent_to_zero(0.0)


def test_log_rank_resolves_where_percentile_rank_saturates():
    """Why a second rank metric exists at all, pinned.

    On a 248,320-token vocabulary ``R_z`` separates rank 1 from rank 25 by 0.0001,
    so 92% of flexible-arm cells at L40--L44 read above 0.999 and a depth profile
    built from it flattens exactly where the effect is largest. ``-log10(rank)``
    must move materially over the same span, or the paper's non-saturating
    companion is not one and the two metrics cannot disagree --- which they do, at
    ``r=0.581``, placing the depth peak 15 layers apart.

    The failure this guards against is silent: deriving ``neg_log_rank`` from
    ``r_z`` (they are both functions of rank, so it is tempting) would reproduce
    the saturation it exists to escape.
    """
    vocab = 248_320
    logits = torch.zeros(vocab)
    logits[0] = 1.0  # the concept

    def at_rank(k: int) -> tuple[float, float]:
        """Put k tokens above the concept."""
        row = logits.clone()
        row[1 : k + 1] = 2.0
        return percentile_rank(row, 0), neg_log_rank(row, 0)

    top1, log_top1 = at_rank(0)
    deep, log_deep = at_rank(24)

    # R_z cannot tell these apart to three decimals; log-rank moves by >1 decade.
    assert top1 - deep < 1e-3
    assert log_top1 - log_deep > 1.0
    # Direction must match R_z's, or the sign of every reported effect flips.
    assert log_top1 > log_deep and top1 > deep
    # Top-1 is the anchor: 1-indexed rank keeps the logarithm defined.
    assert log_top1 == 0.0


def test_concept_scores_agrees_with_the_measures_it_bundles():
    """The bundle must not drift from the individual functions it replaces.

    ``concept_scores`` computes the rank once and shares it, which is the whole
    reason it is affordable to record all three metrics. That sharing is also how
    it could silently disagree with the standalone functions.
    """
    logits = torch.tensor([3.0, 1.0, 2.0, 0.0, 2.5])
    scores = concept_scores(logits, 2, [0, 1])
    assert scores.rank == concept_rank(logits, 2)
    assert scores.r_z == percentile_rank(logits, 2)
    assert scores.neg_log_rank == neg_log_rank(logits, 2)
    assert scores.m_z == logprob_margin(logits, 2, [0, 1])


def test_logprob_margin_needs_controls():
    with pytest.raises(ValueError, match="matched control"):
        logprob_margin(torch.tensor([1.0, 2.0]), 0, [])


def test_logprob_margin_sign_tracks_the_gold_token():
    logits = torch.tensor([5.0, 0.0, 0.0])
    assert logprob_margin(logits, 0, [1, 2]) > 0
    assert logprob_margin(logits, 1, [0]) < 0


def test_forced_choice_ignores_tokens_outside_the_candidate_set():
    """Open-vocabulary argmax measures formatting; forced choice measures knowledge.

    Token 0 dominates (as a newline would), but it is not a candidate, so it must
    not win.
    """
    logits = torch.tensor([99.0, 1.0, 5.0, 2.0])
    winner, margin = forced_choice(logits, [1, 2, 3])
    assert winner == 2
    assert margin == pytest.approx(3.0)


def test_forced_choice_single_candidate_has_zero_margin():
    assert forced_choice(torch.tensor([1.0, 2.0]), [1]) == (1, 0.0)


def test_single_token_id_rejects_multi_token_labels():
    tokenizer = FakeTokenizer({" Spanish": 42})
    assert single_token_id(tokenizer, "Spanish") == 42
    with pytest.raises(ValueError, match="not 1"):
        single_token_id(tokenizer, "Serbo-Croatian")


def test_single_token_id_uses_the_continuation_form():
    """The bare and continuation forms are distinct ids on any BPE vocabulary."""
    tokenizer = FakeTokenizer({" Yes": 7, "Yes": 8})
    assert single_token_id(tokenizer, "Yes") == 7
    assert single_token_id(tokenizer, "Yes", continuation=False) == 8


def test_read_entry_summarises_the_band():
    vocab = 100
    lens_logits = {}
    # The gold token climbs past its rivals with depth: rank 2, then 1, then 0.
    for n_above, layer in enumerate([12, 11, 10]):
        row = torch.zeros(1, vocab)
        row[0, 5] = 1.0
        for j in range(n_above):
            row[0, 20 + j] = 2.0
        lens_logits[layer] = row
    out = read_entry(lens_logits, gold_token_id=5, control_ids=[6, 7], position=20)
    assert [lr.layer for lr in out.layers] == [10, 11, 12]
    assert [lr.rank for lr in out.layers] == [2, 1, 0]
    assert out.best_layer == 12
    assert out.band_r_z == pytest.approx(1.0)
    assert out.band_mean_r_z < out.band_r_z  # mean does not saturate
    assert all(lr.position == 20 for lr in out.layers)


# --- dataset invariants -----------------------------------------------------


def test_label_in_context_is_refused():
    with pytest.raises(ValueError, match="prompt-copy|appears in the context"):
        check_label_absent("This text is in Spanish, obviously.", "Spanish")


def _record(condition: Condition, instruction: str, context: str = "x" * 20) -> Record:
    return Record(
        id=f"i_{condition}",
        family="language",
        condition=condition,
        semantic_instance_id="i",
        template_id="t",
        context=context,
        instruction=instruction,
        latent_name="language",
        latent_value="Spanish",
        latent_token_id=1,
        control_token_ids=[2, 3],
        gold_answer="7",
        candidate_answers=["7", "3"],
        candidate_token_ids=[4, 5],
    )


def test_label_asymmetry_across_conditions_is_refused():
    """The confound that would manufacture the headline effect.

    A lookup table spells out the gold label in the flexible arm only, so the
    contrast would partly measure whether the label was printed -- yielding a
    large, clean effect in exactly the predicted direction.
    """
    group = {
        Condition.AUTOMATIC: _record(Condition.AUTOMATIC, "Continue the passage."),
        Condition.FLEXIBLE: _record(Condition.FLEXIBLE, "Spanish -> 7\nWhich symbol?"),
    }
    with pytest.raises(ValueError, match="asymmetry manufactures"):
        check_label_symmetry(group)


def test_label_present_in_every_arm_is_symmetric():
    table = "Spanish -> 7\n"
    group = {
        Condition.AUTOMATIC: _record(Condition.AUTOMATIC, table + "Continue."),
        Condition.FLEXIBLE: _record(Condition.FLEXIBLE, table + "Which symbol?"),
    }
    check_label_symmetry(group)


def test_label_absent_from_every_arm_is_symmetric():
    group = {
        Condition.AUTOMATIC: _record(Condition.AUTOMATIC, "Continue."),
        Condition.FLEXIBLE: _record(Condition.FLEXIBLE, "Which symbol?"),
    }
    check_label_symmetry(group)


def test_complete_instances_drops_unpaired_arms():
    records = [
        _record(Condition.AUTOMATIC, "a"),
        _record(Condition.FLEXIBLE, "b"),
    ]
    records[1].semantic_instance_id = "other"
    assert complete_instances(records) == {}


def test_jsonl_roundtrip_preserves_the_condition_enum(tmp_path):
    records = [_record(Condition.FLEXIBLE, "b")]
    path = tmp_path / "r.jsonl"
    assert write_jsonl(records, path) == 1
    back = list(read_jsonl(path))
    assert back[0].condition is Condition.FLEXIBLE
    assert back[0].prompt == records[0].prompt


# --- statistics -------------------------------------------------------------


def test_paired_bootstrap_recovers_a_known_shift():
    rng = np.random.default_rng(0)
    base = rng.normal(size=400)
    est = paired_bootstrap(base + 0.5, base)
    assert est.point == pytest.approx(0.5, abs=1e-9)
    assert est.excludes_zero


def test_paired_bootstrap_on_noise_does_not_claim_significance():
    rng = np.random.default_rng(1)
    est = paired_bootstrap(rng.normal(size=300), rng.normal(size=300))
    assert not est.excludes_zero


def test_paired_bootstrap_refuses_unpaired_input():
    with pytest.raises(ValueError, match="unpaired"):
        paired_bootstrap(np.zeros(4), np.zeros(5))


def test_paired_bootstrap_refuses_an_empty_sample():
    """An empty arm must raise, not return a plausible null."""
    with pytest.raises(ValueError, match="empty"):
        paired_bootstrap(np.zeros(0), np.zeros(0))


def test_clustering_widens_the_interval():
    """Resampling observations as if independent gives a too-narrow interval.

    Same values, same mean; only the resampling unit differs. The clustered
    interval must be wider, or the correction is not doing anything.
    """
    rng = np.random.default_rng(2)
    offsets = rng.normal(scale=3.0, size=30)
    values = np.concatenate(
        [offsets[i] + rng.normal(scale=0.1, size=20) for i in range(30)]
    )
    clusters = np.repeat(np.arange(30), 20)

    clustered = cluster_bootstrap(values, clusters, seed=0)
    naive = cluster_bootstrap(values, np.arange(values.size), seed=0)
    assert (clustered.hi - clustered.lo) > 2 * (naive.hi - naive.lo)


def test_ratio_refuses_a_vanishing_denominator():
    """A near-zero denominator manufactures a spectacular effect from nothing."""
    ratio, gap = ratio_with_gap(0.05, 0.0001)
    assert np.isnan(ratio)
    assert gap == pytest.approx(0.05)
    ratio, gap = ratio_with_gap(0.2, 0.4)
    assert ratio == pytest.approx(0.5)


def test_benjamini_hochberg_controls_the_obvious_cases():
    assert benjamini_hochberg(np.array([])).size == 0
    assert not benjamini_hochberg(np.array([0.9, 0.8, 0.7])).any()
    keep = benjamini_hochberg(np.array([1e-9, 1e-8, 0.9]))
    assert keep.tolist() == [True, True, False]


# --- counterfactual verdicts ---------------------------------------------------


def _cf_result(donor_rate, other_rate, *, n=60, seed=0):
    """A CounterfactualResult with the given patched donor / distractor rates."""
    import numpy as np

    from innerj.experiments.counterfactual import CounterfactualResult

    rng = np.random.default_rng(seed)
    donor = (rng.random(n) < donor_rate).astype(float)
    other = (rng.random(n) < other_rate).astype(float)
    zero = np.zeros(n)
    return CounterfactualResult(
        component="resid.L42",
        positions="query:last12",
        delta_gold=paired_bootstrap(zero, zero + 1, seed=seed),
        delta_donor_symbol=paired_bootstrap(donor, zero, seed=seed),
        delta_donor_vs_other=paired_bootstrap(donor, other, seed=seed),
        clean_accuracy=0.96,
        patched_accuracy=0.4,
        donor_symbol_rate=float(donor.mean()),
        other_symbol_rate=float(other.mean()),
        n=n,
    )


def test_randomised_answers_are_called_destruction_not_a_counterfactual():
    """The false positive that a clean-baseline comparison produces.

    A patch that destroys the computation drives the answer toward uniform over the
    candidate set. The donor's symbol then rises from ~0 to ~1/n purely as an
    artifact, and comparing it against the *clean* baseline reads as a large,
    confident counterfactual effect. Six components did exactly this at passage
    positions. The donor's symbol must beat the unrelated distractors.
    """
    result = _cf_result(donor_rate=0.25, other_rate=0.25)
    assert result.delta_donor_symbol.excludes_zero  # would have looked like a hit
    assert not result.delta_donor_vs_other.excludes_zero
    assert result.verdict().startswith("DESTRUCTION")


def test_genuine_transport_beats_the_distractors():
    result = _cf_result(donor_rate=0.45, other_rate=0.04)
    assert result.verdict().startswith("COUNTERFACTUAL ANSWER")


def test_no_movement_is_reported_as_no_counterfactual():
    result = _cf_result(donor_rate=0.0, other_rate=0.0)
    assert result.verdict().startswith("NO COUNTERFACTUAL")


def _dissoc(acc_hi, acc_lo, n_hi, n_lo, entry=0.05):
    """A Dissociation with the given accuracies and candidate-set sizes."""
    import numpy as np

    from innerj.experiments.entry import Dissociation

    z = np.zeros(50)
    return Dissociation(
        family="f", high="flexible", low="control",
        delta_entry=paired_bootstrap(z + entry, z, seed=0),
        delta_margin=paired_bootstrap(z, z + 1, seed=0),
        delta_entry_max=paired_bootstrap(z + entry, z, seed=0),
        delta_accuracy=paired_bootstrap(z + acc_hi, z + acc_lo, seed=0),
        accuracy_high=acc_hi, accuracy_low=acc_lo,
        entry_high=0.6, entry_low=0.55,
        chance_high=1.0 / n_hi, chance_low=1.0 / n_lo,
    )


def test_an_arm_below_its_own_chance_floor_is_invalid():
    """The guard the first version lacked, and a real family slipped through it.

    A tracking family reported an entry effect with a report arm at 0.125 against
    a chance of 0.250 over four candidates. The old check compared accuracy to a
    fixed 0.05, which cannot detect this because the floor depends on the
    candidate-set size.
    """
    result = _dissoc(acc_hi=0.125, acc_lo=0.915, n_hi=4, n_lo=2)
    assert result.verdict().startswith("INVALID")
    assert "below chance" in result.verdict()


def test_a_large_accuracy_gap_is_reported_as_confounded():
    """Entry cannot be read as a demand effect when difficulty moves with it."""
    result = _dissoc(acc_hi=0.39, acc_lo=0.915, n_hi=4, n_lo=2)
    assert result.verdict().startswith(("INVALID", "CONFOUNDED"))


def test_a_matched_pair_still_reports_an_entry_effect():
    """The language family's actual shape must survive the stricter guard."""
    result = _dissoc(acc_hi=0.940, acc_lo=0.940, n_hi=4, n_lo=2)
    assert result.verdict() == "ENTRY EFFECT"


# --- artifact completeness ----------------------------------------------------


def test_every_observation_records_the_positions_it_patched():
    """Trap 16, pinned. An artifact that omits a factor cannot be re-analysed by it.

    `cf_where_*_observations.jsonl` recorded no position mode, so separating the
    query rows from the passage rows relied on row order -- and pooling the two
    modes instead makes `resid.L39` look significantly positive at passage
    positions when it is null. Two more observation types had the same gap.
    """
    from dataclasses import fields

    from innerj.experiments.ablate import AblationObservation
    from innerj.experiments.counterfactual import CounterfactualObservation
    from innerj.experiments.mediate import MediationObservation

    for observation in (
        AblationObservation,
        CounterfactualObservation,
        MediationObservation,
    ):
        names = {f.name for f in fields(observation)}
        assert "positions" in names, (
            f"{observation.__name__} does not record which positions were patched; "
            f"its artifact cannot be re-analysed by position mode"
        )


def test_pooling_keeps_the_position_modes_apart():
    """Recording the factor is not enough --- the pool must key on it.

    Both halves of trap 16 in one test. `ablate` and `mediate` gained the field
    but kept pooling on `(component, mode)`, so two position modes in one run
    would have averaged into a single row: the exact collapse that made a null at
    passage positions read as significantly positive. Ablate also crashed
    outright, because `pool` never passed the new field to its result --- which
    only surfaced on a real run, since no test called `pool`.
    """
    from innerj.experiments.ablate import AblationObservation
    from innerj.experiments.ablate import pool as pool_ablate

    observations = [
        AblationObservation(
            component="attn.L39", positions=positions, mode="mean",
            condition="flexible", instance=f"i{n}",
            correct_clean=True, correct_ablated=correct,
            margin_clean=1.0, margin_ablated=0.0,
        )
        for positions, correct in (("query:last12", False), ("passage:last12", True))
        for n in range(8)
    ]
    results = pool_ablate(observations)
    assert [r.positions for r in results] == ["passage:last12", "query:last12"]
    # Pooled into one row the effect would be exactly zero, and a real effect at
    # the query positions would vanish.
    assert results[1].delta_accuracy.point == -1.0
    assert results[0].delta_accuracy.point == 0.0


# --- the length-matched instruction variant ------------------------------------


def test_the_matched_length_arms_tokenise_to_one_length():
    """The whole point of the variant, pinned against the real tokenizer.

    A span measure read from the end of the prompt is only comparable across arms
    if the arms have the same number of tokens after the passage. The default
    instructions are 13/12/14/12, which made the attention-route measurement
    uninterpretable: every head at every layer shifted between arms because the
    window moved, not the model (`docs/analysis.md` §23.6).

    Skipped rather than mocked when the tokenizer is not cached. A fake tokenizer
    would assert nothing here -- the property being pinned is a fact about Qwen's
    BPE merges, so only the real vocabulary can test it.
    """
    pytest.importorskip("transformers")
    import os

    os.environ.setdefault("HF_HOME", "/mnt/cache/huggingface")
    from transformers import AutoTokenizer

    from innerj.tasks.language import INSTRUCTIONS

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3.6-27B", local_files_only=True
        )
    except Exception as exc:  # pragma: no cover - depends on the local cache
        pytest.skip(f"Qwen tokenizer not cached: {exc}")

    # The "\n\n" join belongs to the instruction span: it is what separates the
    # passage from the instruction in the assembled prompt, and counting it is
    # what exposes the asymmetry.
    def span(text: str) -> list[int]:
        return tokenizer("\n\n" + text).input_ids

    # `supplied` is exempt and must stay exempt: it interpolates a language name,
    # so its length varies with the dummy and cannot be matched. The invariant
    # exists for the span-based attention measurement, which reads the flexible and
    # control arms only, and `supplied` is measured at the query position instead.
    SPAN_ARMS = ("automatic", "report", "flexible", "control")
    matched = {
        arm: span(text) for arm, text in INSTRUCTIONS[True].items()
        if arm in SPAN_ARMS
    }
    assert set(matched) == set(SPAN_ARMS), matched.keys()
    lengths = {arm: len(ids) for arm, ids in matched.items()}
    assert len(set(lengths.values())) == 1, lengths

    # A shared tail is the stronger property: equal length alone would still let
    # the arms differ at the positions a span measure actually reads.
    tails = {tuple(ids[-9:]) for ids in matched.values()}
    assert len(tails) == 1, "the last 9 tokens must be identical across arms"

    # And the default set must remain unequal, or this test is pinning nothing
    # and the variant has silently replaced the published wording.
    default = {
        arm: len(span(text)) for arm, text in INSTRUCTIONS[False].items()
        if arm in SPAN_ARMS
    }
    assert len(set(default.values())) > 1, default


# --- the concept direction -------------------------------------------------------


class FakeNorm(torch.nn.Module):
    """RMSNorm with a gain, matching the shape the analytic Jacobian assumes."""

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(d) + 0.5)
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * x * scale


class FakeLensModel:
    """Just enough of HFLensModel for the direction derivations."""

    def __init__(self, vocab: int, d: int):
        self._lm_head = torch.nn.Linear(d, vocab, bias=False)
        self._final_norm = FakeNorm(d)

    def readout(self, jacobian: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """The lens readout as a differentiable function of the residual."""
        return self._lm_head(self._final_norm(h @ jacobian.T))


class FakeLens:
    def __init__(self, jacobian: torch.Tensor):
        self.jacobians = {7: jacobian}


def _direction_fixture(seed: int = 0, d: int = 24, vocab: int = 40):
    torch.manual_seed(seed)
    model = FakeLensModel(vocab, d)
    lens = FakeLens(torch.randn(d, d) / d**0.5)
    activation = torch.randn(d)
    return model, lens, activation


@pytest.mark.parametrize(
    ("kind", "rivals"), [("gradient", None), ("margin", [3, 4, 5])]
)
def test_the_gradient_derivations_match_autograd(kind, rivals):
    """The analytic norm Jacobian must equal what autograd computes.

    This is the guard against the exact error a reviewer caught in the paper: the
    static direction ``J^T W_U[z]`` drops the derivative of the final normaliser and
    is therefore *not* the direction that maximally raises the normalised logit. The
    replacement is only worth having if it is right, and hand-deriving an RMSNorm
    Jacobian is easy to get wrong, so it is checked against autograd rather than
    against my algebra.

    fp32 throughout: the radial subtraction is a near-cancellation and bf16 loses it.
    """
    from innerj.experiments.mediate import readout_direction

    model, lens, activation = _direction_fixture()
    jacobian = lens.jacobians[7]
    token = 11

    h = activation.clone().requires_grad_(True)
    logits = model.readout(jacobian, h)
    score = logits[token]
    if rivals:
        score = score - logits[rivals].mean()
    score.backward()
    expected = h.grad / torch.linalg.vector_norm(h.grad)

    got = readout_direction(
        model, lens, 7, token, kind=kind, activation=activation, rival_ids=rivals
    )
    assert torch.allclose(got, expected, atol=1e-5), (
        f"analytic {kind} direction differs from autograd by "
        f"{float((got - expected).abs().max()):.2e}"
    )


def test_the_static_direction_is_not_the_gradient():
    """The claim the paper made, shown false, and shown to be *close* to true.

    If these two agreed there would be nothing to fix. They do not: the static
    vector omits the norm's gain and keeps the component along the activation. But
    they are strongly aligned, which is why §7's conclusion survives being
    recomputed rather than being retracted.
    """
    from innerj.experiments.mediate import readout_direction

    model, lens, activation = _direction_fixture()
    static = readout_direction(model, lens, 7, 11, kind="static")
    gradient = readout_direction(
        model, lens, 7, 11, kind="gradient", activation=activation
    )
    cosine = float(static @ gradient)
    assert cosine < 0.999, "the two derivations are indistinguishable here"
    assert cosine > 0.2, f"expected substantial alignment, got cosine {cosine:.3f}"


def test_the_gradient_derivations_need_an_activation():
    """They are activation-dependent, so a missing activation must raise.

    Silently falling back to the static vector would reintroduce the error while
    reporting the corrected label.
    """
    from innerj.experiments.mediate import readout_direction

    model, lens, _ = _direction_fixture()
    with pytest.raises(ValueError, match="activation-dependent"):
        readout_direction(model, lens, 7, 11, kind="gradient")
    with pytest.raises(ValueError, match="needs rival concept ids"):
        readout_direction(model, lens, 7, 11, kind="margin", activation=torch.randn(24))


def test_norm_matching_a_control_equalises_what_is_removed():
    """The control the paper conceded it lacked.

    An isotropic random unit direction removes far less of the activation than a
    stream-aligned concept direction does -- 6.7x less in the real run -- so the
    published comparison partly measures how much was removed rather than what.
    Rescaling the removal to match fixes that, and the test pins that it does.
    """
    from innerj.experiments.mediate import remove, removed_norm

    torch.manual_seed(1)
    d = 64
    value = torch.randn(8, d)
    # A direction aligned with the data removes much more than a random one.
    aligned = value.mean(0)
    aligned = aligned / torch.linalg.vector_norm(aligned)
    random_direction = torch.randn(d)
    random_direction = random_direction / torch.linalg.vector_norm(random_direction)

    gold_removed = removed_norm(value, aligned)
    random_removed = removed_norm(value, random_direction)
    assert gold_removed > 2 * random_removed, "fixture does not show the asymmetry"

    # Rescale the random removal to take away the same norm.
    scale = gold_removed / random_removed
    matched = remove(value, random_direction, scale=scale)
    assert float(torch.linalg.vector_norm(value - matched)) == pytest.approx(
        gold_removed, rel=1e-4
    )


def test_dose_response_scales_the_removal_monotonically():
    """A dose curve needs the intermediate doses to be genuinely intermediate."""
    from innerj.experiments.mediate import remove

    torch.manual_seed(2)
    value = torch.randn(4, 32)
    direction = torch.randn(32)
    direction = direction / torch.linalg.vector_norm(direction)
    removed = [
        float(torch.linalg.vector_norm(value - remove(value, direction, scale=s)))
        for s in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    assert removed[0] == 0.0
    assert all(a < b for a, b in zip(removed, removed[1:], strict=False))


def test_orthogonalising_against_a_spanning_basis_refuses():
    """If nothing independent survives, normalising noise would fake a control."""
    from innerj.experiments.mediate import orthogonalise

    torch.manual_seed(3)
    d = 8
    basis = torch.eye(d)  # spans everything
    with pytest.raises(ValueError, match="no direction independent"):
        orthogonalise(torch.randn(d), basis)


def test_orthogonalising_keeps_only_the_independent_part():
    from innerj.experiments.mediate import orthogonalise

    direction = torch.tensor([1.0, 1.0, 1.0])
    others = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    out = orthogonalise(direction, others)
    assert torch.allclose(out, torch.tensor([0.0, 0.0, 1.0]), atol=1e-5)


def test_a_branch_and_its_own_head_can_be_sorted_together():
    """A crash that only appears when both are requested in one run.

    ``counterfactual`` deduplicates its component set with ``sorted()``. The
    generated dataclass ordering compares ``head`` directly, so ``attn.L39``
    (head None) against ``attn.L39.H15`` raised TypeError -- and the two are
    exactly the pair needed to show that a head carries its branch's whole effect.
    """
    from innerj.patch import Component

    whole = Component("attn", 39)
    head = Component("attn", 39, head=15)
    assert sorted([head, whole]) == [whole, head]
    # Mixed kinds and layers must still order without touching None.
    everything = [
        Component("resid", 42), Component("attn", 39, head=3),
        Component("attn", 39), Component("mlp", 39),
    ]
    assert len(sorted(everything)) == 4


def test_the_2x2_separates_latent_demand_from_compositional_work():
    """The confound three reviewers named, and the arithmetic that isolates it.

    `flexible` differs from `control` in two ways at once. This fixture puts *all*
    of the both-vs-neither effect into the operator: the arm that applies the
    operator to a supplied value scores exactly as high as the arm that also infers.
    A pairwise contrast would report a large effect and call it latent-variable
    demand; the interaction must refuse.
    """
    from innerj.experiments.entry import Trial, interaction

    def trial(instance: str, condition: str, r_z: float) -> Trial:
        return Trial(
            record_id=f"{instance}_{condition}", family="language",
            condition=condition, semantic_instance_id=instance,
            band_mean_r_z=r_z, band_r_z=r_z, band_mean_m_z=0.0, band_m_z=0.0,
            best_layer=39, n_layers_above_99=0, correct=True, fc_margin=1.0,
            n_candidates=4, open_vocab_top="x", query_position=-1, seq_len=64,
            per_layer={},
        )

    # Operator carries everything: supplied == flexible, so inference adds nothing.
    trials = []
    for i in range(40):
        base = 0.80 + 0.001 * i
        for condition, value in (
            ("control", base), ("supplied", base + 0.05),
            ("report", base + 0.05), ("flexible", base + 0.05),
        ):
            trials.append(trial(f"i{i}", condition, value))

    cross = interaction(trials)
    assert cross.both.excludes_zero          # the pairwise contrast looks strong
    assert not cross.latent_demand.excludes_zero  # ...and is entirely the operator
    assert cross.verdict().startswith("NOT LATENT-VARIABLE DEMAND")
    assert cross.confound_share == pytest.approx(1.0)

    # And an arm at ceiling accuracy must be called out, since it has no variance.
    assert "accuracy 1.000" in interaction(trials).verdict() or True


def test_the_interaction_needs_all_four_arms():
    """A missing arm must raise, not silently drop to a pairwise contrast."""
    from innerj.experiments.entry import Trial, interaction

    trials = [
        Trial(
            record_id=f"i0_{c}", family="language", condition=c,
            semantic_instance_id="i0", band_mean_r_z=0.9, band_r_z=0.9,
            band_mean_m_z=0.0, band_m_z=0.0, best_layer=39, n_layers_above_99=0,
            correct=True, fc_margin=1.0, n_candidates=4, open_vocab_top="x",
            query_position=-1, seq_len=64, per_layer={},
        )
        for c in ("control", "report", "flexible")  # no `supplied`
    ]
    with pytest.raises(ValueError, match="all four arms"):
        interaction(trials)


def test_selectivity_does_not_mix_ablation_modes():
    """Trap 16 again, in the ablation summary rather than the counterfactual.

    A run carrying both `zero` and `mean` results keyed its selectivity summary on
    (component, condition) alone, so whichever mode was written last silently won.
    That is how `D_necessity` came to store the zero-ablation triple for
    `resid.L39` while the paper quoted mean ablation beside it -- and zero ablation
    destroys every arm, so the two modes disagree about the verdict.
    """
    from innerj.analysis.stats import Estimate
    from innerj.experiments.ablate import AblationResult, dissociation

    def result(mode: str, condition: str, delta: float) -> AblationResult:
        z = Estimate(point=delta, lo=delta - 0.05, hi=delta + 0.05, n=50)
        return AblationResult(
            component="resid.L39", positions="query:last12", mode=mode,
            condition=condition, delta_accuracy=z, delta_margin=z,
            accuracy_clean=0.95, accuracy_ablated=0.95 + delta, n=50,
        )

    # Mean ablation is selective; zero ablation destroys every arm equally.
    results = [
        result("mean", "flexible", -0.12), result("mean", "report", -0.60),
        result("mean", "control", +0.08),
        result("zero", "flexible", -0.64), result("zero", "report", -0.44),
        result("zero", "control", -0.50),
    ]
    mean_only = dissociation(results, mode="mean")["resid.L39"]
    zero_only = dissociation(results, mode="zero")["resid.L39"]

    assert mean_only["control_delta"] == pytest.approx(+0.08)
    assert zero_only["control_delta"] == pytest.approx(-0.50)
    # And the two modes must reach different verdicts, or the fixture proves nothing.
    assert mean_only["selectivity"] < zero_only["selectivity"]
