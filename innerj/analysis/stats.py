"""Paired statistics, clustered at the semantic instance.

Two rules, both learned expensively.

**Resample the cluster, not the observation.** The four conditions of an instance
share a passage, and readouts at nearby layers of one trial share almost
everything. Resampling records as though independent gives an interval several
times too narrow. Everything here resamples semantic instances.

**Report an effect size with an interval, never a bare p-value, and never a mean
of ratios when a denominator can approach zero.** A ratio whose denominator
nearly vanishes produces a spectacular effect out of nothing; the absolute gap
travels beside every ratio for that reason.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Estimate:
    """A point estimate with a bootstrap interval.

    ``excludes_zero`` is the only significance claim this codebase makes.
    """

    point: float
    lo: float
    hi: float
    n: int

    @property
    def excludes_zero(self) -> bool:
        return self.lo > 0.0 or self.hi < 0.0

    @property
    def largest_not_excluded(self) -> float:
        """The biggest effect magnitude this interval is still compatible with.

        The number that makes an absence claim falsifiable. A wide interval around
        zero does **not** show there is no effect; it shows the data cannot tell an
        effect below this magnitude from none. Quoting "nothing transports here"
        from ``not excludes_zero`` is the error this property exists to prevent:
        at L15 the interval is ``[-0.0074, +0.0399]``, so it admits an effect
        *larger* than the ``+0.0135`` confirmed at the transport peak.

        Say "we can exclude effects larger than ``x``", never "there is nothing".
        """
        return max(abs(self.lo), abs(self.hi))

    def equivalent_to_zero(self, bound: float) -> bool:
        """Is the whole interval inside ``[-bound, +bound]``?

        A ROPE / two-one-sided-tests style equivalence claim: absence is asserted
        only when every effect the data admits is smaller than a *prespecified*
        smallest meaningful effect. Prespecify ``bound`` before looking, and record
        it --- choosing it afterwards from the interval that came back is the same
        error as picking a summary statistic after seeing the ranking.

        Using the 95% interval here makes this conservative relative to the
        conventional 90%-interval TOST at alpha=0.05.
        """
        if bound <= 0:
            raise ValueError(
                f"an equivalence bound must be positive, got {bound}; there is no "
                f"such thing as equivalence to within zero"
            )
        return self.largest_not_excluded < bound

    def __str__(self) -> str:
        return f"{self.point:+.4f} [{self.lo:+.4f}, {self.hi:+.4f}] (n={self.n})"


def paired_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Estimate:
    """Mean of ``a - b`` with a percentile interval over paired resamples.

    ``a`` and ``b`` are aligned per instance, one observation each. Pairs are the
    resampling unit, which is what makes the interval honest for a within-
    instance contrast.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"unpaired arrays: {a.shape} vs {b.shape}")
    if a.size == 0:
        raise ValueError("paired_bootstrap on an empty sample")

    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_resamples, diff.size))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Estimate(float(diff.mean()), float(lo), float(hi), int(diff.size))


def cluster_bootstrap(
    values: np.ndarray,
    clusters: np.ndarray,
    *,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Estimate:
    """Mean of ``values`` with clusters as the resampling unit.

    Use when observations are nested (several layers or positions per instance)
    and there is no natural pairing to difference away.
    """
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    if values.shape[0] != clusters.shape[0]:
        raise ValueError(
            f"{values.shape[0]} values, {clusters.shape[0]} cluster labels"
        )
    if values.size == 0:
        raise ValueError("cluster_bootstrap on an empty sample")

    groups = [values[clusters == c] for c in np.unique(clusters)]
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    for r in range(n_resamples):
        picked = rng.integers(0, len(groups), size=len(groups))
        means[r] = np.concatenate([groups[i] for i in picked]).mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Estimate(float(values.mean()), float(lo), float(hi), len(groups))


def ratio_with_gap(
    numerator: float, denominator: float, *, min_denominator: float = 1e-3
) -> tuple[float, float]:
    """A ratio and its absolute gap, refusing to divide by near-zero.

    Returns ``(nan, numerator)`` when the denominator is too small to support a
    ratio. A vanishing denominator has already produced one retracted "effect is
    largest mid-network" claim from a 310x ratio; the absolute gap is the number
    that survived.
    """
    if abs(denominator) < min_denominator:
        return math.nan, numerator
    return numerator / denominator, numerator


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Boolean mask of discoveries at FDR ``alpha``.

    Component screening tests hundreds of candidates; without correction the
    top of the ranking is mostly noise.
    """
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    thresholds = alpha * (np.arange(1, n + 1) / n)
    passed = p[order] <= thresholds
    keep = np.zeros(n, dtype=bool)
    if passed.any():
        cutoff = np.max(np.nonzero(passed)[0])
        keep[order[: cutoff + 1]] = True
    return keep
