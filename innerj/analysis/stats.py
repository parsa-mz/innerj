"""Paired statistics, clustered at the semantic instance.

**Resample the cluster, not the observation.** An instance's conditions share a passage
and
nearby layers share almost everything, so resampling records as independent gives an
interval
several times too narrow.

**Never a bare p-value, and never a mean of ratios when a denominator can approach
zero** --
a vanishing denominator manufactures an effect, so an absolute gap travels beside every
ratio.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Estimate:
    """A point estimate with a bootstrap interval; ``excludes_zero`` is the only
        significance claim this codebase makes.
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

                What makes an absence claim falsifiable. At L15 the interval is
                ``[-0.0074, +0.0399]``, admitting an effect *larger* than the
                ``+0.0135`` confirmed
                at the peak. Say "we can exclude effects larger than x", never "there is
                nothing".
        """
        return max(abs(self.lo), abs(self.hi))

    def equivalent_to_zero(self, bound: float) -> bool:
        """Is the whole interval inside ``[-bound, +bound]``?

                A ROPE/TOST-style claim: absence is asserted only when every admitted
                effect is
                smaller than a *prespecified* smallest meaningful effect. Choosing
                ``bound`` after
                seeing the interval is the same error as picking a statistic after
                seeing the
                ranking.
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
    """Mean of ``a - b`` over paired resamples, pairs being the resampling unit --
        what makes the interval honest for a within-instance contrast.
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
    """Mean of ``values`` with clusters as the resampling unit, for nested
        observations with no natural pairing to difference away.
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
    """A ratio and its absolute gap, returning ``(nan, numerator)`` when the
        denominator is too small -- one already produced a retracted claim from a 310x
        ratio.
    """
    if abs(denominator) < min_denominator:
        return math.nan, numerator
    return numerator / denominator, numerator


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Boolean mask of discoveries at FDR ``alpha``; without it the top of a
        hundreds-wide ranking is mostly noise.
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
