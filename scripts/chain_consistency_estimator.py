"""U-statistic estimator for chain-consistency distances (Section R2)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

from .common import gaussian_chain_samples, set_seed


@dataclass
class ChainEstimate:
    estimate: float
    variance: float
    bias_proxy: float
    bandwidth: float


def _kernel(x: List[float], y: List[float], sigma: float) -> float:
    diff = sum((a - b) ** 2 for a, b in zip(x, y))
    return math.exp(-diff / (2.0 * sigma * sigma))


def _u_statistic(samples: List[List[float]], sigma: float) -> float:
    n = len(samples)
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += _kernel(samples[i], samples[j], sigma)
            count += 1
    return total / max(count, 1)


def estimate_chain_distance(batches: List[List[List[float]]], sigma: float) -> ChainEstimate:
    estimates = [_u_statistic(batch, sigma) for batch in batches]
    mean_est = sum(estimates) / max(len(estimates), 1)
    variance = sum((value - mean_est) ** 2 for value in estimates) / max(len(estimates), 1)
    bias_proxy = sum(abs(value - mean_est) for value in estimates) / max(len(estimates), 1)
    return ChainEstimate(estimate=mean_est, variance=variance, bias_proxy=bias_proxy, bandwidth=sigma)


def select_bandwidth(batches: List[List[List[float]]], grid: Tuple[float, ...]) -> ChainEstimate:
    best_estimate = None
    best_score = float("inf")
    for sigma in grid:
        candidate = estimate_chain_distance(batches, sigma)
        score = candidate.variance + 0.1 * candidate.bias_proxy**2
        if best_estimate is None or score < best_score:
            best_estimate = candidate
            best_score = score
    assert best_estimate is not None
    return best_estimate


def main() -> None:
    set_seed(23)
    batches = gaussian_chain_samples(n=48, dim=2, steps=4, noise=0.3)
    result = select_bandwidth(batches, grid=(0.2, 0.3, 0.5, 0.8))

    print("Selected bandwidth:", result.bandwidth)
    print("Estimate:", result.estimate)
    print("Variance:", result.variance)
    print("Bias proxy:", result.bias_proxy)


if __name__ == "__main__":
    main()
