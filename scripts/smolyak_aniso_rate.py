"""Simulate the anisotropic Smolyak approximation rate (Claim C1)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

from .common import (
    l2_error,
    make_anisotropic_sampler,
    set_seed,
    smolyak_cpwl_approximation,
)


@dataclass
class ComplexityPoint:
    width: int
    error: float
    depth: int


def simulate_complexity_frontier(levels: List[int]) -> List[ComplexityPoint]:
    beta = [1.0, 1.5, 2.5]
    weights = [1.0, 0.7, 0.5]
    fun, sampler = make_anisotropic_sampler(beta, weights)

    points: List[ComplexityPoint] = []
    for level in levels:
        grid_sizes = [max(3, level + j) for j in range(len(beta))]
        approx, linear_regions = smolyak_cpwl_approximation(fun, grid_sizes)
        error = l2_error(fun, approx, sampler, n_samples=512)
        points.append(ComplexityPoint(width=linear_regions, error=error, depth=4))
    return points


def fit_power_law(points: List[ComplexityPoint]) -> float:
    log_width = [math.log(point.width) for point in points]
    log_error = [math.log(point.error) for point in points]
    mean_x = sum(log_width) / len(log_width)
    mean_y = sum(log_error) / len(log_error)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_width, log_error))
    denominator = sum((x - mean_x) ** 2 for x in log_width)
    return numerator / denominator if denominator else 0.0


def main() -> None:
    set_seed(42)
    levels = [2, 3, 4, 5, 6]
    points = simulate_complexity_frontier(levels)
    slope = fit_power_law(points)

    print("Anisotropic Smolyak approximation complexity frontier")
    print("level width depth error")
    for level, point in zip(levels, points):
        print(f"{level:>5d} {point.width:>5d} {point.depth:>5d} {point.error: .4e}")
    beta_bar = 1.0 / sum(1.0 / b for b in [1.0, 1.5, 2.5])
    theoretical_slope = -2 * beta_bar
    print("Estimated log-log slope (should be close to -2*beta_bar):", slope)
    print("Reference theoretical slope:", theoretical_slope)


if __name__ == "__main__":
    main()
