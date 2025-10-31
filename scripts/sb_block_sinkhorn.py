"""Block coordinate Sinkhorn solver for synthetic chain-consistent transport."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

from .common import r_linear_rate, set_seed


@dataclass
class SinkhornReport:
    potentials: List[List[float]]
    transport_plan: List[List[List[float]]]
    dual_gaps: List[float]
    r_linear_curve: List[float]


def gaussian_grid(n: int, mean: float, std: float) -> Tuple[List[float], List[float]]:
    if n <= 1:
        return [mean], [1.0]
    start = mean - 3.0 * std
    step = (6.0 * std) / (n - 1)
    grid = [start + i * step for i in range(n)]
    density = [math.exp(-0.5 * ((point - mean) / std) ** 2) for point in grid]
    total = sum(density)
    density = [value / max(total, 1e-12) for value in density]
    return grid, density


def make_cost_tensor(points: List[List[float]]) -> List[List[List[float]]]:
    size = len(points[0]) if points else 0
    centre = sum(points[0]) / max(len(points[0]), 1)
    tensor: List[List[List[float]]] = []
    for i in range(size):
        slice_i: List[List[float]] = []
        for j in range(size):
            row: List[float] = []
            for k in range(size):
                coord = [points[0][i], points[1][j], points[2][k]]
                diff = [(c - centre) for c in coord]
                row.append(sum(d * d for d in diff))
            slice_i.append(row)
        tensor.append(slice_i)
    return tensor


def _compute_plan(kernel: List[List[List[float]]], scaling: List[List[float]]) -> List[List[List[float]]]:
    size0 = len(kernel)
    size1 = len(kernel[0]) if kernel else 0
    size2 = len(kernel[0][0]) if kernel and kernel[0] else 0
    plan: List[List[List[float]]] = []
    for i in range(size0):
        plane: List[List[float]] = []
        for j in range(size1):
            row = []
            for k in range(size2):
                value = kernel[i][j][k]
                value *= scaling[0][i]
                value *= scaling[1][j]
                value *= scaling[2][k]
                row.append(value)
            plane.append(row)
        plan.append(plane)
    return plan


def _marginal(plan: List[List[List[float]]], axis: int) -> List[float]:
    size0 = len(plan)
    size1 = len(plan[0]) if plan else 0
    size2 = len(plan[0][0]) if plan and plan[0] else 0
    if axis == 0:
        return [
            sum(plan[i][j][k] for j in range(size1) for k in range(size2))
            for i in range(size0)
        ]
    if axis == 1:
        return [
            sum(plan[i][j][k] for i in range(size0) for k in range(size2))
            for j in range(size1)
        ]
    return [
        sum(plan[i][j][k] for i in range(size0) for j in range(size1))
        for k in range(size2)
    ]


def _normalise(marginal: List[float], target: List[float]) -> List[float]:
    updated = []
    for current, desired in zip(marginal, target):
        denom = current if abs(current) > 1e-12 else 1e-12
        updated.append(desired / denom)
    return updated


def block_sinkhorn(
    marginals: List[List[float]],
    cost: List[List[List[float]]],
    epsilon: float = 0.5,
    max_iter: int = 200,
    tol: float = 1e-8,
    mu_hat: float = 0.05,
    lipschitz: float = 1.0,
) -> SinkhornReport:
    kernel = [
        [
            [math.exp(-cost[i][j][k] / epsilon) for k in range(len(cost[i][j]))]
            for j in range(len(cost[i]))
        ]
        for i in range(len(cost))
    ]
    scaling = [[1.0 for _ in m] for m in marginals]

    dual_gaps: List[float] = []
    r_linear_curve: List[float] = []

    plan = _compute_plan(kernel, scaling)
    for iteration in range(max_iter):
        for axis in range(len(marginals)):
            marginal = _marginal(plan, axis)
            updates = _normalise(marginal, marginals[axis])
            scaling[axis] = [scale * update for scale, update in zip(scaling[axis], updates)]
            plan = _compute_plan(kernel, scaling)
        marginal_errors = []
        for axis in range(len(marginals)):
            marginal = _marginal(plan, axis)
            marginal_errors.append(sum(abs(a - b) for a, b in zip(marginal, marginals[axis])))
        dual_gap = sum(marginal_errors)
        dual_gaps.append(dual_gap)
        r_linear_curve.append(r_linear_rate(mu_hat, lipschitz, iteration + 1))
        if dual_gap < tol:
            break

    return SinkhornReport(potentials=scaling, transport_plan=plan, dual_gaps=dual_gaps, r_linear_curve=r_linear_curve)


def estimate_strong_convexity(plan: List[List[List[float]]]) -> float:
    values = [value for plane in plan for row in plane for value in row]
    if not values:
        return 1e-3
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return max(variance, 1e-3)


def main() -> None:
    set_seed(7)
    grid_size = 15
    points, m1 = gaussian_grid(grid_size, mean=-0.5, std=0.7)
    _, m2 = gaussian_grid(grid_size, mean=0.0, std=1.0)
    _, m3 = gaussian_grid(grid_size, mean=0.3, std=0.6)
    marginals = [m1, m2, m3]

    cost = make_cost_tensor([points, points, points])
    report = block_sinkhorn(marginals, cost, epsilon=0.8, max_iter=80, mu_hat=0.12, lipschitz=1.5)
    mu_est = estimate_strong_convexity(report.transport_plan)

    final_gap = report.dual_gaps[-1] if report.dual_gaps else float("nan")
    print("Final marginal error (L1 norm):", final_gap)
    print("Number of iterations:", len(report.dual_gaps))
    print("Estimated strong convexity from plan:", mu_est)
    print("R-linear reference curve (first five values):", report.r_linear_curve[:5])


if __name__ == "__main__":
    main()
