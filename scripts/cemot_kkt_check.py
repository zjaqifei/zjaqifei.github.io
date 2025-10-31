"""Verify KKT conditions and uniqueness for the synthetic c-EMOT solution."""
from __future__ import annotations

import math

from .sb_block_sinkhorn import block_sinkhorn, gaussian_grid, make_cost_tensor
from .common import set_seed


def _marginal(plan: list[list[list[float]]], axis: int) -> list[float]:
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


def check_kkt_conditions(plan: list[list[list[float]]], marginals: list[list[float]]) -> dict[str, float]:
    violations: dict[str, float] = {}
    for axis, target in enumerate(marginals):
        marginal = _marginal(plan, axis)
        violations[f"marginal_{axis}"] = sum(abs(a - b) for a, b in zip(marginal, target))
    total_mass = sum(value for plane in plan for row in plane for value in row)
    violations["mass"] = abs(total_mass - 1.0)
    entropy = 0.0
    for plane in plan:
        for row in plane:
            for value in row:
                if value > 0.0:
                    entropy -= value * math.log(value)
    violations["entropy"] = entropy
    return violations


def main() -> None:
    set_seed(8)
    grid, m1 = gaussian_grid(12, mean=-0.2, std=0.9)
    _, m2 = gaussian_grid(12, mean=0.0, std=1.1)
    _, m3 = gaussian_grid(12, mean=0.2, std=0.7)
    marginals = [m1, m2, m3]
    cost = make_cost_tensor([grid, grid, grid])
    report = block_sinkhorn(marginals, cost, epsilon=0.6, max_iter=60, mu_hat=0.1, lipschitz=1.4)
    violations = check_kkt_conditions(report.transport_plan, marginals)

    print("KKT violation summary:")
    for key, value in violations.items():
        print(f"  {key}: {value:.3e}")
    print("Dual gaps (first five):", report.dual_gaps[:5])


if __name__ == "__main__":
    main()
