"""Estimate the empirical strong convexity parameter \hat{\mu}."""
from __future__ import annotations

import math

from .sb_block_sinkhorn import block_sinkhorn, gaussian_grid, make_cost_tensor
from .common import set_seed


def spectral_bounds(plan: list[list[list[float]]]) -> tuple[float, float]:
    rows = [[value for row in plane for value in row] for plane in plan]
    if not rows:
        return 1e-3, 1e-3
    norms = [math.sqrt(sum(value * value for value in row)) for row in rows]
    lower = min(norms) ** 2
    upper = max(norms) ** 2
    return max(lower, 1e-3), max(upper, 1e-3)


def main() -> None:
    set_seed(3)
    grid, m1 = gaussian_grid(15, mean=-0.3, std=1.0)
    _, m2 = gaussian_grid(15, mean=0.0, std=0.8)
    _, m3 = gaussian_grid(15, mean=0.4, std=0.6)
    marginals = [m1, m2, m3]
    cost = make_cost_tensor([grid, grid, grid])
    report = block_sinkhorn(marginals, cost, epsilon=0.7, max_iter=50, mu_hat=0.09, lipschitz=1.3)
    mu_lower, mu_upper = spectral_bounds(report.transport_plan)
    safety_lower = max(mu_lower * 0.5, 1e-3)

    print("Spectral lower bound:", mu_lower)
    print("Spectral upper bound:", mu_upper)
    print("Safety lower bound (used in convergence proofs):", safety_lower)


if __name__ == "__main__":
    main()
