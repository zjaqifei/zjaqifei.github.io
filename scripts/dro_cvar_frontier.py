"""Compute a synthetic DRO-CVaR cost frontier (Section 8)."""
from __future__ import annotations

import random

from .common import empirical_cvar, set_seed


def simulate_losses(n: int = 2048) -> list[float]:
    losses = []
    for _ in range(n):
        base = random.gauss(0.0, 1.0)
        tail = random.gauss(0.0, 2.5) if random.random() < 0.1 else 0.0
        losses.append(0.8 * base + tail)
    return losses


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def main() -> None:
    set_seed(17)
    losses = simulate_losses()
    radii = [i / 7 for i in range(8)]
    cvar_curve = []
    cost_curve = []

    for rho in radii:
        perturbed = [loss + rho * random.gauss(0.0, 1.0) for loss in losses]
        cvar = empirical_cvar(perturbed, alpha=0.95)
        cost = mean([max(loss, 0.0) for loss in perturbed])
        cvar_curve.append(cvar)
        cost_curve.append(cost)

    print("rho cvar cost")
    for rho, cvar, cost in zip(radii, cvar_curve, cost_curve):
        print(f"{rho: .2f} {cvar: .3f} {cost: .3f}")


if __name__ == "__main__":
    main()
