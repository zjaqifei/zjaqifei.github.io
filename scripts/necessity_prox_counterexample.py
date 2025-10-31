"""Counterexample demonstrating the need for prox projections."""
from __future__ import annotations

import random

from .common import prox_projection, set_seed


def non_prox_map(samples, scale: float = 1.2):
    return [[scale * value for value in sample] for sample in samples]


def risk(samples):
    return sum(sum(value * value for value in sample) for sample in samples) / max(len(samples), 1)


def main() -> None:
    set_seed(4)
    samples = [[random.gauss(0.0, 1.0) for _ in range(3)] for _ in range(512)]
    proxed = prox_projection(samples)
    scaled = non_prox_map(samples)

    risk_opt = risk(proxed)
    risk_scaled = risk(scaled)

    print("Risk with prox projection:", risk_opt)
    print("Risk with Lipschitz>1 operator:", risk_scaled)
    print("Excess risk (should be positive):", risk_scaled - risk_opt)


if __name__ == "__main__":
    main()
