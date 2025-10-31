"""Check one-step monotonicity of the prox update."""
from __future__ import annotations

import random

from .common import prox_projection, set_seed


def risk(samples):
    return sum(sum(value * value for value in sample) for sample in samples) / max(len(samples), 1)


def main() -> None:
    set_seed(10)
    samples = [[random.gauss(0.0, 1.0) for _ in range(4)] for _ in range(128)]
    projected = prox_projection(samples, lower=-0.8, upper=0.8)
    risk_before = risk(samples)
    risk_after = risk(projected)
    unchanged = sum(
        1
        for original, proj in zip(samples, projected)
        if all(abs(o - p) <= 1e-8 for o, p in zip(original, proj))
    )

    print("Risk before prox step:", risk_before)
    print("Risk after prox step:", risk_after)
    print("Zero-violation samples unaffected:", unchanged)


if __name__ == "__main__":
    main()
