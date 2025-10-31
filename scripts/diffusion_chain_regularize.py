"""Synthetic diffusion model regularised by chain-consistency."""
from __future__ import annotations

from typing import List

from .common import (
    chain_consistency_statistic,
    gaussian_chain_samples,
    martingale_penalty,
    prox_projection,
    set_seed,
)


def _average(values: List[float]) -> float:
    return sum(values) / max(len(values), 1)


def main() -> None:
    set_seed(11)
    batches = gaussian_chain_samples(n=64, dim=3, steps=6, noise=0.2)
    consistency_curve = []
    prox_budget = []
    martingale_curve = []

    running_state = [[0.0, 0.0, 0.0] for _ in range(64)]
    for batch in batches:
        blended = [
            [0.7 * r + 0.3 * b for r, b in zip(row_r, row_b)]
            for row_r, row_b in zip(running_state, batch)
        ]
        projected = prox_projection(blended, lower=-1.5, upper=1.5)
        running_state = projected
        consistency = chain_consistency_statistic([running_state, batch])
        paths = [[running_state[i], batch[i]] for i in range(len(batch))]
        martingale = martingale_penalty(paths)
        budget = [sum((p - b) ** 2 for p, b in zip(row_p, row_b)) for row_p, row_b in zip(projected, blended)]
        consistency_curve.append(consistency)
        prox_budget.append(_average(budget))
        martingale_curve.append(martingale)

    def format_curve(values: List[float]) -> str:
        return ", ".join(f"{v:.3f}" for v in values)

    print("Chain-consistency curve:", format_curve(consistency_curve))
    print("Prox budget per step:", format_curve(prox_budget))
    print("Martingale penalty per step:", format_curve(martingale_curve))


if __name__ == "__main__":
    main()
