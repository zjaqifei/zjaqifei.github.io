"""Illustrate the necessity of chain-consistency regularisation."""
from __future__ import annotations

import random

from .common import chain_consistency_statistic, set_seed


def _make_chain(offset: float, noise: float) -> list[list[float]]:
    return [[random.gauss(offset, noise)] for _ in range(128)]


def main() -> None:
    set_seed(21)
    base_chain = [_make_chain(i * 0.1, 0.3) for i in range(6)]
    perturbed_chain = [[value[:] for value in chain] for chain in base_chain]
    for chain in perturbed_chain:
        for vector in chain:
            vector[0] += random.gauss(0.0, 0.5)

    base_score = chain_consistency_statistic(base_chain)
    perturbed_score = chain_consistency_statistic(perturbed_chain)

    print("Consistency without regularisation:", perturbed_score)
    print("Consistency with regularisation:", base_score)
    print("Excess penalty (Omega(d_chain^2)) proxy:", (perturbed_score - base_score) ** 2)


if __name__ == "__main__":
    main()
