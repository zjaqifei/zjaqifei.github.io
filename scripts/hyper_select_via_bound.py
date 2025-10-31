"""Closed-form hyperparameter selection for the approximation stage."""
from __future__ import annotations

import random

from .common import set_seed


def select_smolyak_level(n_eff: int, beta_bar: float) -> int:
    return max(2, int(round(n_eff ** (1.0 / (2 * beta_bar + 1)))))


def select_rank(spectrum: list[float], tolerance: float = 0.9) -> int:
    total = sum(spectrum)
    cumulative = 0.0
    for idx, value in enumerate(sorted(spectrum, reverse=True), start=1):
        cumulative += value
        if cumulative / max(total, 1e-12) >= tolerance:
            return idx
    return len(spectrum)


def main() -> None:
    set_seed(13)
    n_eff = 500
    beta_bar = 1.2
    level = select_smolyak_level(n_eff, beta_bar)
    spectrum = sorted([random.random() for _ in range(30)], reverse=True)
    rank = select_rank(spectrum, tolerance=0.95)

    print("Selected Smolyak level:", level)
    print("Selected PCA rank:", rank)


if __name__ == "__main__":
    main()
