"""Toy instability as epsilon -> 0 (Appendix G2)."""
from __future__ import annotations

from .common import set_seed


def main() -> None:
    set_seed(29)
    epsilons = [10 ** (-2 + i * (1.5 / 7)) for i in range(8)]
    condition_numbers = []
    for eps in epsilons:
        largest = 2.0 - eps
        smallest = max(eps, 1e-9)
        condition_numbers.append(largest / smallest)
    print("epsilon condition_number")
    for eps, cond in zip(epsilons, condition_numbers):
        print(f"{eps: .3e} {cond: .3f}")


if __name__ == "__main__":
    main()
