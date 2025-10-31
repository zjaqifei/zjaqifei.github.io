"""Simulated bridge upper and lower bounds (Section 3)."""
from __future__ import annotations

from .common import simulate_bridge_statistics, set_seed


def main() -> None:
    set_seed(31)
    upper, lower = simulate_bridge_statistics(n_paths=200, dim=5)
    print("Bridge upper bound:", upper)
    print("Bridge lower bound:", lower)


if __name__ == "__main__":
    main()
