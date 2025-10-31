"""Generate zero-violation certificates for the prox projection (Claim C3)."""
from __future__ import annotations

import random

from .common import prox_projection, set_seed


def certificate(x, projection) -> float:
    return max(abs(a - b) for a, b in zip(x, projection))


def main() -> None:
    set_seed(5)
    samples = [[random.gauss(0.0, 1.0) * 2.0 for _ in range(5)] for _ in range(256)]
    projections = prox_projection(samples, lower=-1.0, upper=1.0)
    certificates = [certificate(x, y) for x, y in zip(samples, projections)]

    avg_cert = sum(certificates) / max(len(certificates), 1)
    max_cert = max(certificates) if certificates else 0.0
    unchanged = sum(
        1 for original, proj in zip(samples, projections) if all(abs(o - p) <= 1e-8 for o, p in zip(original, proj))
    )

    print("Average certificate:", avg_cert)
    print("Maximum certificate:", max_cert)
    print("Number of unchanged points (already feasible):", unchanged)


if __name__ == "__main__":
    main()
