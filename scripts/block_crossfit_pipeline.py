"""Block cross-fitting pipeline (Section R5)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .common import (
    block_partition,
    l2_error,
    make_anisotropic_sampler,
    set_seed,
    smolyak_cpwl_approximation,
)


@dataclass
class BlockReport:
    block_id: int
    width: int
    error: float


def run_block_crossfit(n_samples: int, n_blocks: int) -> List[BlockReport]:
    beta = [1.0, 1.5]
    weights = [1.0, 0.6]
    fun, sampler = make_anisotropic_sampler(beta, weights, noise=0.01)
    samples = sampler(n_samples)
    blocks = block_partition(n_samples, n_blocks)

    reports: List[BlockReport] = []
    for block_id, sl in enumerate(blocks):
        val_samples = samples[sl]
        approx, width = smolyak_cpwl_approximation(fun, [4, 5])
        error = l2_error(fun, approx, lambda n: val_samples, n_samples=len(val_samples))
        reports.append(BlockReport(block_id=block_id, width=width, error=error))
    return reports


def main() -> None:
    set_seed(19)
    reports = run_block_crossfit(n_samples=600, n_blocks=5)
    for report in reports:
        print(f"Block {report.block_id}: width={report.width}, error={report.error:.4f}")


if __name__ == "__main__":
    main()
