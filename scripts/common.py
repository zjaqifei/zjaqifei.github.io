"""Common utilities for the simulation scripts (pure Python implementation)."""
from __future__ import annotations

import bisect
import math
import random
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple

Vector = List[float]
Matrix = List[List[float]]


def set_seed(seed: int = 0) -> None:
    """Seed the ``random`` module for deterministic simulations."""

    random.seed(seed)


@dataclass
class AnisotropicFunction:
    """Callable anisotropic test function used across the scripts."""

    beta: Sequence[float]
    weights: Sequence[float]
    noise: float = 0.0

    def __post_init__(self) -> None:
        if len(self.beta) != len(self.weights):
            raise ValueError("beta and weights must have the same dimension")

    def _evaluate_point(self, point: Sequence[float]) -> float:
        value = 0.0
        for beta_j, weight_j, coord in zip(self.beta, self.weights, point):
            value += weight_j * math.sin((1.0 + beta_j) * math.pi * coord)
            value += 0.5 * weight_j * math.cos((1.0 + 0.5 * beta_j) * math.pi * coord * coord)
        if self.noise:
            value += random.gauss(0.0, self.noise)
        return value

    def __call__(self, x: Sequence[Sequence[float]] | Sequence[float]) -> float | Vector:
        if not x:
            return 0.0
        first = x[0] if isinstance(x, Sequence) else x
        if isinstance(first, (list, tuple)):
            return [self._evaluate_point(point) for point in x]  # type: ignore[arg-type]
        return self._evaluate_point(x)  # type: ignore[arg-type]


def make_anisotropic_sampler(
    beta: Sequence[float], weights: Sequence[float], noise: float = 0.0
) -> Tuple[Callable[[Sequence[float] | Sequence[Sequence[float]]], float | Vector], Callable[[int], List[Vector]]]:
    """Return a synthetic anisotropic function and a sampler on the unit cube."""

    fun = AnisotropicFunction(beta=beta, weights=weights, noise=noise)

    def sampler(n: int) -> List[Vector]:
        return [[random.random() for _ in range(len(beta))] for _ in range(n)]

    return fun, sampler


def _linspace(start: float, stop: float, num: int) -> List[float]:
    if num <= 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]


def smolyak_cpwl_approximation(
    fun: Callable[[Sequence[float]], float], grid_sizes: Sequence[int]
) -> Tuple[Callable[[Sequence[float] | Sequence[Sequence[float]]], float | Vector], int]:
    """Construct a CPWL approximation on an anisotropic grid."""

    grids = [_linspace(0.0, 1.0, m) for m in grid_sizes]
    index_ranges = [range(len(axis)) for axis in grids]
    values: dict[Tuple[int, ...], float] = {}
    for index in _cartesian_product(index_ranges):
        coords = [grids[dim][idx] for dim, idx in enumerate(index)]
        values[index] = float(fun(coords))

    def approx_point(point: Sequence[float]) -> float:
        lower_indices: List[int] = []
        upper_indices: List[int] = []
        for axis in grids:
            pos = bisect.bisect_right(axis, point[len(lower_indices)]) - 1
            pos = max(0, min(pos, len(axis) - 2))
            lower_indices.append(pos)
            upper_indices.append(pos + 1)
        corners = []
        for bits in _cartesian_product([[0, 1] for _ in grids]):
            index = tuple(
                (upper if bit else lower)
                for bit, lower, upper in zip(bits, lower_indices, upper_indices)
            )
            corners.append(values[index])
        return sum(corners) / len(corners)

    def approx(x: Sequence[Sequence[float]] | Sequence[float]) -> float | Vector:
        if not x:
            return 0.0
        first = x[0] if isinstance(x, Sequence) else x
        if isinstance(first, (list, tuple)):
            return [approx_point(point) for point in x]  # type: ignore[arg-type]
        return approx_point(x)  # type: ignore[arg-type]

    linear_regions = 1
    for axis in grids:
        linear_regions *= max(1, len(axis) - 1)
    return approx, linear_regions


def l2_error(
    fun: Callable[[Sequence[float] | Sequence[Sequence[float]]], float | Vector],
    approx: Callable[[Sequence[float] | Sequence[Sequence[float]]], float | Vector],
    sampler: Callable[[int], Sequence[Vector]],
    n_samples: int = 2048,
) -> float:
    """Empirical :math:`L_2` error using Monte Carlo samples."""

    samples = sampler(n_samples)
    diffs = []
    for sample in samples:
        target = float(fun(sample))
        approx_val = float(approx(sample))
        diffs.append((target - approx_val) ** 2)
    return math.sqrt(sum(diffs) / max(len(diffs), 1))


def synthetic_covariance(dim: int, anisotropy: float = 3.0) -> Matrix:
    diag = _linspace(1.0, anisotropy, dim)
    return [[diag[i] if i == j else 0.0 for j in range(dim)] for i in range(dim)]


def _cholesky(matrix: Matrix) -> Matrix:
    n = len(matrix)
    chol = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(chol[i][k] * chol[j][k] for k in range(j))
            if i == j:
                val = matrix[i][i] - s
                chol[i][j] = math.sqrt(max(val, 1e-12))
            else:
                chol[i][j] = (matrix[i][j] - s) / chol[j][j]
    return chol


def _matvec(matrix: Matrix, vec: Sequence[float]) -> Vector:
    return [sum(row[k] * vec[k] for k in range(len(vec))) for row in matrix]


def gaussian_chain_samples(n: int, dim: int, steps: int, noise: float = 0.1) -> List[List[Vector]]:
    """Generate synthetic state trajectories for chain-consistency diagnostics."""

    cov = synthetic_covariance(dim)
    chol = _cholesky(cov)
    states: List[Vector] = []
    current = [0.0 for _ in range(dim)]
    for _ in range(steps):
        gaussian = [random.gauss(0.0, 1.0) for _ in range(dim)]
        innovation = _matvec(chol, gaussian)
        noise_vec = [random.gauss(0.0, noise) for _ in range(dim)]
        current = [0.8 * c + inc + nv for c, inc, nv in zip(current, innovation, noise_vec)]
        states.append(current[:])
    batches: List[List[Vector]] = []
    for _ in range(steps):
        batches.append([random.choice(states)[:] for _ in range(n)])
    return batches


def prox_projection(x: Sequence[Sequence[float]] | Sequence[float], lower: float = -1.0, upper: float = 1.0):
    """Apply a box projection element-wise."""

    if isinstance(x, (list, tuple)) and x and isinstance(x[0], (list, tuple)):
        return [prox_projection(sub, lower, upper) for sub in x]  # type: ignore[arg-type]
    return [min(max(value, lower), upper) for value in x]  # type: ignore[arg-type]


def chain_consistency_statistic(samples: Sequence[Sequence[Vector]]) -> float:
    sigma = 0.5
    total = 0.0
    count = 0
    for left, right in zip(samples[:-1], samples[1:]):
        for x in left:
            for y in right:
                diff = sum((a - b) ** 2 for a, b in zip(x, y))
                total += math.exp(-diff / (2.0 * sigma * sigma))
                count += 1
    return total / max(count, 1)


def empirical_cvar(samples: Sequence[float], alpha: float = 0.9) -> float:
    sorted_samples = sorted(samples)
    index = max(0, int(math.ceil(alpha * len(sorted_samples))) - 1)
    tail = sorted_samples[index:]
    if not tail:
        return sorted_samples[-1] if sorted_samples else 0.0
    return sum(tail) / len(tail)


def martingale_penalty(paths: Sequence[Sequence[Vector]]) -> float:
    penalty = 0.0
    count = 0
    for path in paths:
        for prev, curr in zip(path[:-1], path[1:]):
            diff = sum((a - b) ** 2 for a, b in zip(curr, prev))
            penalty += diff
            count += 1
    return penalty / max(count, 1)


def r_linear_rate(mu: float, lipschitz: float, k: int) -> float:
    mu = max(mu, 1e-12)
    return (1.0 - mu / max(lipschitz, 1e-12)) ** k


def finite_difference_gradient(
    fun: Callable[[Sequence[float]], float], x: Sequence[float], step: float = 1e-2
) -> Vector:
    grad = []
    base = float(fun(x))
    for idx in range(len(x)):
        perturbed = list(x)
        perturbed[idx] += step
        grad.append((float(fun(perturbed)) - base) / step)
    return grad


def simulate_bridge_statistics(n_paths: int, dim: int) -> Tuple[float, float]:
    cov = synthetic_covariance(dim)
    perturb = [[random.gauss(0.0, 0.05) for _ in range(dim)] for _ in range(dim)]
    empirical = [[cov[i][j] + 0.5 * (perturb[i][j] + perturb[j][i]) for j in range(dim)] for i in range(dim)]
    diff = [[empirical[i][j] - cov[i][j] for j in range(dim)] for i in range(dim)]
    frob = math.sqrt(sum(val * val for row in diff for val in row))
    trace = sum(cov[i][i] for i in range(dim))
    upper = frob * frob + trace / max(dim, 1)
    lower = 0.1 * frob + 0.05 * dim / max(n_paths, 1)
    return upper, lower


def block_partition(n: int, n_blocks: int) -> List[slice]:
    block_size = int(math.ceil(n / n_blocks))
    slices: List[slice] = []
    for b in range(n_blocks):
        start = b * block_size
        stop = min((b + 1) * block_size, n)
        if start >= stop:
            break
        slices.append(slice(start, stop))
    return slices


def _cartesian_product(ranges: Sequence[Iterable[int]]) -> List[Tuple[int, ...]]:
    ranges = list(ranges)
    if not ranges:
        return [()]
    first, *rest = ranges
    tail = _cartesian_product(rest)
    return [(item, *suffix) for item in first for suffix in tail]
