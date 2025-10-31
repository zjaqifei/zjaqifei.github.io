"""Finite-difference Greeks stability under prox approximation."""
from __future__ import annotations

from statistics import variance

from .common import finite_difference_gradient, make_anisotropic_sampler, prox_projection, set_seed


def _rescale_samples(samples):
    return [[value * 2.0 - 1.0 for value in sample] for sample in samples]


def flatten(values):
    return [item for sublist in values for item in sublist]


def main() -> None:
    set_seed(37)
    beta = [1.0, 2.0]
    weights = [1.0, 0.5]
    fun, sampler = make_anisotropic_sampler(beta, weights)
    samples = _rescale_samples(sampler(64))
    projected = prox_projection(samples, lower=-0.9, upper=0.9)

    grads_before = [finite_difference_gradient(fun, x) for x in samples]
    grads_after = [finite_difference_gradient(fun, x) for x in projected]

    var_before = variance(flatten(grads_before)) if len(grads_before) > 1 else 0.0
    var_after = variance(flatten(grads_after)) if len(grads_after) > 1 else 0.0

    print("Variance of Greeks before prox:", var_before)
    print("Variance of Greeks after prox:", var_after)
    baseline = var_before if var_before > 1e-12 else 1e-12
    print("Stability ratio:", var_after / baseline)


if __name__ == "__main__":
    main()
