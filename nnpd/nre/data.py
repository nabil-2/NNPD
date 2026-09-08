"""The balanced joint-vs-product classification experiment (paper Eq. 3)."""
import numpy as np

from .distributions import Prior, Simulator


def balanced_pairs(problem: Simulator, prior: Prior, size: int, rng: np.random.Generator):
    if not isinstance(size, int) or size < 2 or size % 2:
        raise ValueError("Each split size must be a positive even integer for exact class balance.")
    half = size // 2
    a, b, c = np.split(prior.sample(3 * half, rng), 3)
    theta = np.concatenate((a, c))
    x = np.concatenate((problem.sample(b, rng), problem.sample(c, rng)))
    labels = np.concatenate((np.zeros(half), np.ones(half)))
    order = rng.permutation(size)
    return x[order].astype(np.float32), theta[order].astype(np.float32), labels[order].astype(np.float32)
