"""Distribution interfaces, independent bounded priors, and exact samplers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math

import numpy as np
from scipy.stats import truncnorm


@dataclass(frozen=True)
class Parameter:
    name: str
    low: float
    high: float
    options: dict

    def __post_init__(self):
        if not np.isfinite([self.low, self.high]).all() or self.low >= self.high:
            raise ValueError(f"Invalid bounds for {self.name}: {self.low}, {self.high}")


def grid_axis(low: float, high: float, step: float) -> np.ndarray:
    if not np.isfinite(step) or step <= 0 or not np.isfinite([low, high]).all() or high <= low:
        raise ValueError("Grid bounds must increase and grid step must be positive and finite.")
    count = int(np.floor((high - low) / step + 1e-10)) + 1
    if count > 10_000_000:
        raise ValueError("A single grid axis would exceed ten million points.")
    return low + step * np.arange(count, dtype=np.float64)


class Simulator(ABC):
    parameters: tuple[Parameter, ...]
    observation_dim: int

    @abstractmethod
    def sample(self, theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """One observation per parameter row; result shape (..., observation_dim)."""
        ...

    def log_likelihood(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Optional exact reference, not used by the NRE training objective."""
        raise NotImplementedError("This simulator has no tractable likelihood.")


class Prior(ABC):
    parameters: tuple[Parameter, ...]

    @abstractmethod
    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        """Draw shape (size, number_of_inferred_parameters)."""
        ...

    @abstractmethod
    def log_prob(self, theta: np.ndarray) -> np.ndarray:
        """Log density or log mass with respect to the declared support measure."""
        ...

    @property
    def support_axes(self) -> tuple[np.ndarray | None, ...]:
        """None means continuous; an array means discrete counting measure."""
        return (None,) * len(self.parameters)

    @property
    def measure(self) -> str:
        count = sum(axis is not None for axis in self.support_axes)
        return "continuous" if count == 0 else "discrete" if count == len(self.parameters) else "mixed"


def _log_exprel(k: float) -> float:
    if abs(k) < 1e-6:
        return math.log1p(k / 2 + k * k / 6 + k**3 / 24)
    if k > 0:
        return k + math.log1p(-math.exp(-k)) - math.log(k)
    return math.log(-math.expm1(k)) - math.log(-k)


class IndependentPrior(Prior):
    """Uniform, truncated normal, bounded signed exponential, or grid per axis.

    A negative exponential rate is valid on a finite box; it is not an unbounded
    exponential distribution. Grid log_prob returns a mass, never a density.
    'by_parameter' may override the default family for particular coordinates.
    """
    families = {"uniform", "normal", "exponential", "grid"}

    def __init__(self, parameters: tuple[Parameter, ...], definition: dict):
        self.parameters = parameters
        overrides = definition.get("by_parameter", {})
        unknown = set(overrides) - {p.name for p in parameters}
        if unknown:
            raise ValueError(f"Prior overrides refer to uninferred parameters: {sorted(unknown)}")
        self.kinds = tuple(overrides.get(p.name, definition["family"]) for p in parameters)
        axes = []
        for parameter, kind in zip(parameters, self.kinds):
            if kind not in self.families:
                raise ValueError(f"Unknown prior family {kind!r}.")
            options = parameter.options
            if kind == "normal":
                if not np.isfinite([options["normal_mean"], options["normal_std"]]).all() or options["normal_std"] <= 0:
                    raise ValueError("Normal prior location must be finite and scale positive.")
            if kind == "exponential":
                if not np.isfinite(options["exponential_rate"] * (parameter.high - parameter.low)):
                    raise ValueError("Exponential rate times support width must be finite.")
            axes.append(grid_axis(parameter.low, parameter.high, options["grid_step"])
                        if kind == "grid" else None)
        self._axes = tuple(axes)

    @property
    def support_axes(self):
        return self._axes

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        if not isinstance(size, (int, np.integer)) or size < 0:
            raise ValueError("Sample size must be a nonnegative integer.")
        result = np.empty((size, len(self.parameters)), dtype=np.float64)
        for j, (p, kind, axis) in enumerate(zip(self.parameters, self.kinds, self._axes)):
            u = rng.random(size)
            if kind == "grid":
                result[:, j] = axis[np.minimum((u * len(axis)).astype(int), len(axis) - 1)]
            elif kind == "normal":
                loc, scale = p.options["normal_mean"], p.options["normal_std"]
                result[:, j] = truncnorm.ppf(u, (p.low - loc) / scale, (p.high - loc) / scale,
                                            loc=loc, scale=scale)
            elif kind == "exponential":
                k = -p.options["exponential_rate"] * (p.high - p.low)
                if abs(k) < 1e-8:
                    result[:, j] = p.low + u * (p.high - p.low)
                else:
                    with np.errstate(divide="ignore"):
                        fraction = np.logaddexp(np.log1p(-u), np.log(u) + k) / k
                    result[:, j] = p.low + (p.high - p.low) * fraction
            else:
                result[:, j] = p.low + u * (p.high - p.low)
        if not np.isfinite(result).all():
            raise FloatingPointError("Prior sampler returned nonfinite values.")
        return result.astype(np.float32)

    def log_prob(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim < 1 or theta.shape[-1] != len(self.parameters):
            raise ValueError("Prior input has the wrong parameter dimension.")
        result = np.zeros(theta.shape[:-1], dtype=np.float64)
        for j, (p, kind, axis) in enumerate(zip(self.parameters, self.kinds, self._axes)):
            values = theta[..., j]
            valid = np.isfinite(values) & (values >= p.low - 1e-7) & (values <= p.high + 1e-7)
            if kind == "grid":
                index = np.rint((values - p.low) / p.options["grid_step"])
                clipped = np.clip(np.nan_to_num(index), 0, len(axis) - 1).astype(np.int64)
                valid &= (index >= 0) & (index < len(axis))
                valid &= np.isclose(values, axis[clipped], rtol=0,
                                    atol=max(1e-6, p.options["grid_step"] * 1e-4))
                log_value = -math.log(len(axis))
            elif kind == "normal":
                loc, scale = p.options["normal_mean"], p.options["normal_std"]
                log_value = truncnorm.logpdf(values, (p.low - loc) / scale, (p.high - loc) / scale,
                                            loc=loc, scale=scale)
            elif kind == "exponential":
                k = -p.options["exponential_rate"] * (p.high - p.low)
                log_value = k * (values - p.low) / (p.high - p.low) - math.log(p.high - p.low) - _log_exprel(k)
            else:
                log_value = -math.log(p.high - p.low)
            result += np.where(valid, log_value, -np.inf)
        return result
