"""A Gaussian simulator with explicit, named inference parameters."""
from __future__ import annotations

import re
import numpy as np
from nnpd.nre.distributions import Parameter, Simulator


class GaussianProblem(Simulator):
    def __init__(self, settings: dict):
        self.settings = settings
        dimension = settings["dimension"]
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1:
            raise ValueError("problem.dimension must be a positive integer.")
        self.observation_dim = dimension
        all_names = [f"{kind}:{j}" for kind in ("mean", "std") for j in range(dimension)]
        names = []
        for selector in settings["infer"]:
            if selector in {"mean:*", "std:*"}:
                names.extend(f"{selector[:-1]}{j}" for j in range(dimension))
            elif isinstance(selector, str) and re.fullmatch(r"(mean|std):\d+", selector):
                names.append(selector)
            else:
                raise ValueError(f"Invalid inference selector {selector!r}.")
        if not names or len(names) != len(set(names)) or set(names) - set(all_names):
            raise ValueError("Infer a nonempty, unique, in-range set of named parameters.")
        if set(settings["fixed"]) - set(all_names) or set(settings["overrides"]) - set(all_names):
            raise ValueError("A fixed/overridden parameter name is out of range.")
        if set(settings["fixed"]) & set(names):
            raise ValueError("A parameter cannot be both fixed and inferred.")
        self.names = names
        self.indices = [(name.split(":")[0], int(name.split(":")[1])) for name in names]
        self._parameters = []
        for name, (kind, _) in zip(names, self.indices):
            options = {**settings["parameters"][kind], **settings["overrides"].get(name, {})}
            parameter = Parameter(name, options["low"], options["high"], options)
            if kind == "std" and parameter.low <= 0:
                raise ValueError("Inferred standard deviations need strictly positive bounds.")
            self._parameters.append(parameter)
        self.fixed_mean = np.full(dimension, settings["fixed_mean"], dtype=np.float64)
        self.fixed_std = np.full(dimension, settings["fixed_std"], dtype=np.float64)
        for name, value in settings["fixed"].items():
            kind, index = name.split(":")
            (self.fixed_mean if kind == "mean" else self.fixed_std)[int(index)] = value
        if not np.isfinite(self.fixed_mean).all() or not np.isfinite(self.fixed_std).all():
            raise ValueError("Fixed Gaussian parameters must be finite.")
        if (self.fixed_std <= 0).any():
            raise ValueError("Fixed standard deviations must be positive.")

    @property
    def parameters(self):
        return tuple(self._parameters)

    def unpack(self, theta):
        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim < 1 or theta.shape[-1] != len(self.parameters):
            raise ValueError("theta's last axis must match the ordered inferred parameters.")
        if not np.isfinite(theta).all():
            raise ValueError("theta must be finite.")
        shape = (*theta.shape[:-1], self.observation_dim)
        mean = np.broadcast_to(self.fixed_mean, shape).copy()
        std = np.broadcast_to(self.fixed_std, shape).copy()
        for column, (kind, axis) in enumerate(self.indices):
            (mean if kind == "mean" else std)[..., axis] = theta[..., column]
        if (std <= 0).any():
            raise ValueError("Gaussian standard deviations must be positive.")
        return mean, std

    def sample(self, theta, rng):
        mean, std = self.unpack(theta)
        return rng.normal(mean, std).astype(np.float32)

    def log_likelihood(self, x, theta):
        mean, std = self.unpack(theta)
        x = np.asarray(x, dtype=np.float64)
        return (-0.5 * ((x - mean) / std) ** 2 - np.log(std) - 0.5 * np.log(2 * np.pi)).sum(-1)
