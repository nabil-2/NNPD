"""The extension surface. No Gaussian settings or metric names live here."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import inspect
from importlib.metadata import version as package_version
from importlib import import_module
import platform
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch

from .config import digest
from .storage import Artifact, Store, Writer, file_hash
from .runtime import Runtime


def implementation_hash(functions: tuple[Callable, ...], version: str) -> dict:
    """Hash complete source modules, not just the registration function.

    Declare helper modules in Product.sources or Experiment.sources. External
    simulators/data releases should be identified explicitly in the recipe/version.
    """
    files, entrypoints = {}, []
    for function in functions:
        path = inspect.getsourcefile(function)
        if path is None:
            raise ValueError(f"Cannot fingerprint {function}; wrap it in a Python module.")
        files[f"{function.__module__}:{Path(path).name}"] = file_hash(Path(path))
        entrypoints.append({"name": f"{function.__module__}.{function.__qualname__}",
                            "line": getattr(getattr(function, "__code__", None), "co_firstlineno", None)})
    return {"version": version, "modules": files, "entrypoints": entrypoints,
            "environment": {"python": platform.python_version(),
                            **{name: package_version(name) for name in ("numpy", "scipy", "torch")}}}


@dataclass(frozen=True)
class Product:
    """A reusable dataset; dependencies are resolved and cached before build()."""
    build: Callable[["Context", Mapping[str, Artifact], Writer], dict]
    needs: tuple[str, ...] = ()
    settings: Callable[["Context"], dict] = lambda context: context.config
    sources: tuple[Callable, ...] = ()
    version: str = "1"


@dataclass(frozen=True)
class Metric:
    compute: Callable[["Context", Mapping[str, Artifact]], Any]
    needs: tuple[str, ...] = ()


@dataclass(frozen=True)
class Plot:
    # Return {filename_stem: matplotlib.figure.Figure}. One hook may make many plots.
    draw: Callable[["Context", Mapping[str, Artifact]], Mapping[str, Any]]
    needs: tuple[str, ...] = ()


class Experiment(ABC):
    """Implement the scientific decisions; the runner owns execution and files."""
    name = "experiment"
    version = "1"

    def members(self, config: dict) -> list[dict]:
        """A fixed comparison cohort within each sweep configuration."""
        return [{}]

    def validate(self, config: dict) -> None:
        """Fail before any sampling or training, including for invalid alternatives."""

    def estimate(self, config: dict, member: dict) -> dict:
        return {}

    @abstractmethod
    def make_problem(self, config: dict) -> Any: ...

    @abstractmethod
    def make_prior(self, config: dict, member: dict, problem: Any) -> Any: ...

    @abstractmethod
    def sample_training(self, context: "Context", writer: Writer) -> dict: ...

    @abstractmethod
    def build_model(self, context: "Context") -> torch.nn.Module: ...

    @abstractmethod
    def train(self, context: "Context", model: torch.nn.Module) -> dict:
        """Modify model in place; return JSON-serializable training history."""
        ...

    def signature(self, stage: str, context: "Context") -> dict:
        """Override for cross-run sharing; conservative default uses all settings."""
        return {"config": context.config, "member": context.member}

    def sources(self, stage: str) -> tuple[Callable, ...]:
        """Include modules implementing transitive scientific dependencies."""
        return (type(self),)

    def model_seed(self, context: "Context") -> int:
        return context.seed("model")

    def products(self) -> dict[str, Product]:
        return {}

    def metrics(self) -> dict[str, Metric]:
        return {}

    def plots(self) -> dict[str, Plot]:
        return {}


def load_experiment(specification: str) -> Experiment:
    """Instantiate a trusted module:class extension configured in settings.py."""
    module, name = specification.split(":", 1)
    experiment = getattr(import_module(module), name)()
    if not isinstance(experiment, Experiment):
        raise TypeError("The configured application must subclass Experiment.")
    return experiment


@dataclass
class Context:
    experiment: Experiment
    config: dict
    member: dict
    store: Store
    runtime: Runtime
    run_dir: Path
    problem: Any = field(init=False)
    prior: Any = field(init=False)
    artifacts: dict[str, Artifact] = field(default_factory=dict)
    _model: torch.nn.Module | None = field(default=None, repr=False)
    _active: list[str] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.problem = self.experiment.make_problem(self.config)
        self.prior = self.experiment.make_prior(self.config, self.member, self.problem)

    def seed(self, namespace: str, *, base: int | None = None, member: bool = True) -> int:
        base = int(self.config.get("seed", 0)) if base is None else int(base)
        payload = {"seed": base, "namespace": namespace,
                   "member": self.member if member else {}}
        return int(digest(payload)[:8], 16)

    def rng(self, namespace: str, *, base: int | None = None, member: bool = True):
        return np.random.default_rng(self.seed(namespace, base=base, member=member))

    @property
    def model(self) -> torch.nn.Module:
        if self._model is None:
            if "model" not in self.artifacts:
                raise RuntimeError("This context does not have a trained model.")
            model = self.experiment.build_model(self)
            model.load_state_dict(self.artifacts["model"].state_dict())
            self._model = model.to(self.runtime.device).eval()
        return self._model

    def require(self, name: str) -> Artifact:
        """Get or build a named dataset. Cycles and unknown dependencies fail clearly."""
        if name in self.artifacts:
            return self.artifacts[name]
        if name in self._active:
            raise ValueError("Dataset dependency cycle: " + " -> ".join([*self._active, name]))
        registry = self.experiment.products()
        if name not in registry:
            raise KeyError(f"Unknown or unavailable dataset {name!r}.")
        self._active.append(name)
        try:
            product = registry[name]
            dependencies = {dependency: self.require(dependency) for dependency in product.needs}
            recipe = {"experiment": self.experiment.name, "name": name,
                      "settings": product.settings(self),
                      "dependencies": {key: artifact.key for key, artifact in dependencies.items()},
                      "implementation": implementation_hash((product.build, *product.sources),
                                                              product.version)}
            artifact = self.store.get(name, digest(recipe), recipe,
                                      lambda writer: product.build(self, dependencies, writer))
            self.artifacts[name] = artifact
            return artifact
        finally:
            self._active.pop()

    def dependencies(self, names: tuple[str, ...]) -> dict[str, Artifact]:
        return {name: self.require(name) for name in names}
