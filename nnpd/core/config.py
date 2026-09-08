"""Explicit one-factor-at-a-time sweeps. Ordinary lists are literal values."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import runpy
from typing import Any


def canonical(value: Any) -> str:
    """Stable JSON; reject NaN, infinity, callables, and ambiguous custom objects."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()[:24]


@dataclass(frozen=True)
class Choice:
    """An atomic sweep knob: Choice([baseline, alternative, ...]).

    Values may themselves be lists or dictionaries (e.g. complete architectures).
    Nesting Choice inside an option is deliberately unsupported.
    """
    values: list[Any] | tuple[Any, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.values, (list, tuple)) or not self.values:
            raise ValueError("Choice needs a non-empty list or tuple of options.")
        for value in self.values:
            canonical(value)


@dataclass(frozen=True)
class Variant:
    config: dict
    changed: str | None
    value: Any = None

    @property
    def id(self) -> str:
        return digest(self.config)


def at(config: dict, path: str) -> Any:
    value: Any = config
    for name in path.split("."):
        value = value[name]
    return value


def select(config: dict, *paths: str) -> dict:
    return {path: at(config, path) for path in paths}


def expand_sweep(config: dict) -> list[Variant]:
    """Baseline once, then one changed knob per run; never a Cartesian product.

    Dictionary insertion order determines presentation order, not run identity.
    Duplicate resolved configurations are removed, including repeated baselines.
    """
    if not isinstance(config, dict):
        raise TypeError("The top-level configuration must be a dictionary.")
    knobs: list[tuple[tuple[str, ...], Choice]] = []

    def baseline(value: Any, path: tuple[str, ...]) -> Any:
        if isinstance(value, Choice):
            knobs.append((path, value))
            return deepcopy(value.values[0])
        if isinstance(value, dict):
            if not all(isinstance(key, str) and key and "." not in key for key in value):
                raise ValueError("Configuration dictionary keys must be nonempty strings without dots.")
            return {key: baseline(item, (*path, key)) for key, item in value.items()}
        canonical(value)  # A Choice hidden inside a literal list is an error.
        return deepcopy(value)

    base = baseline(config, ())
    if not isinstance(base, dict):
        raise TypeError("The top-level configuration must be a dictionary.")
    result = [Variant(base, None)]
    seen = {canonical(base)}
    for path, knob in knobs:
        for option in knob.values[1:]:
            candidate = deepcopy(base)
            parent = candidate
            for name in path[:-1]:
                parent = parent[name]
            parent[path[-1]] = deepcopy(option)
            key = canonical(candidate)
            if key not in seen:
                seen.add(key)
                result.append(Variant(candidate, ".".join(path), deepcopy(option)))
    return result


def load_settings(path: str | Path, profile: str | None = None) -> dict:
    """Load a *trusted* Python settings file; this executes Python, not a sandbox."""
    path = Path(path).resolve()
    namespace = runpy.run_path(str(path))
    return namespace["make_config"](profile or namespace["DEFAULT_PROFILE"])
