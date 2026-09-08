"""NNPD: a small experiment framework and optional NRE building blocks."""
from .core.api import Experiment, Metric, Plot, Product, load_experiment
from .core.config import Choice, expand_sweep
from .core.runner import execute, plan, restore

__all__ = ["Choice", "Experiment", "Metric", "Plot", "Product", "load_experiment", "execute", "expand_sweep", "plan", "restore"]
__version__ = "0.1.0"
