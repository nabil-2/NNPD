"""NNPD: a small experiment framework and optional NRE building blocks."""
from .core.api import Analysis, CohortPlot, Experiment, Metric, Plot, Product, load_analysis, load_experiment
from .core.settings import Choice, expand_sweep, load_settings
from .core.runner import execute, plan, restore

__all__ = ["Analysis", "Choice", "CohortPlot", "Experiment", "Metric", "Plot", "Product", "execute",
           "expand_sweep", "load_analysis", "load_experiment", "load_settings", "plan", "restore"]
__version__ = "0.1.0"
