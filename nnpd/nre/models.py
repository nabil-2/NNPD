"""Interchangeable classifiers returning logits, not sigmoid probabilities."""
from __future__ import annotations

import torch
from torch import nn


def activation(name: str) -> nn.Module:
    choices = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}
    if name not in choices:
        raise ValueError(f"Unknown activation {name!r}; choose {sorted(choices)}.")
    return choices[name]()


class MLP(nn.Module):
    def __init__(self, inputs: int, hidden: list[int], nonlinearity: str = "relu", dropout: float = 0):
        super().__init__()
        layers = []
        for width in hidden:
            layers.extend((nn.Linear(inputs, width), activation(nonlinearity)))
            if dropout:
                layers.append(nn.Dropout(dropout))
            inputs = width
        layers.append(nn.Linear(inputs, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs).squeeze(-1)


class ResidualBlock(nn.Module):
    def __init__(self, width: int, nonlinearity: str, dropout: float):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(width, width), activation(nonlinearity),
                                  nn.Dropout(dropout), nn.Linear(width, width))
        self.activation = activation(nonlinearity)

    def forward(self, x):
        return self.activation(x + self.body(x))


class ResidualMLP(nn.Module):
    def __init__(self, inputs: int, width: int, blocks: int, nonlinearity: str, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(inputs, width), activation(nonlinearity),
                                   *(ResidualBlock(width, nonlinearity, dropout) for _ in range(blocks)),
                                   nn.Linear(width, 1))

    def forward(self, inputs):
        return self.layers(inputs).squeeze(-1)


def make_mlp(inputs, definition):
    return MLP(inputs, definition["hidden"], definition["activation"], definition["dropout"])


def make_residual(inputs, definition):
    return ResidualMLP(inputs, definition["width"], definition["blocks"],
                       definition["activation"], definition["dropout"])


# Add a factory here to make another model selectable in settings.py.
NETWORKS = {"mlp": make_mlp, "residual": make_residual}


def build_network(inputs: int, definition: dict) -> nn.Module:
    try:
        factory = NETWORKS[definition["kind"]]
    except KeyError as error:
        raise ValueError(f"Unknown model kind: {definition['kind']!r}") from error
    return factory(inputs, definition)
