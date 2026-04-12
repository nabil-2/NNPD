from __future__ import annotations

import torch
import torch.nn as nn


class BinaryClassifier(nn.Module):
    def __init__(self, config):
        super(BinaryClassifier, self).__init__()
        n_inputs = config["classifier"]["n_inputs"] + config["data"]["n_parameters"]
        input_layer = [
            nn.Linear(n_inputs, config["classifier"]["n_units"]),
            nn.ReLU(),
        ]
        hidden_layers = []
        for _ in range(config["classifier"]["n_hidden_layers"]):
            hidden_layers += [
                nn.Linear(
                    config["classifier"]["n_units"],
                    config["classifier"]["n_units"],
                ),
                nn.ReLU(),
            ]
        output_layer = [nn.Linear(config["classifier"]["n_units"], 1), nn.Sigmoid()]
        self.model = nn.Sequential(*(input_layer + hidden_layers + output_layer))

    def forward(self, x):
        return self.model(x)

