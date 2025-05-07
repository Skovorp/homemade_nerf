import torch
import torch.nn as nn


class Nerf(nn.Module):
    def __init__(self, num_layers, d_model, **kwargs):
        super().__init__()
        layers = []
        layers.append(nn.Linear(3, d_model))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(d_model, d_model))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(d_model))

        self.stack = nn.Sequential(*layers)

        # Final output layer: maps to 4 values (RGB + sigma)
        self.out_layer = nn.Linear(d_model, 4)

    def forward(self, x):
        x = x / 4
        x = self.stack(x)
        return self.out_layer(x)