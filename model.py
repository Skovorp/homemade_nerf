import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, l, **kwargs):
        super().__init__()
        weights = 2 ** torch.arange(l)
        weights = weights.reshape(1, l, 1)
        self.weights = nn.Parameter(weights, requires_grad=False)
    
    def forward(self, x):
        x = x.unsqueeze(1) # bs, 1, 3
        x = self.weights * x
        x = torch.cat([torch.sin(x), torch.cos(x)], 1)
        x = x.flatten(1)
        return x
        

class Nerf(nn.Module):
    def __init__(self, num_layers, d_model, pe_l, **kwargs):
        super().__init__()
        layers = []
        layers.append(nn.Linear(6 * pe_l, d_model))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(d_model, d_model))
            layers.append(nn.ReLU())
            # layers.append(nn.BatchNorm1d(d_model))
        self.pe = PositionalEncoding(pe_l)
        self.stack = nn.Sequential(*layers)
        self.out_layer = nn.Linear(d_model, 4)
        
        self.out_layer.bias.data.fill_(0.)
        self.out_layer.weight.data *= 1e-4 

    def forward(self, x):
        x = x / 4
        x = self.pe(x)
        x = self.stack(x)
        return self.out_layer(x)