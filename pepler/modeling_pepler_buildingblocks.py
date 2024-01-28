import copy
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from transformers.modeling_outputs import ModelOutput


@dataclass
class ContinuousModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None

@dataclass
class RecModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    aspect_rating: torch.FloatTensor = None
    overall_rating: torch.FloatTensor = None
    outputs: any = None


class OverallMLP(nn.Module):
    def __init__(self, aspect_num, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.relu = nn.LeakyReLU()
        self.regressor = nn.Linear(aspect_num, 1)
    def forward(self, aspect_score: torch.FloatTensor) -> torch.FloatTensor:
        return self.regressor(self.relu(aspect_score))


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MLP(nn.Module):
    def __init__(self, emsize, hidden_size=400, num_layers=2):
        super(MLP, self).__init__()
        self.first_layer = nn.Linear(emsize * 3, hidden_size)
        self.last_layer = nn.Linear(hidden_size, 1)
        layer = nn.Linear(hidden_size, hidden_size)
        self.layers = _get_clones(layer, num_layers)
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.first_layer.weight.data.uniform_(-initrange, initrange)
        self.first_layer.bias.data.zero_()
        self.last_layer.weight.data.uniform_(-initrange, initrange)
        self.last_layer.bias.data.zero_()
        for layer in self.layers:
            layer.weight.data.uniform_(-initrange, initrange)
            layer.bias.data.zero_()

    def forward(self, user, item, aspect):  # (batch_size, emsize)
        uia_cat = torch.cat([user, item, aspect], 1)  # (batch_size, emsize * 3)
        hidden = self.sigmoid(self.first_layer(uia_cat))  # (batch_size, hidden_size)
        for layer in self.layers:
            hidden = self.sigmoid(layer(hidden))  # (batch_size, hidden_size)
        rating = torch.squeeze(self.last_layer(hidden))  # (batch_size,)
        return rating


class WeightedMLP(nn.Module):
    # personalized attention weights to decide

    def __init__(self, emsize, hidden_size=400, num_layers=2):
        super(WeightedMLP, self).__init__()
        self.weight_layer = nn.Linear(emsize*3, 3)
        self.weight_relu = nn.LeakyReLU()
        self.first_layer = nn.Linear(emsize, hidden_size)
        layer = nn.Linear(hidden_size, hidden_size)
        self.layers = _get_clones(layer, num_layers)
        self.last_layer = nn.Linear(hidden_size, 1)

    def init_weights(self):
        initrange = 0.1
        self.weight_layer.weight.data.uniform_(-initrange, initrange)
        self.weight_layer.bias.data.zero_()
        self.first_layer.weight.data.uniform_(-initrange, initrange)
        self.first_layer.bias.data.zero_()
        self.last_layer.weight.data.uniform_(-initrange, initrange)
        self.last_layer.bias.data.zero_()
        for layer in self.layers:
            layer.weight.data.uniform_(-initrange, initrange)
            layer.bias.data.zero_()

    def forward(self, user, item, aspect):
        cat = torch.cat([user, item, aspect], dim = 1)  # (batch_size, emsize * 3)
        weights = self.weight_relu(self.weight_layer(cat))
        # (batch_size, 3)
        weighted = (
            weights[:, 0].unsqueeze(1) * user +
            weights[:, 1].unsqueeze(1) * item +
            weights[:, 2].unsqueeze(1) * aspect
        ) # (batch_size, emsize)
        weighted = self.first_layer(weighted)
        for layer in self.layers:
            weighted = layer(weighted) # (batch_size, hidden_size)
        rating = torch.squeeze(self.last_layer(weighted)) # (batch_size,)
        return rating



