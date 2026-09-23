"""
src/mlp_baseline.py — B4: MLP over the current event and the n previous events (notebook 04).
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import src.model as model_module


class PreviousEventsMLP(nn.Module):
    def __init__(self, config, n_previous=3, hidden=512):
        super().__init__()
        embedding_config = copy.deepcopy(config)
        embedding_config.use_position = False          # order is given by concatenation instead
        self.config = embedding_config
        self.n_previous = n_previous
        self.embedding = model_module.FactorizedEventEmbedding(embedding_config)
        input_width = config.d_model * (n_previous + 1)
        self.layer_1 = nn.Linear(input_width, hidden)
        self.layer_2 = nn.Linear(hidden, hidden)
        self.output_layer = nn.Linear(hidden, config.n_actions)

    def forward(self, inputs):
        event_vectors, _ = self.embedding(inputs)                         # [B, T, D]
        pieces = [event_vectors]
        for lag in range(1, self.n_previous + 1):
            shifted = torch.zeros_like(event_vectors)
            shifted[:, lag:] = event_vectors[:, :-lag]                    # event t-lag, zeros if absent
            pieces.append(shifted)
        combined = torch.cat(pieces, dim=-1)                              # [B, T, D * (n_previous + 1)]
        hidden = F.gelu(self.layer_1(combined))
        hidden = F.gelu(self.layer_2(hidden))
        return self.output_layer(hidden)                                  # [B, T, C]
