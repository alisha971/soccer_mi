"""
src/batching.py — build model inputs from windows of event row numbers (notebook 03).
"""
import numpy as np
import torch


class EncodedData:
    """All model inputs for all events, stored as tensors on one device.

    features : dict name -> numpy array with one value per event (from data.encode_features)
    targets  : numpy array of action ids, one per event
    """

    def __init__(self, features, targets, device):
        self.device = device
        self.tensors = {}
        for name, values in features.items():
            self.tensors[name] = torch.as_tensor(np.asarray(values), device=device)
        self.targets = torch.as_tensor(np.asarray(targets), device=device)

    def batch(self, window_rows):
        """window_rows: LongTensor [batch, positions] of event row numbers.

        Returns (inputs, targets):
          inputs  dict name -> tensor [batch, positions]
          targets tensor [batch, positions] of action ids
        """
        inputs = {}
        for name, tensor in self.tensors.items():
            inputs[name] = tensor[window_rows]
        targets = self.targets[window_rows]
        return inputs, targets
