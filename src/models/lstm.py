# src/models/lstm.py

import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    Baseline LSTM model for traffic forecasting.
    Input: (B, 12, 207, 1) -> Output: (B, 3, 207, 1)
    """
    def __init__(self):
        super(LSTMModel, self).__init__()
        # Implementation to be added in Phase 3
        pass

    def forward(self, x):
        pass
