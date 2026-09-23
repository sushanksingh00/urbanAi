import torch
import torch.nn as nn

class HistoricalAverage(nn.Module):
    """
    Historical Average (HA) baseline model.
    Predicts the mean of the historical window for all future steps.
    """
    def __init__(self, horizon_steps: int):
        super().__init__()
        self.horizon_steps = horizon_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, history_steps, nodes, features)
        Returns:
            y_pred: Predicted tensor of shape (batch, horizon_steps, nodes, features)
        """
        # Compute the mean over the history_steps dimension (dim=1)
        # Shape becomes (batch, 1, nodes, features)
        mean_val = torch.mean(x, dim=1, keepdim=True)
        
        # Repeat the mean for the horizon_steps
        # Shape becomes (batch, horizon_steps, nodes, features)
        y_pred = mean_val.repeat(1, self.horizon_steps, 1, 1)
        return y_pred


class LastValue(nn.Module):
    """
    Last Value (LV) baseline model.
    Predicts the most recent value from the historical window for all future steps.
    """
    def __init__(self, horizon_steps: int):
        super().__init__()
        self.horizon_steps = horizon_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, history_steps, nodes, features)
        Returns:
            y_pred: Predicted tensor of shape (batch, horizon_steps, nodes, features)
        """
        # Extract the last time step from the history_steps dimension (dim=1)
        # Shape becomes (batch, 1, nodes, features)
        last_val = x[:, -1:, :, :]
        
        # Repeat the last value for the horizon_steps
        # Shape becomes (batch, horizon_steps, nodes, features)
        y_pred = last_val.repeat(1, self.horizon_steps, 1, 1)
        return y_pred


class LinearRegressionModel(nn.Module):
    """
    Linear Regression Baseline.
    Flattens the historical window (history_steps * nodes * features) to predict
    the future window (horizon_steps * nodes * features).
    """
    def __init__(self, history_steps: int, horizon_steps: int, nodes: int, features: int = 1):
        super().__init__()
        self.horizon_steps = horizon_steps
        self.nodes = nodes
        self.features = features
        
        in_dim = history_steps * nodes * features
        out_dim = horizon_steps * nodes * features
        
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        # Flatten: (batch, history_steps * nodes * features)
        x_flat = x.view(batch_size, -1)
        y_flat = self.linear(x_flat)
        # Reshape: (batch, horizon_steps, nodes, features)
        y_pred = y_flat.view(batch_size, self.horizon_steps, self.nodes, self.features)
        return y_pred


class LSTMModel(nn.Module):
    """
    LSTM Baseline.
    Treats each node as having a sequence of `history_steps`.
    """
    def __init__(self, history_steps: int, horizon_steps: int, nodes: int, features: int = 1, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.horizon_steps = horizon_steps
        self.nodes = nodes
        self.features = features
        self.hidden_dim = hidden_dim
        
        # We'll reshape the input to treat nodes as part of the batch dimension.
        self.lstm = nn.LSTM(input_size=features, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, horizon_steps * features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        # x: (batch, history_steps, nodes, features)
        # Permute to: (batch, nodes, history_steps, features)
        x = x.permute(0, 2, 1, 3).contiguous()
        # Reshape to: (batch * nodes, history_steps, features)
        x_seq = x.view(batch_size * self.nodes, -1, self.features)
        
        # LSTM output
        out, _ = self.lstm(x_seq)
        
        # Take the last hidden state
        out_last = out[:, -1, :]
        
        # Predict future horizon
        y_pred = self.fc(out_last)
        
        # Reshape back
        y_pred = y_pred.view(batch_size, self.nodes, self.horizon_steps, self.features)
        
        # Permute to: (batch, horizon_steps, nodes, features)
        y_pred = y_pred.permute(0, 2, 1, 3).contiguous()
        return y_pred
