import torch
import torch.nn as nn

class TwoLayerLSTM(nn.Module):
    """
    Two-layer LSTM model.
    
    Args:
        input_size (int): Number of features per timestep.
        hidden_size (int): Number of hidden units in LSTM layers.
        
    Returns:
        Output tensor of shape (batch_size, 1) for regression.
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # LSTM returns (output, (hidden_state, cell_state))
        output, (hn, cn) = self.lstm(x)
        # Use the last hidden state from the top layer
        out = self.fc(hn[-1])
        return out
