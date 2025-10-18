import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# 1. Fuzzy Input Layer
# ---------------------------
class FuzzyInputLayer(nn.Module):
    def __init__(self, input_dim, num_mfs):
        """
        input_dim: number of input features
        num_mfs: number of fuzzy membership functions per feature
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_mfs = num_mfs

        # Learnable Gaussian parameters
        self.centers = nn.Parameter(torch.randn(input_dim, num_mfs))
        self.sigmas = nn.Parameter(torch.ones(input_dim, num_mfs))

    def forward(self, x):
        """
        x: [batch_size, input_dim, seq_len]
        returns: [batch_size, input_dim * num_mfs, seq_len]
        """
        batch_size, input_dim, seq_len = x.shape
        x_expanded = x.unsqueeze(2)  # [batch, input_dim, 1, seq_len]
        c = self.centers.unsqueeze(0).unsqueeze(-1)  # [1, input_dim, num_mfs, 1]
        s = self.sigmas.unsqueeze(0).unsqueeze(-1)

        mu = torch.exp(-0.5 * ((x_expanded - c) / (s + 1e-6)) ** 2)
        mu = mu.view(batch_size, input_dim * self.num_mfs, seq_len)
        return mu

# ---------------------------
# 2. Full Model
# ---------------------------
class FuzzyConvLSTM(nn.Module):
    def __init__(self, input_dim, num_mfs, conv_channels, lstm_hidden, output_dim,
                 kernel_size=3, pool_size=2, dropout=0.3):
        super().__init__()

        self.fuzzy = FuzzyInputLayer(input_dim, num_mfs)
        fuzzy_out_channels = input_dim * num_mfs

        self.conv1 = nn.Conv1d(in_channels=fuzzy_out_channels,
                               out_channels=conv_channels,
                               kernel_size=kernel_size,
                               padding=kernel_size // 2)
        self.pool = nn.MaxPool1d(pool_size)
        self.lstm = nn.LSTM(input_size=conv_channels,
                            hidden_size=lstm_hidden,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        x: [batch_size, input_dim, seq_len]
        """
        # Fuzzification
        x = self.fuzzy(x)

        # Conv + Pool
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # [batch, conv_channels, reduced_seq_len]

        # Prepare for LSTM: [batch, seq_len, features]
        x = x.permute(0, 2, 1)

        # LSTM
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]  # last hidden state

        # Dropout + FC + ReLU
        x = self.dropout(x)
        x = self.fc(x)
        x = self.relu(x)

        return x
