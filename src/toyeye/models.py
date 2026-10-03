"""The tiny learnable eye: a miniature columnar circuit in the spirit of flyvis.

A 1D convolution across neighbouring receptors (circular padding for the ring),
then a small GRU at every column with weights shared across columns.
"""

import torch
from torch import nn


class TinyEye(nn.Module):
    def __init__(self, hidden: int = 16, kernel: int = 3, channels: int = 8):
        super().__init__()
        self.conv = nn.Conv1d(1, channels, kernel, padding=kernel // 2, padding_mode="circular")
        self.gru = nn.GRUCell(channels, hidden)
        self.hidden = hidden

    def forward(self, movie: torch.Tensor) -> torch.Tensor:
        """movie: (batch, steps, n) -> hidden states (batch, steps, n, hidden)."""
        raise NotImplementedError


class VelocityHead(nn.Module):
    """Decode a velocity per column from the hidden state."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.linear = nn.Linear(hidden, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h).squeeze(-1)
