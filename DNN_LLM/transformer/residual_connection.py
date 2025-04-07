import torch
import torch.nn as nn
import torch.nn.functional as F
from layer_norm import LayerNormalization

class ResidualConnection(nn.Module):

    def __init__(self, size, dropout=0.1):
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(size)

    def forward(self, x, sublayer):
        """
        In the original paper, the residual connection is defined as:
        y = x + Dropout(sublayer(LayerNorm(x)))
        first apply layer normalization to x, then apply the sublayer (e.g., multi-head attention or feedforward), and finally add the original x back to the output.
        """
        return x + self.dropout(sublayer(self.layer_norm(x)))