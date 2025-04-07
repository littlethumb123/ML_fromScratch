import torch
import torch.nn as nn
from multihead_attention import MultiHeadAttention
from feedforward import FeedForward
from residual_connection import ResidualConnection
from layer_norm import LayerNormalization

class EncoderLayer(nn.Module):
    """Encoder Layer with multi-head attention and feed-forward network.
    Each layer includes
    1. Multi-head self-attention
    2. residual connection (layer norm is already applied inside)
    3. Feed-forward network
    4. residual connection (layer norm is already applied inside)
    """
    def __init__(self, d_model, n_heads, d_ff, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.residual_connection_1 = ResidualConnection(d_model, dropout)
        self.residual_connection_2 = ResidualConnection(d_model, dropout)

    def forward(self, x, src_mask=None):
        """
        x shape: (batch_size, seq_len, d_model)
        src_mask shape: (batch_size, 1, 1, seq_len) or (batch_size, seq_len, seq_len)
        """
        # Multi-head self-attention + residual connection
        attn_output, _ = self.self_attn(x, x, x, attn_mask=src_mask)
        x = self.residual_connection_1(x, attn_output) # X + attn_output, layer norm is already applied inside

        # Feed-forward network + residual connection
        ff_output = self.feed_forward(x)
        x = self.residual_connection_2(x, ff_output) # X + ff_output, layer norm is already applied inside
        return x

class Encoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.encoder_layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.layer_norm = LayerNormalization(encoder_layer.d_model) # Layer normalization after all layers
    
    def forward(self, x, src_mask=None):
        """
        x shape: (batch_size, seq_len, d_model)
        src_mask shape: (batch_size, 1, 1, seq_len) or (batch_size, seq_len, seq_len)
        """
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return self.layer_norm(x)