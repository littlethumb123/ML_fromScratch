import torch
import torch.nn as nn
from multihead_attention import MultiHeadAttention
from feedforward import FeedForward
from residual_connection import ResidualConnection
from layer_norm import LayerNormalization

class DecoderLayer(nn.Module):
    """Decoder Layer with multi-head attention and feed-forward network.
    Each layer includes
    1. Multi-head self-attention
    2. residual connection
    3. Cross-attention (encoder-decoder attention)
    4. residual connection
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
        self.residual_connection_1 = ResidualConnection(d_model, dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.residual_connection_2 = ResidualConnection(d_model, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.residual_connection_3 = ResidualConnection(d_model, dropout)
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        x: the input tensor of shape (batch_size, seq_len, d_model)
        encoder_output: the output of the encoder of shape (batch_size, seq_len, d_model)
        src_mask: the source mask of shape (batch_size, 1, seq_len)
        tgt_mask: the target mask of shape (batch_size, 1, seq_len)
        """
        att_output, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.residual_connection_1(x, att_output)
        cross_att_output, _ = self.cross_attn(x, encoder_output, encoder_output, attn_mask=src_mask)
        x = self.residual_connection_2(x, cross_att_output)
        output = self.feed_forward(x)
        x = self.residual_connection_3(x, output)
        return x

class Decoder(nn.Module):
    """Decoder with multiple layers of DecoderLayer.
    Each layer includes
    1. Multi-head self-attention
    2. residual connection
    3. Cross-attention (encoder-decoder attention)
    4. residual connection
    3. Feed-forward network
    4. residual connection (layer norm is already applied inside)
    """
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.decoder_layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.layer_norm = LayerNormalization(decoder_layer.d_model)
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        x: the input tensor of shape (batch_size, seq_len, d_model)
        encoder_output: the output of the encoder of shape (batch_size, seq_len, d_model)
        src_mask: the source mask of shape (batch_size, 1, seq_len)
        tgt_mask: the target mask of shape (batch_size, 1, seq_len)
        """
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.layer_norm(x)



