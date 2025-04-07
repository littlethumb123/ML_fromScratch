import torch
from torch import nn
from position_encoding import PositionalEncoding
from transformer.token_encoding import TokenEncoding
from encoder import EncoderLayer, Encoder
from decoder import DecoderLayer, Decoder
import math
class Transformer(nn.Module):
    def __init__(self, 
                 src_vocab_size, # size of the source vocabulary corpus
                 tgt_vocab_size, # size of the target vocabulary corpus. exist in translation or summarization
                 max_len = 5000, # maximum length of the input sequence, e.g., 5000 in the original paper
                 d_model = 512, # dimension of the model, e.g., 256, 512
                 num_heads = 8, # number of attention heads, e.g., 8 in the original paper
                 num_transformer_layers = 6, # number of transformer blocks, e.g., 6 in the original paper
                 d_ff = 2048, # dimension of the feedforward layer, e.g., 2048 in the original paper
                 dropout_rate=0.1
                 ): # dropout rate, e.g., 0.1 in the original paper
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_transformer_layers = num_transformer_layers
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

        self.src_embedding = TokenEncoding(src_vocab_size, d_model, max_len)
        self.tgt_embedding = TokenEncoding(tgt_vocab_size, d_model, max_len)
        self.src_pos_encoding = PositionalEncoding(d_model, max_len, dropout_rate)
        self.tgt_pos_encoding = PositionalEncoding(d_model, max_len, dropout_rate)

        self.encoder = Encoder(
            EncoderLayer(d_model, num_heads, d_ff, dropout_rate),
            num_transformer_layers
        )
        self.decoder = Decoder(
            DecoderLayer(d_model, num_heads, d_ff, dropout_rate),
            num_transformer_layers
        )

        # projection layer to convert the output of the decoder to the target vocabulary size
        self.projection = nn.Linear(d_model, tgt_vocab_size)
    
    def encode(self, src, src_mask=None):
        """
        src: the input tensor of shape (batch_size, seq_len)
        src_mask: the source mask of shape (batch_size, 1, 1, seq_len) or (batch_size, seq_len, seq_len)
        """
        # src shape: (batch_size, seq_len)
        src = self.src_embedding(src)
        src = self.src_pos_encoding(src) # (batch_size, seq_len, d_model)
        return self.encoder(src, src_mask)

    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """
        tgt: the target tensor of shape (batch_size, seq_len)
        encoder_output: the output of the encoder of shape (batch_size, seq_len, d_model)
        src_mask: the source mask of shape (batch_size, 1, 1, seq_len) or (batch_size, seq_len, seq_len)
        tgt_mask: the target mask of shape (batch_size, 1, 1, seq_len) or (batch_size, seq_len, seq_len)
        """
        # tgt shape: (batch_size, seq_len)
        tgt = self.tgt_embedding(tgt)
        tgt = self.tgt_pos_encoding(tgt) # (batch_size, seq_len, d_model)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def forward(self, 
                src, # the input tensor of shape (batch_size, seq_len)
                tgt, # the target tensor of shape (batch_size, seq_len)
                src_mask=None, # the source mask of shape (batch_size, 1, 1, seq_len) or (batch_size, seq_len, seq_len)
                tgt_mask=None # the target mask of shape (batch_size, 1, 1, seq_len) or (batch_size, seq_len, seq_len)
                ):
        # Create masks if not provided
        if src_mask is None:
            src_mask = self._create_src_mask(src)
        if tgt_mask is None:
            tgt_mask = self._create_tgt_mask(tgt)
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        output = self.projection(decoder_output)
        return output