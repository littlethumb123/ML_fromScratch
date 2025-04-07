import torch
import torch.nn as nn
import math

class TokenEncoding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len):
        """
        d_model: the dimension of the embedding vector
        vocab_size: the size of the vocabulary corpus
        max_len: the maximum length of the input sequence
        """
        super(TokenEncoding, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        """
        x: (batch_size, seq_len)
        """
        # x: (batch_size, seq_len, d_model)
        # "We multiply those weights by √dₖ (where dₖ is the dimension of keys) to counteract the effect 
        # that the dot products grow large in magnitude for large values of dₖ, pushing the softmax into regions with extremely small gradients."
        x = self.embedding(x) * math.sqrt(self.d_model)
        return x
