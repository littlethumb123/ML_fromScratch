import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    """
    Caculuate the mean and variance of the input tensor along the last dimension.
    Normalize the embedding dimension for all batch samples
    The purpose is to stabilize the training process and improve the convergence speed.
    
    Where to apply Layer Normalization to the Transformer model?
    1. Post layer normalization: after the residual connection and before the feedforward layer in the encoder and decoder layers.
    2. Pre layer normalization: before the residual connection in the encoder and decoder layers.
    In the original paper, the post layer normalization is used, however, the pre layer normalization improves training stability and convergence
    """
    def __init__(self, size, eps=1e-6):
        """
        size: the dimension of the model, e.g., 256, 512
        eps: a small value to avoid division by zero, e.g., 1e-6 in the original paper
        """
        super(self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(size))
        self.beta = nn.Parameter(torch.zeros(size))
    
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        var = x.var(dim = -1, keepdim=True, unbiased=False)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.alpha * out + self.beta
        return out