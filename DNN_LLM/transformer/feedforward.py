import torch
import torch.nn as nn

class FeedForward(nn.Module):
    
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        # X = (batch_size, seq_len, d_model)
        # W1 = (d_model, d_ff), b1 = (d_ff)
        x = self.linear1(x) # W1 and b1
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x) # W2 and b2
        return x  # (batch_size, seq_len, d_model)
