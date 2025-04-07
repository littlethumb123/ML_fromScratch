import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout):
        """
        d_model: the dimension of the model, e.g., 256, 512
        h: the number of attention heads, e.g., 8 in the original paper
        dropout: the dropout rate, e.g., 0.1 in the original paper
        """
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h # dimension of each attention head
        self.dropout = nn.Dropout(dropout) # dropout layer
        self.linear_q = nn.Linear(d_model, d_model) # linear layer for query
        self.linear_k = nn.Linear(d_model, d_model) # linear layer for key
        self.linear_v = nn.Linear(d_model, d_model) # linear layer for value

        self.linear_out = nn.Linear(d_model, d_model) # linear layer for output

    def scaled_dot_product_attention(self, query, key, value, mask=None):
        """
        query: query tensor of shape (batch_size, h, seq_len, d_k)
        key: key tensor of shape (batch_size, h, seq_len, d_k)
        value: value tensor of shape (batch_size, h, seq_len, d_k)
        mask: optional mask tensor of shape (batch_size, 1, 1, seq_len) or (batch_size, 1, seq_len, seq_len); 
              this is used for inference stage to prevent the model from attending to future tokens
        Formulas: 
        scores = torch.matmul(query, key.transpose(-2, -1)) / sqrt(d_k)
        scores = scores.masked_fill(mask == 0, -1e9) # fill the masked positions with a large negative value
        """
        # query shape: (batch_size, h, seq_len, d_k)
        # key shape: (batch_size, h, seq_len, d_k) -> transpose to (batch_size, h, d_k, seq_len) -> scores shape: (batch_size, h, seq_len, seq_len)
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k).float())
        if mask is not None:
            # Ensure the mask is broadcastable to the shape of scores
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9) # fill the masked positions with a large negative value
        attention_weights = torch.softmax(scores, dim=-1) # shape: (batch_size, h, seq_len, seq_len)
        # If dropout is used, apply it to the attention weights
        if self.dropout:
            attention_weights = self.dropout(attention_weights)
        
        # Multiply the attention weights with the value tensor to get the output
        output = torch.matmul(attention_weights, value) # value shape is (batch_size, h, seq_len< d_k) -> output shape: (batch_size, h, seq_len, d_k)
        return output, attention_weights # output shape: (batch_size, h, seq_len, d_k), attention_weights shape: (batch_size, h, seq_len, seq_len)
    
    def forward(self, q, k, v, mask=None):
        """
        q: query tensor of shape (batch_size, seq_len, d_model)
        k: key tensor of shape (batch_size, seq_len, d_model)
        v: value tensor of shape (batch_size, seq_len, d_model)
        mask: optional mask tensor of shape (batch_size, 1, 1, seq_len) or (batch_size, 1, seq_len, seq_len); 
              this is used for inference stage to prevent the model from attending to future tokens
        """
        query = self.linear_q(q) # shape (batch_size, seq_len, d_model)
        key = self.linear_k(k) # shape (batch_size, seq_len, d_model)
        value = self.linear_v(v) # shape (batch_size, seq_len, d_model)

        # Now split the d_model dimension into h heads, each of size self.d_k, transpose the result to get the shape (batch_size, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)  # shape (batch_size, h, seq_len, d_k)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)  # shape (batch_size, h, seq_len, d_k)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)  # shape (batch_size, h, seq_len, d_k)

        # Calculate the attention scores using the scaled dot-product attention formula
        x, self.attention_scores = self.scaled_dot_product_attention(query, key, value, mask) # shape (batch_size, h, seq_len, d_k)

        # Concatenate the output from all heads and pass it through the final linear layer
        # (batch_size, h, seq_len, d_k) -> (batch_size, seq_len, h * d_k) -> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2)
        x = x.contiguous() # Ensure the tensor is contiguous in memory
        x = x.view(x.shape[0], x.shape[1], self.d_model) # shape (batch_size, seq_len, d_model)

        x = self.linear_out(x)
        return x