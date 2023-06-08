import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Class implementing attention block. Is used by Visinet.
    """

    def __init__(self, dim=512):
        super(Attention, self).__init__()

        self.query_layer = nn.Linear(dim, dim, bias=False)
        self.key_layer = nn.Linear(dim, dim, bias=False)
        self.value_layer = nn.Linear(dim, dim, bias=False)

        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=5, batch_first=True)

        self.final_linear = nn.Linear(dim, dim)

        self.batch_norm1 = nn.LayerNorm(dim)
        self.batch_norm2 = nn.LayerNorm(dim)

    def forward_(self, x, weights=False):
        q = self.query_layer(x)
        k = self.key_layer(x)
        v = self.value_layer(x)

        x, w = self.attention(q, k, v)

        # print(f"x shape: {x.shape} w shape {w.shape}")
        # w.shape = (batch, seq_len, seq_len)
        if weights:
            return x, w

        return x

    def forward(self, x, weights=False):
        x_orig = x

        q = self.query_layer(x)
        k = self.key_layer(x)
        v = self.value_layer(x)

        x, w = self.attention(q, k, v)

        x = self.batch_norm1(x + x_orig)

        x = self.batch_norm2(x + F.relu(self.final_linear(x)))

        # print(f"x shape: {x.shape} w shape {w.shape}")
        # w.shape = (batch, seq_len, seq_len)
        if weights:
            return x, w
        return x
