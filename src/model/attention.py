import random

import torch.nn as nn
import torch.nn.functional as F
import torch


class Attention(nn.Module):
    """Implements additive attention and return the attention vector used to weight the values.
        Additive attention consists in concatenating key and query and then passing them trough a linear layer."""

    def __init__(self, enc_hid_dim, dec_hid_dim, attention_hidden_size):
        super().__init__()

        # dec hid dim può essere cambiato, è la dimensione dell'hidden state dell'attention
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, attention_hidden_size)
        self.v = nn.Parameter(torch.rand(attention_hidden_size), requires_grad=True)

    def forward(self, key, queries):
        # key = [batch size, dec hid dim]
        # queries = [batch size, src sent len, enc hid dim]

        batch_size = queries.shape[0]
        src_len = queries.shape[1]

        # repeat encoder hidden state src_len times
        key = key.unsqueeze(1).repeat(1, src_len, 1)

        # hidden = [batch size, src sent len, dec hid dim]
        # encoder_outputs = [batch size, src sent len, enc hid dim]
        energy = torch.tanh(self.attn(torch.cat((key, queries), dim=2)))
        # energy = [batch size, src sent len, dec hid dim]

        energy = energy.permute(0, 2, 1)
        # energy = [batch size, dec hid dim, src sent len]

        # v = [dec hid dim]
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        # v = [batch size, 1, dec hid dim]

        # This multiplication generate a number for each query
        attention = torch.bmm(v, energy).squeeze(1)
        # attention= [batch size, src len]

        return F.softmax(attention, dim=1)