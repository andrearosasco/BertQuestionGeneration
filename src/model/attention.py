import torch.nn as nn
import torch.nn.functional as f
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
        #linear layer apply to the last dimension of the input tensor
        energy = torch.tanh(self.attn(torch.cat((key, queries), dim=2)))
        # energy = [batch size, src sent len, att hid dim]


        # energy = [batch size, att hid dim, src sent len]

        # v = [att hid dim]
        v = self.v.repeat(batch_size, 1).unsqueeze(2)
        # v = [batch size, 1, att hid dim]

        # This multiplication generate a number for each query
        attention = torch.bmm(energy, v).squeeze(2)
        # attention= [batch size, src len]

        return f.softmax(attention, dim=1)
