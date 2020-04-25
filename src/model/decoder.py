import random

import torch.nn as nn
import torch.nn.functional as F
import torch

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU(enc_hid_dim + emb_dim, dec_hid_dim, batch_first=True)
        #  The input will be the concat between attention result and input

        self.out = nn.Linear(enc_hid_dim + dec_hid_dim + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, queries, key=None):
        # input = [batch size]
        # queries = [batch size, dec hid dim]
        # encoder_outputs = [src sent len, batch size, enc hid dim * 2]
        if key is None:
            key = torch.zeros(input.shape[0], self.dec_hid_dim)

        input = input.unsqueeze(1)
        # input = [batch size, senq len]

        embedded = self.dropout(self.embedding(input))
        # embedded = [batch size, seq len, emb dim]

        a = self.attention(key, queries)

        # a = [batch size, src len]
        a = a.unsqueeze(1)
        # a = [batch size, 1, src len]

        # queries = [batch size, src sent len, enc hid dim]

        weighted = torch.bmm(a, queries)
        # weighted = [batch size, 1, enc hid dim]

        rnn_input = torch.cat((embedded, weighted), dim=2)
        # rnn_input = [1, batch size, enc hid dim + emb dim]

        output, hidden = self.rnn(rnn_input, key.unsqueeze(0))

        # output = [sent len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        # sent len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden

        embedded = embedded.squeeze(1)
        output = output.squeeze(1)
        weighted = weighted.squeeze(1)

        output = self.out(torch.cat((output, weighted, embedded), dim=1))

        # output = [bsz, output dim]

        return output, hidden.squeeze(0)
