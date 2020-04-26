from random import random

import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Implements additive attention and return the attention vector used to weight the values.
        Additive attention consists in concatenating key and query and then passing them trough a linear layer."""

    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Parameter(torch.rand(dec_hid_dim))

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


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention, device):
        super().__init__()

        self.device = device

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU(enc_hid_dim + emb_dim, dec_hid_dim, batch_first=True, num_layers=1)
        #  The input will be the concat between attention result and input

        self.out = nn.Linear(enc_hid_dim + dec_hid_dim + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, queries, key):
        # input = [batch size]
        # queries = [batch size, dec hid dim]
        # encoder_outputs = [src sent len, batch size, enc hid dim * 2]

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


class Seq2Seq(nn.Module):
    def __init__(self, decoder, device):
        super().__init__()

        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        batch_size = src.shape[0]
        max_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, max_len, trg_vocab_size).to(self.device)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer

        # first input to the decoder is the <sos> tokens
        output = trg[:, 0]

        hidden = torch.zeros(output.shape[0], self.decoder.dec_hid_dim).to(self.device)

        for t in range(1, max_len):
            output, hidden = self.decoder(output, src, hidden)
            outputs[:, t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            # il primo 1 indica che il massimo viene cercato per ogni riga. Il secondo prende l'indice e non il valore
            top1 = output.max(1)[1]
            output = (trg[:, t] if teacher_force else top1)

        return outputs

