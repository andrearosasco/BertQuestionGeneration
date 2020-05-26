import random

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


import torch.nn.functional as F
from torch import nn
import torch


class BeamSearch(nn.Module):

    def __init__(self, decoder, device, k):
        super().__init__()
        self.decoder = decoder
        self.device = device
        self.k = k

    def forward(self, src):
        batch_size = src.shape[0]
        max_len = 40
        trg_vocab_size = self.decoder.output_dim

        search_results = torch.zeros(batch_size, self.k, max_len).type(torch.LongTensor).to(self.device)
        search_map = torch.zeros(batch_size, self.k, max_len).type(torch.LongTensor).to(self.device)
        outputs = torch.zeros(batch_size, max_len).type(torch.LongTensor).to(self.device)
        hiddens = torch.zeros(batch_size, self.k, self.decoder.dec_hid_dim).to(self.device)
        ended = torch.zeros(batch_size, self.k).to(self.device)
        true = torch.ones(ended.shape).to(self.device)
        no_prob = torch.Tensor(batch_size, trg_vocab_size).fill_(float('-Inf')).to(self.device)
        no_prob[:, 102] = 0
        lengths = torch.zeros(batch_size, self.k).to(self.device)

        output = torch.Tensor(batch_size).fill_(102).type(torch.LongTensor).to(self.device)
        hidden = torch.zeros(output.shape[0], self.decoder.dec_hid_dim).to(self.device)

        output, hidden = self.decoder(output, src, hidden)
        output = F.log_softmax(output, dim=1)

        for i in range(self.k):
            hiddens[:, i, :] = hidden

        scores, search_results[:, :, 0] = torch.topself.k(output, self.k, 1)

        for t in range(1, max_len):  # walk over each step in the sequence
            candidates = torch.Tensor(batch_size, 0).to(self.device)
            for i in range(self.k):  # expands each candidate

                idx = search_map[:, 0, t - 1].unsqueeze(1).unsqueeze(1)
                idx = idx.expand(-1, -1, hiddens.shape[2])
                hidden = hiddens.gather(1, idx).squeeze(1).squeeze(1)

                output, hiddens[:, i, :] = self.decoder(search_results[:, i, t - 1], src,
                                                        hidden)  # for every word it contains the probability
                output = F.log_softmax(output, dim=1)

                output = torch.where(ended[:, i].unsqueeze(1).expand_as(output) == 0, output, no_prob)
                lengths[:, i] = torch.where(ended[:, i] == 0, lengths[:, i] + 1, lengths[:, i])

                output = output + scores[:, i].unsqueeze(1)

                candidates = torch.cat((candidates, output), 1)  # concatenate for every possibility

            norm_cand = torch.tensor(candidates)

            for i in range(self.k - 1):
                norm_cand[:, trg_vocab_size * i:trg_vocab_size * (i + 1)] /= (lengths[:, i] ** 0.7).unsqueeze(1)

            _, topk = torch.topk(norm_cand, self.k, 1)  # topk dim is 15*3, scores too

            for i in range(topk.shape[0]):
                scores[i, :] = candidates[i, topk[i, :]]

            ended = torch.where((topk - (topk / trg_vocab_size) * trg_vocab_size) == 102, true, ended)
            #         print(f'{ended[0]} {lengths[0]}')

            #         print(scores[0])
            #         print(tokenizer.convert_ids_to_tokens(search_results[0, :, 0].tolist()))

            search_results[:, :, t] = topk - (
                    topk / trg_vocab_size) * trg_vocab_size  # si può fare durante la ricostruzione
            search_map[:, :, t] = topk / trg_vocab_size

        _, idx = torch.max(scores, 1)
        #     idx[0] = 4
        #     print(idx[0])
        #     print(scores[0])

        for t in range(max_len - 1, -1, -1):
            outputs[:, t] = search_results[:, :, t].gather(1, idx.unsqueeze(1)).squeeze(1)
            idx = search_map[:, :, t].gather(1, idx.unsqueeze(1)).squeeze(1)
        return outputs