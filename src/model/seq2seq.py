import random

import torch.nn as nn
import torch.nn.functional as F
import torch

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

        for t in range(1, max_len):
            output, hidden = self.decoder(output, src)
            outputs[:, t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            # il primo 1 indica che il massimo viene cercato per ogni riga. Il secondo prende l'indice e non il valore
            top1 = output.max(1)[1]
            output = (trg[:, t] if teacher_force else top1)

        return outputs
