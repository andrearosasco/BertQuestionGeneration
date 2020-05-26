import logging

from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate import bleu

logging.getLogger('transformers').setLevel(logging.WARNING)
log = logging.getLogger(__name__)

import time
import math

import torch
from torch import optim, nn, cuda
from transformers import AdamW, BertTokenizer
from torch.utils.data import DataLoader
from transformers import BertModel

from config import checkpoint, bert_path, mb, dl_workers, device, bert_hidden_size, decoder_hidden_size, \
    bert_vocab_size, decoder_input_size, dropout, epochs, clip, model_path, stage, bert_model, encoder_trained, \
    attention_hidden_size, num_layers, weight_decay, betas, lr, momentum
from model.utils import load_checkpoint, init_weights, save_checkpoint, enable_reproducibility, model_size, no_grad
from model import Attention, Decoder, Seq2Seq
from data import BertDataset
from run import train, eval
from run.utils.time import epoch_time

# best_valid_loss = float('inf')

def bleu_score(prediction, ground_truth):
    prediction = prediction.max(2)[1]
    bleu_list = []

    for x, y in zip(ground_truth, prediction):
        x = tokenizer.convert_ids_to_tokens(x.tolist())
        y = tokenizer.convert_ids_to_tokens(y.tolist())
        idx1 = x.index('[SEP]') if '[SEP]' in x else len(x)
        idx2 = y.index('[SEP]') if '[SEP]' in y else len(y)

        bleu_list.append((bleu([x[1:idx1]], y[1:idx2], [0.25, 0.25, 0.25, 0.25],
                         smoothing_function=SmoothingFunction().method4), x[1:idx1], y[1:idx2]))
    return (max(bleu_list, key=lambda x: x[0]))

def loss(prediction, ground_truth):
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
    train_loss = []
    for x, y in zip(ground_truth, prediction):
        w = tokenizer.convert_ids_to_tokens(x.tolist())
        z = tokenizer.convert_ids_to_tokens(y.max(1)[1].tolist())
        idx1 = w.index('[SEP]') if '[SEP]' in w else len(w)
        idx2 = z.index('[SEP]') if '[SEP]' in z else len(z)

        loss = criterion(y, x.to(device))
        loss = loss.sum()
        train_loss.append((loss, w[1:idx1], z[1:idx1]))
    return (max(train_loss, key=lambda x: x[0]))

    #
    # trg_sent_len = prediction.size(1)
    # prediction = prediction[:, 1:].contiguous().view(-1, prediction.shape[-1])
    # output_data = ground_truth[:, 1:].contiguous().view(-1)
    # criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
    # loss = criterion(prediction, output_data.to(device))
    # loss = loss.view(-1, trg_sent_len - 1)
    # loss = loss.sum(1)


# Importante! Se il training viene fermato e poi ripreso senza cambiare il seed lo shuffling non avviene
tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
if __name__ == '__main__':
    enable_reproducibility(121314)

    valid_set = BertDataset(bert_path / bert_model / 'test')
    valid_loader = DataLoader(valid_set, batch_size=mb, shuffle=True,
                              num_workers=dl_workers, pin_memory=True if device == 'cuda' else False)

    attention = Attention(bert_hidden_size, decoder_hidden_size)
    decoder = Decoder(bert_vocab_size, decoder_input_size, bert_hidden_size, decoder_hidden_size, dropout, attention, device)
    model = Seq2Seq(decoder, device)

    encoder = BertModel.from_pretrained(model_path / 'stage_one' / bert_model)
    encoder.to(device)

    _, model_dict, _, _, _, _ = load_checkpoint(checkpoint)
    model.load_state_dict(model_dict)

    model.to(device)
    bleu_list = []
    loss_list = []
    with torch.no_grad():

        for i, (input_, output_) in enumerate(valid_loader):

            input_data, input_length = input_
            output_data, output_length = output_

            input_ids, token_type_ids, attention_mask = input_data

            bert_hs = encoder(input_ids.to(device), token_type_ids=token_type_ids.to(device),
                              attention_mask=attention_mask.to(device))

            prediction = model(bert_hs[0], output_data.to(device), 0)  # turn off teacher forcing

            # bleu_list.append(bleu_score(prediction, output_data.to(device)))
            loss_list.append(loss(prediction, output_data.to(device)))

            # trg = [(trg sent len - 1) * batch size]
            # output = [(trg sent len - 1) * batch size, output dim]
    print(sorted(loss_list, key=lambda x: x[0])[0:10])


