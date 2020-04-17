import logging

import torch
from nltk.translate.bleu_score import SmoothingFunction
from torch import nn
from nltk.translate import bleu
from transformers import BertTokenizer

pw_criterion = nn.CrossEntropyLoss(ignore_index=0)  # Pad Index

def eval(model, device, dataloader, criterion):
    log = logging.getLogger('QGModel')
    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for i, (input_, output_) in enumerate(dataloader):

            input_data, input_length = input_
            output_data, output_length = output_

            prediction = model([x.to(device) for x in input_data], output_data.to(device), 0)  # turn off teacher forcing

            bleu_score(prediction, output_data.to(device))

            trg_sent_len = prediction.size(1)
            # trg = [trg sent len, batch size]
            # output = [trg sent len, batch size, output dim]

            prediction = prediction[:, 1:].contiguous().view(-1, prediction.shape[-1])
            output_data = output_data[:, 1:].contiguous().view(-1)  # Find a way to avoid calling contiguous

            # trg = [(trg sent len - 1) * batch size]
            # output = [(trg sent len - 1) * batch size, output dim]

            pw_loss = pw_criterion(prediction, output_data.to(device))

            loss = criterion(prediction, output_data.to(device))
            loss = loss.view(-1, trg_sent_len - 1)
            loss = loss.sum(1)
            loss = loss.mean(0)

            if i % int(len(dataloader) * 0.1) == int(len(dataloader) * 0.1) - 1:
                log.info(f'Batch {i} Sentence loss: {loss.item()} Word loss: {pw_loss.item()}')

            epoch_loss += pw_loss.item()

    return epoch_loss / len(dataloader)

def bleu_score(prediction, ground_truth):
    prediction = prediction.max(2)[1]

    print(prediction.shape)
    print(ground_truth.shape)

    idx = ground_truth.index(0) + 1
    print(bleu(ground_truth[:, idx], prediction[:, idx], smoothing_function=SmoothingFunction().method4))
    exit(0)

    # candidate = tokenizer.convert_ids_to_tokens(prediction[0].max(1)[1].tolist())
    # reference = tokenizer.convert_ids_to_tokens(ground_truth[0].tolist())