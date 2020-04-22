import logging

import torch
from nltk.translate.bleu_score import SmoothingFunction
from torch import nn
from nltk.translate import bleu
from transformers import BertTokenizer

pw_criterion = nn.CrossEntropyLoss(ignore_index=0)  # Pad Index

def eval(model, device, dataloader, criterion):
    log = logging.getLogger(__name__)
    model.eval()

    epoch_loss = 0
    epoch_bleu = 0

    with torch.no_grad():

        for i, (input_, output_) in enumerate(dataloader):

            input_data, input_length = input_
            output_data, output_length = output_

            prediction = model([x.to(device) for x in input_data], output_data.to(device), 0)  # turn off teacher forcing

            # Non può essere fatto in parallelo... Facciamolo dopo il training sul test set
            bleu = bleu_score(prediction, output_data.to(device))

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
                log.info(f'Batch {i} Sentence loss: {loss.item()} Word loss: {pw_loss.item()} BLEU score: {bleu}')

            epoch_loss += pw_loss.item()
            epoch_bleu += bleu

        return epoch_loss / len(dataloader), epoch_bleu / len(dataloader)

def bleu_score(prediction, ground_truth):
    prediction = prediction.max(2)[1]
    acc_bleu = 0

    for x, y in zip(ground_truth, prediction):

        idx1 = x.tolist().index(0) if 0 in x.tolist() else len(x.tolist)
        idx2 = y.tolist().index(102) if 102 in y.tolist() else len(y.tolist)
        acc_bleu += bleu([x[1:idx1-1]], y[1:idx2-1], smoothing_function=SmoothingFunction().method4)
        print(x[1:idx1])
        print(y[1:idx2])
        exit(0)
    return acc_bleu / prediction.size(0)


    # candidate = tokenizer.convert_ids_to_tokens(prediction[0].max(1)[1].tolist())
    # reference = tokenizer.convert_ids_to_tokens(ground_truth[0].tolist())