import logging

import torch
from nltk.translate.bleu_score import SmoothingFunction
from torch import nn
from nltk.translate import bleu
from transformers import BertTokenizer

pw_criterion = nn.CrossEntropyLoss(ignore_index=0)  # Pad Index
tokenizer = BertTokenizer.from_pretrained('bert-large-cased')

def eval(model, device, dataloader, criterion, encoder):
    log = logging.getLogger(__name__)
    model.eval()

    epoch_loss = 0
    epoch_bleu = 0

    with torch.no_grad():

        for i, (input_, output_) in enumerate(dataloader):

            input_data, input_length = input_
            output_data, output_length = output_

            input_ids, token_type_ids, attention_mask = input_data

            bert_hs = encoder(input_ids.to(device), token_type_ids=token_type_ids.to(device),
                              attention_mask=attention_mask.to(device))

            prediction = model(bert_hs[0], output_data.to(device), 0)  # turn off teacher forcing

            sample_t = tokenizer.convert_ids_to_tokens(output_data[0].tolist())
            sample_p = tokenizer.convert_ids_to_tokens(prediction[0].max(1)[1].tolist())
            idx1 = sample_t.index('[SEP]') if '[SEP]' in sample_t else len(sample_t)
            idx2 = sample_p.index('[SEP]') if '[SEP]' in sample_p else len(sample_p)

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
                log.info(f'Batch {i} Sentence loss: {loss.item()} Word loss: {pw_loss.item()} BLEU score: {bleu}\n'
                         f'Target {sample_t[1:idx1]}\n'
                         f'Prediction {sample_p[1:idx2]}\n\n')

            epoch_loss += pw_loss.item()
            epoch_bleu += bleu

        return epoch_loss / len(dataloader), epoch_bleu / len(dataloader)

def bleu_score(prediction, ground_truth):
    prediction = prediction.max(2)[1]
    acc_bleu = 0

    for x, y in zip(ground_truth, prediction):
        x = tokenizer.convert_ids_to_tokens(x.tolist())
        y = tokenizer.convert_ids_to_tokens(y.tolist())
        idx1 = x.index('[SEP]') if '[SEP]' in x else len(x)
        idx2 = y.index('[SEP]') if '[SEP]' in y else len(y)

        acc_bleu += bleu([x[1:idx1]], y[1:idx2], [0.25, 0.25, 0.25, 0.25],smoothing_function=SmoothingFunction().method4)
    return acc_bleu / prediction.size(0)


    # candidate = tokenizer.convert_ids_to_tokens(prediction[0].max(1)[1].tolist())
    # reference = tokenizer.convert_ids_to_tokens(ground_truth[0].tolist())