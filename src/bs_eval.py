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
    encoder.eval()

    epoch_loss = 0
    epoch_bleu = 0

    with torch.no_grad():

        for i, (input_, output_) in enumerate(dataloader):

            input_data, input_length = input_
            output_data, output_length = output_

            input_ids, token_type_ids, attention_mask = input_data

            bert_hs = encoder(input_ids.to(device), token_type_ids=token_type_ids.to(device),
                              attention_mask=attention_mask.to(device))

            prediction = model(bert_hs[0])

            bleu = bleu_score(prediction, output_data.to(device))

            epoch_bleu += bleu

        return epoch_loss / len(dataloader), epoch_bleu / len(dataloader)

def bleu_score(prediction, ground_truth):
    acc_bleu = 0

    for x, y in zip(ground_truth, prediction):
        x = tokenizer.convert_ids_to_tokens(x.tolist())
        y = tokenizer.convert_ids_to_tokens(y.tolist())
        idx1 = x.index('[SEP]') if '[SEP]' in x else len(x)
        idx2 = y.index('[SEP]') if '[SEP]' in y else len(y)
        try:
            acc_bleu += bleu([x[1:idx1]], y[1:idx2], [0.25, 0.25, 0.25, 0.25],
                             smoothing_function=SmoothingFunction().method4)
        except ZeroDivisionError:
            print(f'{idx1} {x}')
            print(f'{idx2} {y}')
    return acc_bleu / prediction.size(0)
