import time
import logging

import torch
import torch.nn as nn

from .utils import epoch_time

pw_criterion = nn.CrossEntropyLoss(ignore_index=0)  # Pad Index

def train(model, device, dataloader, optimizer, criterion, clip, encoder, encoder_trained):
    log = logging.getLogger(__name__)
    model.train()
    encoder.train() if encoder_trained else encoder.eval()

    epoch_loss = 0

    start = time.time()
    for i, (input_, output_) in enumerate(dataloader):

        input_data, input_length = input_
        output_data, output_length = output_

        input_ids, token_type_ids, attention_mask = input_data

        optimizer.zero_grad()

        if encoder_trained:
            bert_hs = encoder(input_ids.to(device), token_type_ids=token_type_ids.to(device),
                              attention_mask=attention_mask.to(device))
        else:
            with torch.no_grad():
                bert_hs = encoder(input_ids.to(device), token_type_ids=token_type_ids.to(device),
                                  attention_mask=attention_mask.to(device))


        prediction = model(bert_hs[0],  output_data.to(device))

        trg_sent_len = prediction.size(1)

        prediction = prediction[:, 1:].contiguous().view(-1, prediction.shape[-1])
        output_data = output_data[:, 1:].contiguous().view(-1)  # Find a way to avoid calling contiguous

        with torch.no_grad():
            pw_loss = pw_criterion(prediction,  output_data.to(device))

        loss = criterion(prediction,  output_data.to(device))

        # reshape to [trg sent len - 1, batch size]
        loss = loss.view(-1, trg_sent_len - 1)
        loss = loss.sum(1)
        loss = loss.mean(0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), clip)
        optimizer.step()

        if i % int(len(dataloader) * 0.1) == int(len(dataloader) * 0.1) - 1:
            log.info(
                f'Batch {i} Sentence loss {loss.item()} Word loss {pw_loss.item()}   Time: {epoch_time(start, time.time())}')
            start = time.time()

        epoch_loss += pw_loss.item()

    return epoch_loss / len(dataloader)
