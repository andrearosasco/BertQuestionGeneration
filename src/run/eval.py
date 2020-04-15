import torch
from torch import nn

from config import device

pw_criterion = nn.CrossEntropyLoss(ignore_index=0)  # Pad Index

def eval(model, dataloader, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for i, (input_, output_) in enumerate(dataloader):

            input_data, input_length = input_
            output_data, output_length = output_

            prediction = model(input_data.to(device), output_data.to(device), 0)  # turn off teacher forcing

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
                print(f'Batch {i} Sentence loss: {loss.item()} Word loss: {pw_loss.item()}')

            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)