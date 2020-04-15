import time
import math

import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from config import checkpoint, bert_path, mb, dl_workers, device, bert_hidden_size, decoder_hidden_size, \
    bert_vocab_size, decoder_input_size, dropout, epochs, clip, model_path, encoder
from model.utils import load_checkpoint, init_weights, save_checkpoint, enable_reproducibility
from model import Attention, Decoder, Seq2Seq
from data import BertDataset
from run import train, eval
from run.utils.time import epoch_time
# best_valid_loss = float('inf')


# Importante! Se il training viene fermato e poi ripreso senza cambiare il seed lo shuffling non avviene

if __name__ == '__main__':
    enable_reproducibility(1234)

    train_set = BertDataset(bert_path/'toy')
    valid_set = BertDataset(bert_path/'toy')
    training_loader = DataLoader(train_set, batch_size=mb, shuffle=True,
                                 num_workers=dl_workers, pin_memory=True if device=='cuda' else False)
    valid_loader = DataLoader(valid_set, batch_size=mb, shuffle=False,
                              num_workers=dl_workers, pin_memory=True if device=='cuda' else False)

    attention = Attention(bert_hidden_size, decoder_hidden_size) #add attention_hidden_size
    decoder = Decoder(bert_vocab_size, decoder_input_size, bert_hidden_size, decoder_hidden_size,
                      dropout, attention, device)

    model = Seq2Seq(encoder, decoder, device)

    optimizer = optim.Adam(decoder.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index = 0, reduction='none')  # Pad Index


    if checkpoint is not None:
        last_epoch, model_dict, optim_dict, valid_loss_list, train_loss_list = load_checkpoint(checkpoint)
        last_epoch += 1
        model.load_state_dict(model_dict)
        best_valid_loss = min(valid_loss_list)

        optimizer.load_state_dict(optim_dict)
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

        print(f'Using Checkpoint')
    else:
        last_epoch = 0
        valid_loss_list, train_loss_list = [], []
        model.apply(init_weights)

    model.to(device)

    for epoch in range(last_epoch, epochs):
        start_time = time.time()

        train_loss = train(model, training_loader, optimizer, criterion, clip)
        valid_loss = eval(model, valid_loader, criterion)

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        #     if valid_loss < best_valid_loss:
        #         best_valid_loss = valid_loss
        save_checkpoint(model_path/f'model0epoch{epoch}', epoch, model, optimizer, valid_loss_list, train_loss_list)

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')