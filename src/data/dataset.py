import json
import pickle
from pprint import pprint

import numpy as np
import torch
from pytorch_transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader


class BertDataset(Dataset):

    def __init__(self, path):
        with open(path, 'rb') as read_file:
            self.data = pickle.load(read_file)

    def __len__(self):
        return self.data['input_ids'].shape[0]

    def __getitem__(self, idx):
        return (((self.data['input_ids'][idx],
                self.data['attention_mask'][idx],
                self.data['token_type_ids'][idx]),
                self.data['input_len'][idx]),
                (self.data['output_ids'][idx],
                self.data['output_len'][idx])
                )


if __name__ == '__main__':
    ds = BertDataset('../../data/bert/toy')
    #train_set, batch_size=mb, shuffle=True, num_workers=dl_workers, pin_memory=True if device=='cuda' else False
    dl = DataLoader(ds, batch_size=8, num_workers=1, shuffle=True, pin_memory=True)
    src, dest = next(iter(dl))
    src, src_len = src

    print(src[0][0].shape)
    print(src[1][0].shape)
    print(src[2][0].shape)



