from pathlib import Path

import torch
from transformers import BertModel, BertConfig


#runtime environment
device = 'cuda' if torch.cuda.is_available() else 'cpu'

bert_model = 'bert-base-cased'
#paths
squad_path = Path('../data/squad')
bert_path = Path('../data/bert')/bert_model
model_path = Path('../data/model/')

stage = 'stage_one'

#encoder parameter

try:
    encoder = BertModel.from_pretrained(model_path/stage/bert_model)
except OSError:
    encoder = BertModel.from_pretrained(bert_model)
    encoder.save_pretrained(model_path/stage/bert_model)

for p in encoder.parameters():
    p.requires_grad = False

bert_hidden_size = encoder.config.hidden_size
bert_vocab_size = encoder.config.vocab_size

#decoder parameter
decoder_hidden_size = 512
decoder_input_size = 512  # embedding dimesions
clip = 1
dropout = 0.5

# training parameters
epochs = 4
mb = 8
dl_workers = 1
checkpoint = None