import json
from pathlib import Path
import torch
import logging

from transformers import BertModel

from data import Preprocess


def setup(bert_model, model_path, stage, squad_path, bert_path):
    log = logging.getLogger('QGModel')

    file = Path('.setup').open('a+')
    file.seek(0, 0)

    if bert_model in (file.readline().split()):
        file.close()
        log.info(f'Setup: {bert_model} setup already performed')
        return
    file.write(f' {bert_model} ')
    file.close()

    log.info(f'Setup: downloading {bert_model}')
    BertModel.from_pretrained(bert_model).save_pretrained(model_path/stage/bert_model)


    log.info(f'Setup: preprocessing {bert_model} input')
    for x in squad_path.iterdir():
        if x.is_file():
            dataset = Preprocess(squad_path/x.name, bert_model)
            dataset.save(bert_path/bert_model/x.name[11:-5])
    log.info(f'Setup: {bert_model} setup completed')




#runtime environment
device = 'cuda' if torch.cuda.is_available() else 'cpu'


#paths
squad_path = Path('../data/squad')
bert_path = Path('../data/bert')
model_path = Path('../data/model/')

stage = 'stage_one'
bert_model = 'bert-large-cased'

# if not present download the right bert version and preprocess and save the dataset
setup(bert_model, model_path, stage, squad_path, bert_path)

#encoder parameter

with (model_path/stage/bert_model/'config.json').open('r') as f:
    conf = json.load(f)
    bert_hidden_size = conf['hidden_size']
    bert_vocab_size = conf['vocab_size']

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
