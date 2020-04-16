import json
import pickle

from transformers import BertTokenizer


class Preprocess:

    def __init__(self, squad_path, bert_model):
        with open(squad_path, 'r') as read_file:
            data = json.load(read_file)
        input, output = _extract_squad_data(data)
        self.data = _tokenize_data(input, output, bert_model)


    def save(self, path):
        with open(path, 'wb') as write_file:
            pickle.dump(self.data, write_file)


def _extract_squad_data(data):
    data = data['data']

    input = []
    output = []
    for document in data:
        for paragraph in document['paragraphs']:
            context = paragraph['context']
            for qas in paragraph['qas']:
                answer = qas['answers'][0]['text']
                question = qas['question']
                input.append((context, answer))
                output.append(question)
                # qui si possono inserire condizioni per limitare la grandezza del dataset
    input = input[:int(0.1 * len(input))]  # prende il 10% di tutto il dataset
    output = output[:int(0.1 * len(output))]
    return input, output


def _tokenize_data(input, output, bert_model):
    tokenizer = BertTokenizer.from_pretrained(bert_model)

    data = tokenizer.batch_encode_plus(input, pad_to_max_length=True, return_tensors='pt')
    out_dict = tokenizer.batch_encode_plus(output, pad_to_max_length=True, return_tensors='pt')

    data['output_ids'] = out_dict['input_ids']
    data['output_len'] = out_dict['attention_mask'].sum(dim=1)
    data['input_len'] = data['attention_mask'].sum(dim=1)

    idx = (data['input_len'] <= 512)
    in_m = max(data['input_len'][idx])
    out_m = max(data['output_len'][idx])

    data['input_ids'] = data['input_ids'][idx, :in_m]
    data['attention_mask'] = data['attention_mask'][idx, :in_m]
    data['token_type_ids'] = data['token_type_ids'][idx, :in_m]
    data['input_len'] = data['input_len'][idx]

    data['output_ids'] = data['output_ids'][idx, :out_m]
    data['output_len'] = data['output_len'][idx]

    return data



if __name__ == '__main__':
    pass
    # dataset = Preprocess('../data/squad/squad-v1.1-train.json', bert_model)
    # dataset.save(f'../data/bert/{bert_model}/train')
