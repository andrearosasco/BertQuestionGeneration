# BertQuestionGeneration
This is a university project for the course Human Language Technology, University of Pisa.
The aim is to train Question Generation model on the SQuAD dataset using BERT as the encoder.

## Installation
After cloning the repository you will need to install the PyTorch framework and the library transformers by HuggingFace.

## Usage
To run the code move to the src directory and run
```bash
python main.py
```
At the first run, this will preprocess the SQuAD dataset for BERT and save the new version in a directory. Also, the BERT weights will be downloaded and saved to speed subsequent runs.
After each epoch the weight of the whole model and the optimizer state are saved in the directory specified by the paths in ```config.py```.
> **WARNING**: If you resume the training from a checkpoint you should be aware that the data loader random shuffle will be repeated as if it was just initialized. To avoid it you can modify the seed in ```main.py```

## Config
When you run the code all the hyperparameters are read from the file ```config.py```. Ideally that should be the only file to modify to try new configuration. For more complex configuration tough (e.g changing the optimizer) you will have to modify ```main.py ``` or the model classes.
> **WARNING**: The teacher-forcing parameter is passed directly to the forward method of the Seq2Seq model so that it can easily be deactivated during evaluation (this will be fixed). For now you have to change the parameter where the forward method is called (i.e. ```train.py``` and ```eval.py```)

## Model
For the encoder part I use the implementation of BERT provided by HuggingFace.
The decoder instead is implemented by a GRU with additive attention. The attention is performed between the "higher" hidden state of the GRU and the word encoding extracted from the last layer of BERT. The result of the attention is concatenated to the hidden state and passed in input to a softmax layer to get the next word.

## Results

### Config 1
```python
#optimizer Adam
weight_decay = 0
lr = 1e-3
betas = (0.9, 0.999)

#encoder
bert_model = 'bert-large-cased'

# decoder parameter
decoder_hidden_size = 512
decoder_input_size = 512
attention_hidden_size = 512
num_layers = 1
clip = 1
input_dropout = dropout = 0.5
teacher_forcing = 0.5

# training parameters
epochs = 2
mb = 32
dl_workers = 0
checkpoint = None
encoder_trained = False
```
|   | Train | Valid |
|---|:-----:|-------|
| 1 | 6.088 | 6.508 |
| 2 | 5.543 | 6.530 |

### Config 2
```python
#optimizer Adam
weight_decay = 0
lr = 1e-3
betas = (0.9, 0.999)

#encoder
bert_model = 'bert-large-cased'

# decoder parameter
decoder_hidden_size = 1024
decoder_input_size = 1024
attention_hidden_size = 1024
num_layers = 1
clip = 1
input_dropout = dropout = 0.5
teacher_forcing = 0.5

# training parameters
epochs = 2
mb = 32
dl_workers = 0
checkpoint = None
encoder_trained = False
```
|   | Train | Valid |
|---|:-----:|-------|
| 1 | 5.913 | 6.461 |
| 2 | 5.403 | 6.488 |

### Config 3
```python
#optimizer AdamW
weight_decay = 0.01
lr = 1e-3
betas = (0.9, 0.999)

#encoder
bert_model = 'bert-large-cased'

# decoder parameter
decoder_hidden_size = 512
decoder_input_size = 512
attention_hidden_size = 512
num_layers = 1
clip = 1
input_dropout = dropout = 0.5
teacher_forcing = 0.5

# training parameters
epochs = 2
mb = 32
dl_workers = 0
checkpoint = None
encoder_trained = False
```
|   | Train | Valid |
|---|:-----:|-------|
| 1 | 6.176 | 6.466 |
| 2 | 5.578 | 6.465 |

### Config 4
```python
#optimizer AdamW
weight_decay = 0.01
lr = 1e-3
betas = (0.9, 0.999)

#encoder
bert_model = 'bert-large-cased'

# decoder parameter
decoder_hidden_size = 1024
decoder_input_size = 1024
attention_hidden_size = 1024
num_layers = 1
clip = 1
input_dropout = dropout = 0.5
teacher_forcing = 0.5

# training parameters
epochs = 2
mb = 32
dl_workers = 0
checkpoint = None
encoder_trained = False
```
|   | Train | Valid |
|---|:-----:|-------|
| 1 | 5.902 | 6.495 |
| 2 | 5.398 | 6.530 |

### Config 5
```python
#optimizer AdamW
weight_decay = 0.01
lr = 1e-3
betas = (0.9, 0.999)

#encoder
bert_model = 'bert-large-cased'

# decoder parameter
decoder_hidden_size = 512
decoder_input_size = 512
attention_hidden_size = 512
num_layers = 1
clip = 1
input_dropout = dropout = 0
teacher_forcing = 0.5

# training parameters
epochs = 2
mb = 32
dl_workers = 0
checkpoint = None
encoder_trained = False
```
|   | Train | Valid |
|---|:-----:|-------|
| 1 | 6.129 | 6.451 |
| 2 | 5.539 | 6.489 |

### Config 6
```python
#optimizer AdamW
weight_decay = 0.05
lr = 1e-3
betas = (0.9, 0.999)

#encoder
bert_model = 'bert-large-cased'

# decoder parameter
decoder_hidden_size = 512
decoder_input_size = 512
attention_hidden_size = 512
num_layers = 1
clip = 1
input_dropout = dropout = 0.5
teacher_forcing = 0.5

# training parameters
epochs = 4
mb = 32
dl_workers = 0
checkpoint = None
encoder_trained = False
```
|   | Train | Valid |
|---|:-----:|-------|
| 1 | 6.156 | 6.455 |
| 2 | 5.588 | 6.501 |
| 3 | 5.392 | 6.505 |
| 4 | 5.235 | 6.515 |

### Config 7
```python
#optimizer AdamW
weight_decay = 0.1
lr = 1e-3
betas = (0.9, 0.999)

#encoder
bert_model = 'bert-large-cased'

# decoder parameter
decoder_hidden_size = 512
decoder_input_size = 512
attention_hidden_size = 512
num_layers = 1
clip = 1
input_dropout = dropout = 0.5
teacher_forcing = 0.5

# training parameters
epochs = 2
mb = 32
dl_workers = 0
checkpoint = None
encoder_trained = False
```
|   | Train | Valid |
|---|:-----:|-------|
| 1 | 6.158 | 6.447 |
| 2 | 5.582 | 6.474 |

### Config 8
```python
#optimizer AdamW
weight_decay = 0.5
lr = 1e-3
betas = (0.9, 0.999)

#encoder
bert_model = 'bert-large-cased'

# decoder parameter
decoder_hidden_size = 512
decoder_input_size = 512
attention_hidden_size = 512
num_layers = 2
clip = 1
input_dropout = 0
dropout = 0.5
teacher_forcing = 0.5

# training parameters
epochs = 5
mb = 32
dl_workers = 0
checkpoint = None
encoder_trained = False
```
|   | Train | Valid |
|---|:-----:|-------|
| 1 | 6.692 | 6.687 |
| 2 | 6.459 | 6.590 |
| 3 | 6.064 | 6.560 |
| 4 | 5.704 | 6.500 |
| 5 | 5.528 | 6.518 |

### Config 9
```python
#optimizer AdamW
weight_decay = 0.05
betas = (0.9, 0.99)
lr = 4e-4

#encoder
bert_model = 'bert-large-cased'

# decoder parameter
decoder_hidden_size = 512
decoder_input_size = 512
attention_hidden_size = 512
num_layers = 1
clip = 1
dropout = 0.15

# training parameters
epochs = 4
mb = 32
dl_workers = 0
checkpoint = None
encoder_trained = False
```
|   | Train | Valid |
|---|:-----:|-------|
| 1 | 6.336 | 6.471 |
| 2 | 5.798 | 6.488 |
| 3 | 5.634 | 6.503 |
| 4 | 5.525 | 6.524 |
