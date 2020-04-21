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
