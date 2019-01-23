# SentClass
Sentence classification on the Sentihood dataset with a couple CRF models.

## Requirements
Code is tested with [torch](https://github.com/pytorch/pytorch) 1.0.0, [pyro](https://github.com/uber/pyro) 0.3.0,
and [torchtext](https://github.com/pytorch/text) 0.4.0.

## Data
Data was obtained from [jack](https://github.com/uclmr/jack/tree/master/data/sentihood).

## Training
Run the models with the following command:
```
python main.py --flat-data --model {lstmfinal | crflstmdiag | crflstmlstm | crfemblstm} {--save} {--devid #}
```
Add the `--save` option to save checkpoints along with their valid and test performance.
Be sure to create the `save/{modelname}` directory before saving.
Models are saved based on validation accuracy.

Use the `--devid #` option to utilize a GPU.

