# SentClass
Sentence classification on the Sentihood dataset.

# Training
Run the models with the following command:
```
python main.py --flat-data --model {lstmfinal | crflstmdiag | crflstmlstm | crfemblstm} {--save} {--devid #}
```
Add the `--save` option to save checkpoints along with their valid and test performance.
Models are saved based on validation accuracy.
Use the `--devid #` option to utilize a GPU.

# Data
Data was obtained from [jack](https://github.com/uclmr/jack/tree/master/data/sentihood).
