import argparse

import math

import torchtext
from torchtext.vocab import Vectors, GloVe

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.nn.utils import clip_grad_norm

from tqdm import tqdm

import random

random.seed(1111)
torch.manual_seed(1111)
torch.cuda.manual_seed_all(1111)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=-1)

    parser.add_argument("--nhid", type=int, default=256)
    parser.add_argument("--nlayers", type=int, default=1)
    parser.add_argument("--xgiveny", action="store_true")

    parser.add_argument("--tieweights", action="store_true")

    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--optim", choices=["SGD", "Adam"], default="Adam")

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lrd", type=float, default=0.9)
    parser.add_argument("--wd", type=float, default=1e-4)

    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--clip", type=float, default=5)

    # Adam parameters
    parser.add_argument("--b1", type=float, default=0.9)
    parser.add_argument("--b2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)

    # SGD parameters
    parser.add_argument("--mom", type=float, default=0.99)
    parser.add_argument("--dm", type=float, default=0)
    parser.add_argument("--nonag", action="store_true", default=False)

    # Execution options
    parser.add_argument("--evaluatemodel", type=str, default=None)
    parser.add_argument("--savemodel", action="store_true")

    args = parser.parse_args()
    return args

args = parse_args()

# Our input $x$
TEXT = torchtext.data.Field()

# Our labels $y$
LABEL = torchtext.data.Field(sequential=False)
train, valid, test = torchtext.datasets.SST.splits(
    TEXT, LABEL,
    filter_pred=lambda ex: ex.label != 'neutral')

print('len(train)', len(train))
print('vars(train[0])', vars(train[0]))

TEXT.build_vocab(train)
vsize = len(TEXT.vocab.itos)
bos = "<bos>"
TEXT.vocab.itos.append(bos)
TEXT.vocab.stoi[bos] = vsize
eos = "<eos>"
TEXT.vocab.itos.append(eos)
TEXT.vocab.stoi[eos] = vsize + 1
positive = "<positive>"
TEXT.vocab.itos.append(positive)
TEXT.vocab.stoi[positive] = vsize + 2
negative= "<negative>"
TEXT.vocab.itos.append(negative)
TEXT.vocab.stoi[negative] = vsize + 3
assert(vsize + 4 == len(TEXT.vocab.itos))
vsize = len(TEXT.vocab.itos)

pad_id = TEXT.vocab.stoi["<pad>"]
unk_id = TEXT.vocab.stoi["<unk>"]
bos_id = TEXT.vocab.stoi[bos]
eos_id = TEXT.vocab.stoi[eos]
positive_id = TEXT.vocab.stoi[positive]
negative_id = TEXT.vocab.stoi[negative]

LABEL.build_vocab(train)
LABEL.vocab.itos = ['positive', 'negative']
LABEL.vocab.stoi['positive'] = 0
LABEL.vocab.stoi['negative'] = 1

print('len(TEXT.vocab)', len(TEXT.vocab))
print('len(LABEL.vocab)', len(LABEL.vocab))

labels = [ex.label for ex in train.examples]

train_iter, _, _ = torchtext.data.BucketIterator.splits(
    (train, valid, test), batch_size=args.bsz, device=-1, repeat=False)

_, valid_iter, test_iter = torchtext.data.BucketIterator.splits(
    (train, valid, test), batch_size=10, device=-1)

# Build the vocabulary with word embeddings
url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))
#simple_vec = TEXT.vocab.vectors.clone()

#url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec'
#TEXT.vocab.load_vectors(vectors=Vectors('wiki.en.vec', url=url))
#complex_vec = TEXT.vocab.vectors
#
def output_test(model):
    "All models should be able to be run with following command."
    upload = []
    loss.reduce = False
    for batch in test_iter:
        # Your prediction data here (don't cheat!)
        x = batch.text
        y = batch.label
        x, y = model.prep_sample(x, y)
        preds = model(x)
        # if the score of the positive is higher than negative
        # we want to predict the label 0
        yhat = preds[-1, :, positive_id] < preds[-1, :, negative_id] 
        upload += yhat.tolist()
    loss.reduce = True
    with open(args.model + ".txt", "w") as f:
        f.write("Id,Cat\n")
        for i,u in enumerate(upload):
            f.write(str(i) + "," + str(u+1) + "\n")

def output_test_gen(model):
    "All models should be able to be run with following command."
    upload = []
    loss.reduce = False
    for batch in test_iter:
        # Your prediction data here (don't cheat!)
        x = batch.text
        y = batch.label
        x, y = model.prep_sample(x, y)
        x[0,:].fill_(positive_id)
        pospreds = model(x)
        x[0,:].fill_(negative_id)
        negpreds = model(x)
        T, N, V = pospreds.size()
        # negative log probabilities? or something lol
        nlpxgivenpos = loss(pospreds.view(-1, V), y.view(-1)).view(T, N).sum(0)
        nlpxgivenneg = loss(negpreds.view(-1, V), y.view(-1)).view(T, N).sum(0)
        # if the score of the positive is higher than negative
        # we want to predict the label 0
        yhat = nlpxgivenpos > nlpxgivenneg
        upload += yhat.tolist()
    loss.reduce = True
    with open(args.model + ".txt", "w") as f:
        f.write("Id,Cat\n")
        for i,u in enumerate(upload):
            f.write(str(i) + "," + str(u+1) + "\n")

def validate(model, valid):
    model.eval()
    correct = 0.
    total = 0.
    with torch.no_grad():
        for batch in valid:
            x = batch.text
            y = batch.label
            x, y = model.prep_sample(x, y)
            preds = model(x)
            # if the score of the positive is higher than negative
            # we want to predict the label 0
            yhat = preds[-1, :, positive_id] < preds[-1, :, negative_id] 
            results = yhat.long().cpu() == batch.label
            correct += results.float().sum().data[0]
            total += results.size(0)
    return correct, total, correct / total

def validate_gen(model, valid):
    model.eval()
    correct = 0.
    total = 0.
    loss.reduce = False
    with torch.no_grad():
        for batch in valid:
            x = batch.text
            y = batch.label
            x, y = model.prep_sample(x, y)
            x[0,:].fill_(positive_id)
            pospreds = model(x)
            x[0,:].fill_(negative_id)
            negpreds = model(x)
            T, N, V = pospreds.size()
            # negative log probabilities? or something lol
            nlpxgivenpos = loss(pospreds.view(-1, V), y.view(-1)).view(T, N).sum(0)
            nlpxgivenneg = loss(negpreds.view(-1, V), y.view(-1)).view(T, N).sum(0)
            # if the score of the positive is higher than negative
            # we want to predict the label 0
            yhat = nlpxgivenpos > nlpxgivenneg
            results = yhat.long().cpu() == batch.label
            correct += results.float().sum().data[0]
            total += results.size(0)
    loss.reduce = True
    return correct, total, correct / total


# Models
class LstmGen(nn.Module):
    # ignore nhid, lol
    def __init__(self, vocab, nhid, nlayers, tie_weights, dropout=0, xgiveny=False):
        super(LstmGen, self).__init__()
        self.vsize = len(vocab.itos)
        self.bos = vocab.stoi[bos]
        self.eos = vocab.stoi[eos]
        self.positive = vocab.stoi[positive]
        self.negative = vocab.stoi[negative]

        self.xgiveny = xgiveny

        self.lut = nn.Embedding(self.vsize, nhid)
        self.rnn = nn.LSTM(nhid, nhid, nlayers, bidirectional=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(nhid, self.vsize)
        if tie_weights:
            self.tie_weights = tie_weights
            self.decoder.weight = self.lut.weight

        self.buffer = torch.LongTensor().cuda(args.gpu)
        # Pinned memory cannot be resized.
        self.workbuffer = torch.LongTensor()

    def forward(self, input):
        vectors = self.lut(input)
        out, (h, c) = self.rnn(vectors)
        droppedout = self.dropout(out)
        return self.decoder(droppedout)

    def prep_sample(self, x, y):
        T, N = x.size()
        if self.xgiveny:
            # Samples should look like
            #     a b c d
            # coming in, but
            #     <positive> a b c d <eos>
            # coming out.
            self.workbuffer.resize_(T+2, N)
            self.workbuffer[1:-1,:].copy_(x.data)
            self.workbuffer[0].masked_fill_(y.data.eq(0), self.positive)
            self.workbuffer[0].masked_fill_(y.data.eq(1), self.negative)
            self.workbuffer[-1,:].fill_(self.eos)
            self.buffer.resize_(self.workbuffer.size())
            self.buffer.copy_(self.workbuffer, async=True)
            return V(self.buffer[:-1,:]), V(self.buffer[1:,:])

        else:
            # Samples should look like
            #     a b c d
            # coming in, but
            #     <bos> a b c d <eos> <positive>
            # coming out.
            self.workbuffer.resize_(T+3, N)
            self.workbuffer[1:-2,:].copy_(x.data)
            self.workbuffer[0,:].fill_(self.bos)
            self.workbuffer[-2,:].fill_(self.eos)
            self.workbuffer[-1].masked_fill_(y.data.eq(0), self.positive)
            self.workbuffer[-1].masked_fill_(y.data.eq(1), self.negative)
            self.buffer.resize_(self.workbuffer.size())
            self.buffer.copy_(self.workbuffer, async=True)
            return V(self.buffer[:-1,:]), V(self.buffer[1:,:])


def save_model(model, valid, epoch, nlayers, dropout, lr, lrd):
    name = "generative_{}_{}_valid_{}_nl{}_do{}_lr{}_lrd{}".format(
        "xgiveny" if args.xgiveny else "ygivenx", epoch, valid, nlayers, dropout, lr, lrd)
    torch.save(model.cpu().state_dict(), name)
    # lol, whatever
    model.cuda(args.gpu)

def train_model(model, valid_fn, loss=nn.CrossEntropyLoss(), epochs=args.epochs, lr=args.lr):
    params = [p for p in model.parameters() if p.requires_grad]

    if args.optim == "SGD":
        optimizer = optim.SGD(
            params, lr = lr, weight_decay = args.wd, momentum=args.mom, dampening=args.dm, nesterov=not args.nonag)
    elif args.optim == "Adam":
        optimizer = optim.Adam(params, lr = lr, weight_decay = args.wd, amsgrad=False)

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.lrd)
    for epoch in range(epochs):
        scheduler.step()
        model.train()
        train_loss = 0
        print()
        print("Epoch {} | lr {}".format(epoch, scheduler.get_lr()))
        gradnorms = torch.FloatTensor(len(train_iter)) 
        for i, batch in enumerate(tqdm(train_iter)):
            x = batch.text
            y = batch.label
            x, y = model.prep_sample(x, y)
            def closure():
                nonlocal train_loss
                optimizer.zero_grad()
                bloss = loss(model(x).view(-1, vsize), y.view(-1))
                bloss.backward()
                gradnorms[i] = nn.utils.clip_grad_norm(params, args.clip)
                train_loss += bloss
                return bloss
            optimizer.step(closure)
        train_loss /= len(train_iter)
        print("Train loss: " + str(train_loss.data[0]))
        print("Max grad norm: {}, Avg grad norm: {}".format(gradnorms.max(), gradnorms.mean()))
        #train_acc = valid_fn(model, train_iter)
        #print("Train acc: " + str(train_acc))
        valid_acc = valid_fn(model, valid_iter)
        print("Valid acc: " + str(valid_acc))
        if args.savemodel:
            save_model(model, valid_acc[-1], epoch, args.nlayers, args.dropout, args.lr, args.lrd)

model = LstmGen(
    TEXT.vocab, args.nhid, args.nlayers, args.tieweights, args.dropout, args.xgiveny)
if args.gpu > 0:
    model.cuda(args.gpu)
print(model)

weight = torch.Tensor(vsize).fill_(1)
weight[pad_id] = 0 
weight[positive_id] = 5 
weight[negative_id] = 5
if args.gpu >= 0:
    weight = weight.cuda(args.gpu)
loss = nn.CrossEntropyLoss(weight=weight)

if args.evaluatemodel:
    model.load_state_dict(torch.load(args.evaluatemodel))
    if args.xgiveny:
        output_test_gen(model)
    else:
        output_test(model)
else:
    train_model(model, validate if not args.xgiveny else validate_gen, loss=loss, epochs=args.epochs, lr=args.lr)
    if args.xgiveny:
        _, _, train_acc = validate_gen(model, train_iter)
        _, _, valid_acc = validate_gen(model, valid_iter)
        _, _, test_acc = validate_gen(model, test_iter)
    else:
        _, _, train_acc = validate(model, train_iter)
        _, _, valid_acc = validate(model, valid_iter)
        _, _, test_acc = validate(model, test_iter)
    print("train: {}, valid: {}, test: {}".format(train_acc, valid_acc, test_acc))
