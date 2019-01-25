 
import argparse

import torch
import torch.optim as optim

from torchtext.data import BucketIterator
from torchtext.vocab import GloVe

from sentclass.models.lstmfinal import LstmFinal
from sentclass.models.crflstmdiag import CrfLstmDiag
from sentclass.models.crfemblstm import CrfEmbLstm
from sentclass.models.crflstmlstm import CrfLstmLstm
from sentclass.models.crfneg import CrfNeg

import json

torch.backends.cudnn.enabled = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filepath",
        default="data",
        type=str,
    )

    parser.add_argument("--devid", default=-1, type=int)

    parser.add_argument("--flat-data", action="store_true", default=False)
    parser.add_argument("--data", choices=["sentihood",]) 

    parser.add_argument("--bsz", default=33, type=int)
    parser.add_argument("--ebsz", default=150, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--once", action="store_true")

    parser.add_argument("--clip", default=5, type=float)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--lrd", default=1, type=float)
    parser.add_argument("--pat", default=0, type=int)
    parser.add_argument("--dp", default=0.2, type=float)
    parser.add_argument("--wdp", default=0, type=float)
    parser.add_argument("--wd", default=1e-4, type=float)

    parser.add_argument("--optim", choices=["Adam", "SGD"])

    # Adam
    parser.add_argument("--b1", type=float, default=0.9)
    parser.add_argument("--b2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)

    # SGD
    parser.add_argument("--mom", type=float, default=0)
    parser.add_argument("--dm", type=float, default=0)
    parser.add_argument("--nonag", action="store_true", default=False)

    # Model
    parser.add_argument(
        "--model",
        choices=[
            "lstmfinal", "crflstmdiag", "crfemblstm", "crflstmlstm", "crfneg",
        ],
        default="lstmfinal"
    )

    parser.add_argument("--nlayers", default=2, type=int)
    parser.add_argument("--emb-sz", default=300, type=int)
    parser.add_argument("--rnn-sz", default=50, type=int)

    parser.add_argument("--save", action="store_true")

    parser.add_argument("--re", default=100, type=int)

    parser.add_argument("--seed", default=1111, type=int)
    return parser.parse_args()


args = get_args()
print(args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

device = torch.device(f"cuda:{args.devid}" if args.devid >= 0 else "cpu")

# Data
import sentclass.sentihood as data
from sentclass.sentihood import RandomIterator

TEXT, LOCATION, ASPECT, SENTIMENT = data.make_fields()
train, valid, test = data.SentihoodDataset.splits(
    TEXT, LOCATION, ASPECT, SENTIMENT, flat=args.flat_data, path=args.filepath)

data.build_vocab(TEXT, LOCATION, ASPECT, SENTIMENT, train, valid, test)
TEXT.vocab.load_vectors(vectors=GloVe(name="840B"))
TEXT.vocab.vectors[TEXT.vocab.stoi["transit-location"]] = (
    (TEXT.vocab.vectors[TEXT.vocab.stoi["transit"]] +
        TEXT.vocab.vectors[TEXT.vocab.stoi["location"]]) / 2
)

iterator = BucketIterator if not args.flat_data else RandomIterator

train_iter, valid_iter, test_iter = iterator.splits(
    (train, valid, test),
    batch_sizes = (args.bsz, args.ebsz, args.ebsz),
    device = device,
    repeat = False,
    sort_within_batch = True,
)
full_train_iter = RandomIterator(
    dataset = train,
    batch_size = args.ebsz,
    device = device,
    repeat = False,
    sort_within_batch = True,
    train = False,
)

asp_iterator = BucketIterator
asp_train, asp_valid, asp_test = data.SentihoodDataset.splits(
    TEXT, LOCATION, ASPECT, SENTIMENT, flat=False, path=args.filepath)
asp_train_iter, asp_valid_iter, asp_test_iter = asp_iterator.splits(
    (asp_train, asp_valid, asp_test),
    batch_size = args.ebsz,
    device = device,
    repeat = False,
    sort_within_batch = True,
)

# Model
if args.model == "lstmfinal":
    assert(args.flat_data)
    model = LstmFinal(
        V       = TEXT.vocab,
        L       = LOCATION.vocab if LOCATION is not None else None,
        A       = ASPECT.vocab,
        S       = SENTIMENT.vocab,
        emb_sz  = args.emb_sz,
        rnn_sz  = args.rnn_sz,
        nlayers = args.nlayers,
        dp      = args.dp,
    )
elif args.model == "crflstmdiag":
    assert(args.flat_data)
    model = CrfLstmDiag(
        V       = TEXT.vocab,
        L       = LOCATION.vocab if LOCATION is not None else None,
        A       = ASPECT.vocab,
        S       = SENTIMENT.vocab,
        emb_sz  = args.emb_sz,
        rnn_sz  = args.rnn_sz,
        nlayers = args.nlayers,
        dp      = args.dp,
    )
elif args.model == "crfemblstm":
    assert(args.flat_data)
    model = CrfEmbLstm(
        V       = TEXT.vocab,
        L       = LOCATION.vocab if LOCATION is not None else None,
        A       = ASPECT.vocab,
        S       = SENTIMENT.vocab,
        emb_sz  = args.emb_sz,
        rnn_sz  = args.rnn_sz,
        nlayers = args.nlayers,
        dp      = args.dp,
    )
elif args.model == "crflstmlstm":
    assert(args.flat_data)
    model = CrfLstmLstm(
        V       = TEXT.vocab,
        L       = LOCATION.vocab if LOCATION is not None else None,
        A       = ASPECT.vocab,
        S       = SENTIMENT.vocab,
        emb_sz  = args.emb_sz,
        rnn_sz  = args.rnn_sz,
        nlayers = args.nlayers,
        dp      = args.dp,
    )
elif args.model == "crfneg":
    assert(args.flat_data)
    model = CrfNeg(
        V       = TEXT.vocab,
        L       = LOCATION.vocab if LOCATION is not None else None,
        A       = ASPECT.vocab,
        S       = SENTIMENT.vocab,
        emb_sz  = args.emb_sz,
        rnn_sz  = args.rnn_sz,
        nlayers = args.nlayers,
        dp      = args.dp,
    )
elif args.model == "crfsimple":
    model = CrfSimple(
        V       = TEXT.vocab,
        L       = LOCATION.vocab,
        A       = ASPECT.vocab,
        S       = SENTIMENT.vocab,
        emb_sz  = args.emb_sz,
        rnn_sz  = args.rnn_sz,
        nlayers = args.nlayers,
        dp      = args.dp,
    )

model.to(device)
print(model)

params = list(model.parameters())

optimizer = optim.Adam(
    params, lr = args.lr, weight_decay = args.wd, betas=(args.b1, args.b2))

best_val = 0
for e in range(args.epochs):
    print(f"Epoch {e} lr {optimizer.param_groups[0]['lr']}")
    train_iter.init_epoch()
    # Train
    train_loss, tntok = model.train_epoch(
        diter     = train_iter,
        clip      = args.clip,
        re        = args.re,
        optimizer = optimizer,
        once      = args.once,
    )

    # Validate
    valid_loss, ntok = model.validate(valid_iter)

    # Accuracy on train
    train_acc = model.acc(full_train_iter)
    # Accuracy on Valid
    valid_acc = model.acc(valid_iter, skip0=True)
    valid_f1 = model.f1(asp_valid_iter)
    test_acc = model.acc(test_iter, skip0=True)
    test_f1 = model.f1(asp_test_iter)

    # Report
    print(f"Epoch {e}")
    print(f"train loss: {train_loss / tntok} train acc: {train_acc}")
    print(f"valid loss: {valid_loss / ntok} valid acc: {valid_acc} valid f1: {valid_f1}")
    print(f"test acc: {test_acc} test f1: {test_f1}")

    if args.save and valid_acc > best_val:
        best_val = valid_acc
        savestring = f"saves/{args.model}/{args.model}-lr{args.lr}-nl{args.nlayers}-rnnsz{args.rnn_sz}-dp{args.dp}-va{valid_acc}-vf{valid_f1}-ta{test_acc}-tf{test_f1}.pt"
        torch.save(model, savestring)
