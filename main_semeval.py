# python main.py --devid 3 --bsz 32 --ebsz 32 --rnn-sz 50 --lr 0.01 --dp 0.2 --flat-data --nlayers 2 --clip 5 --lrd 0.8 --epochs 1000 --model boring
 
import argparse

import torch
import torch.optim as optim

from torchtext.data import BucketIterator
from torchtext.vocab import GloVe

from sentclass.models.boring import Boring
from sentclass.models.crfnb import CrfNb
from sentclass.models.crfsimple import CrfSimple

import json

#torch.set_anomaly_enabled(True)
#torch.backends.cudnn.enabled = False
torch.backends.cudnn.enabled = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filepath",
        #default="data",
        default="~/research/GCAE/acsa-restaurant-large",
        #default="~/research/GCAE/acsa-restaurant-2014",
        type=str,
    )

    parser.add_argument("--devid", default=-1, type=int)

    parser.add_argument("--flat-data", action="store_true", default=False)
    parser.add_argument("--data", choices=["sentihood", "semeval"]) # use acsa vs atsa? lol same shit

    parser.add_argument("--bsz", default=48, type=int)
    parser.add_argument("--ebsz", default=48, type=int)
    parser.add_argument("--epochs", default=32, type=int)

    parser.add_argument("--clip", default=5, type=float)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--lrd", default=0.3, type=float)
    parser.add_argument("--pat", default=0, type=int)
    parser.add_argument("--dp", default=0.1, type=float)
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
        choices=["boring", "crfnb", "crfsimple"],
        default="boring"
    )

    parser.add_argument("--nlayers", default=2, type=int)
    parser.add_argument("--emb-sz", default=300, type=int)
    parser.add_argument("--rnn-sz", default=50, type=int)

    parser.add_argument("--tieweights", action="store_true", default=False)

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
import sentclass.semeval as data

TEXT, LOCATION, ASPECT, SENTIMENT = data.make_fields()
train, valid, test = data.SemevalDataset.splits(
    TEXT, ASPECT, SENTIMENT, flat=args.flat_data, path=args.filepath,
    #train="acsa_hard_train.json", test="acsa_hard_test.json",
    train="acsa_train.json.train", validation="acsa_train.json.valid", test="acsa_test.json",
)
data.build_vocab(TEXT, ASPECT, SENTIMENT, train, valid, test)
TEXT.vocab.load_vectors(vectors=GloVe(name="840B"))

train_iter, valid_iter, test_iter = BucketIterator.splits(
    (train, valid, test),
    batch_sizes = (args.bsz, args.ebsz, args.ebsz),
    device = device,
    repeat = False,
    sort_within_batch = True,
    #sort_key = already given in dataset?
)
full_train_iter = train_iter

# Model
if args.model == "boring":
    assert(args.flat_data)
    model = Boring(
        V       = TEXT.vocab,
        L       = LOCATION.vocab if LOCATION is not None else None,
        A       = ASPECT.vocab,
        S       = SENTIMENT.vocab,
        emb_sz  = args.emb_sz,
        rnn_sz  = args.rnn_sz,
        nlayers = args.nlayers,
        dp      = args.dp,
        tieweights = args.tieweights,
    )
if args.model == "crfnb":
    assert(args.flat_data)
    model = CrfNb(
        V       = TEXT.vocab,
        L       = LOCATION.vocab if LOCATION is not None else None,
        A       = ASPECT.vocab,
        S       = SENTIMENT.vocab,
        emb_sz  = args.emb_sz,
        rnn_sz  = args.rnn_sz,
        nlayers = args.nlayers,
        dp      = args.dp,
        tieweights = args.tieweights,
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
schedule = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=args.pat, factor=args.lrd, threshold=1e-3)

best_val = float("inf")
for e in range(args.epochs):
    print(f"Epoch {e} lr {optimizer.param_groups[0]['lr']}")
    #train_iter.init_epoch()
    #print(" ".join([TEXT.vocab.itos[x] for x in next(iter(train_iter)).text[0][0].tolist()]))
    # Train
    train_loss, tntok = model.train_epoch(
        diter     = train_iter,
        clip      = args.clip,
        re        = args.re,
        optimizer = optimizer,
    )

    # Validate
    valid_loss, ntok = model.validate(valid_iter)
    # No schedule...dataset too small and gradients too noisy
    #schedule.step(valid_loss / ntok)

    # Accuracy on train
    train_acc = model.acc(full_train_iter)
    #train_acc = 0
    # Accuracy on Valid
    valid_acc = model.acc(valid_iter)

    #valid_f1 = model.f1(asp_valid_iter)
    valid_f1 = 0
    test_acc = model.acc(test_iter)

    # Report
    print(f"Epoch {e}")
    print(f"train loss: {train_loss / tntok} train acc: {train_acc}")
    print(f"valid loss: {valid_loss / ntok} valid acc: {valid_acc}")
    print(f"test acc: {test_acc}")

    if args.save and valid_loss < best_val:
        best_val = valid_loss
        savestring = f"{args.model}-lr{args.lr}-dp{args.dp}-tw{args.tieweights}-if{args.inputfeed}.pt"
        torch.save(model, savestring)
