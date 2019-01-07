# rnnlm main.py --devid 0 --tieweights => 2.7
# crnnlm main.py --devid 0 --model crnnlm --tieweights --lr 0.003 => 2.337
 
import argparse

import torch
import torch.optim as optim

from torchtext.data import BucketIterator

import sentclass.sentihood as data
from sentclass.models.boring import Boring

import json

#torch.set_anomaly_enabled(True)
#torch.backends.cudnn.enabled = False
torch.backends.cudnn.enabled = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filepath",
        default="data",
        type=str,
    )

    parser.add_argument("--devid", default=-1, type=int)

    parser.add_argument("--bsz", default=32, type=int)
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
        choices=["boring", "crf"],
        default="boring"
    )

    parser.add_argument("--nlayers", default=2, type=int)
    parser.add_argument("--emb-sz", default=256, type=int)
    parser.add_argument("--rnn-sz", default=256, type=int)

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
TEXT, SENTIMENT = data.make_fields()
train, valid, test = data.SentihoodDataset.splits(
    TEXT, SENTIMENT, path=args.filepath)

data.build_vocab(TEXT, SENTIMENT, train)

train_iter, valid_iter, test_iter = BucketIterator.splits(
    (train, valid, test),
    batch_size = args.bsz,
    device = device,
    repeat = False,
    sort_within_batch = True,
    #sort_key = already given in dataset?
)

# Model
if args.model == "boring":
    model = Boring(
        V       = TEXT.vocab,
        Y_shape = data.Y_shape,
        emb_sz  = args.emb_sz,
        rnn_sz  = args.rnn_sz,
        nlayers = args.nlayers,
        dp      = args.dp,
        tieweights = args.tieweights,
    )
elif args.model == "crf":
    model = Crf(
        Ve = ENT.vocab,
        Vt = TYPE.vocab,
        Vv = VALUE.vocab,
        Vx = TEXT.vocab,
        r_emb_sz = args.emb_sz,
        x_emb_sz = args.emb_sz,
        rnn_sz = args.rnn_sz,
        nlayers = args.nlayers,
        dropout = args.dp,
        tieweights = args.tieweights,
        inputfeed = args.inputfeed,
    )

model.to(device)
print(model)

params = list(model.parameters())

optimizer = optim.Adam(
    params, lr = args.lr, weight_decay = args.wd, betas=(args.b1, args.b2))
schedule = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=args.pat, factor=args.lrd, threshold=1e-3)
#batch = next(iter(train_iter))
# TODO: try truncating sequences early on?

best_val = float("inf")
for e in range(args.epochs):
    print(f"Epoch {e} lr {optimizer.param_groups[0]['lr']}")
    # Train
    train_loss, tntok = model.train_epoch(
        iter      = train_iter,
        clip      = args.clip,
        re        = args.re,
        optimizer = optimizer,
    )

    # Validate
    valid_loss, ntok = model.validate(valid_iter)
    print(f"Epoch {e} train loss: {train_loss / tntok} valid loss: {valid_loss / ntok}")
    schedule.step(valid_loss / ntok)

    if args.save and valid_loss < best_val:
        best_val = valid_loss
        savestring = f"{args.model}-lr{args.lr}-dp{args.dp}-tw{args.tieweights}-if{args.inputfeed}.pt"
        torch.save(model, savestring)
