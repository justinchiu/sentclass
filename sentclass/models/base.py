from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torch.nn.utils import clip_grad_norm_ as clip_


class Sent(nn.Module):
    PAD = "<pad>"

    def _loop(self, diter, optimizer=None, clip=0, learn=False, re=None, once=False):
        context = torch.enable_grad if learn else torch.no_grad

        cum_loss = 0
        cum_ntokens = 0
        cum_rx = 0
        cum_kl = 0
        batch_loss = 0
        batch_ntokens = 0
        with context():
            titer = tqdm(diter) if learn else diter
            for i, batch in enumerate(titer if not once else [next(iter(diter))]):
                if learn:
                    optimizer.zero_grad()
                x, lens = batch.text

                lx, _ = batch.locations_text
                ax, _ = batch.aspects_text
                l = batch.locations
                a = batch.aspects
                y = batch.sentiments

                # keys 
                k = [l, a]
                kx = [lx, ax]

                # N x l x a x y, now N x y for dealing w imbalance
                logits = self(x, lens, k, kx)

                nll = self.loss(logits, y)
                #import pdb; pdb.set_trace()
                nelbo = nll
                N = y.shape[0]
                if learn:
                    nelbo.div(N).backward()
                    if clip > 0:
                        gnorm = clip_(self.parameters(), clip)
                        #for param in self.rnn_parameters():
                            #gnorm = clip_(param, clip)
                    optimizer.step()
                cum_loss += nelbo.item()
                cum_ntokens += N
                batch_loss += nelbo.item()
                batch_ntokens += N
                if re is not None and i % re == -1 % re:
                    titer.set_postfix(loss = batch_loss / batch_ntokens, gnorm = gnorm)
                    batch_loss = 0
                    batch_ntokens = 0
        #print(f"train n: {cum_ntokens}")
        return cum_loss, cum_ntokens

    def train_epoch(self, diter, optimizer, clip=0, re=None, once=False):
        self.train()
        return self._loop(diter=diter, learn=True, optimizer=optimizer, clip=clip, re=re, once=once)

    def validate(self, diter):
        self.eval()
        return self._loop(diter=diter, learn=False)

    def forward(self):
        raise NotImplementedError

    def rnn_parameters(self):
        raise NotImplementedError

    def init_state(self):
        raise NotImplementedError

    def loss(self, logits, y):
        #N = y.shape
        yflat = y.view(-1, 1)
        return -(F.log_softmax(logits, dim=-1)
            #.view(-1, logits.shape[-1])
            .gather(-1, yflat)#[yflat != 1]
            .sum())

    def acc(self, iter, skip0=False):
        self.eval()
        correct = 0.
        total = 0.
        ftotal = 0.
        self._N += 1
        with torch.no_grad():
            for i, batch in enumerate(iter):
                x, lens = batch.text

                l = batch.locations
                a = batch.aspects
                lx = batch.locations_text
                ax = batch.aspects_text
                y = batch.sentiments
                N = y.shape[0]
                # Aspects: 7, 8, 12, 16, or first 4

                # keys 
                k = [l, a]
                kx = [lx, ax]

                # N x l x a x y
                logits = self(x, lens, k, kx)
                _, hy = logits.view(N, -1, len(self.S)).max(-1)
                #y = y.view(N, 2, -1)

                #hy = hy[:,:,:4]
                #y = y[:,:,:4]
                if skip0:
                    correct += (hy[y != 0] == y[y!=0]).sum().item()
                    total += y[y!=0].nelement()
                else:
                    correct += (hy == y).sum().item()
                    total += y.nelement()
                ftotal += y.nelement()
                #import pdb; pdb.set_trace()
                #if self._N > 10:
                    #import pdb; pdb.set_trace()
        print(f"acc total y!=0: {total}")
        #print(f"acc total: {ftotal}")
        return correct / total

    def f1(self, iter):
        """ Compute aspect F1
            Only checks if aspects are nonzero.
            Cannot take in a flattened iterator.
        """
        self.eval()
        p = 0.
        r = 0.
        Nf = 0
        with torch.no_grad():
            for i, batch in enumerate(iter):
                x, lens = batch.text

                l = batch.locations
                a = batch.aspects
                lx, _ = batch.locations_text
                ax, _ = batch.aspects_text
                y = batch.sentiments
                N = y.shape[0]
                # Aspects: 7, 8, 12, 16, or first 4

                hys = []
                for j in range(l.shape[-1]):
                    # keys 
                    k = [l[:,j], a[:,j]]
                    kx = [lx[:,j], ax[:,j]]

                    # N x l x a x y
                    logits = self(x, lens, k, kx)
                    _, hy = logits.view(N, -1, 3).max(-1)
                    hys.append(hy)
                hy = torch.cat(hys, dim=-1)

                # reshape into targets
                hy0 = hy.view(N*2, len(self.A))
                y0 = y.view(N*2, len(self.A))
                hy = hy.view(N*2, len(self.A)).ne(0)
                y = y.view(N*2, len(self.A)).ne(0)

                mask = y.sum(-1).ne(0)
                hy0 = hy0[mask]
                y0 = y0[mask]
                hy = hy[mask]
                y = y[mask]

                pi = (hy * y).sum(-1).float() / hy.sum(-1).float()
                ri = (hy * y).sum(-1).float() / y.sum(-1).float()
                Ni = y.shape[0]
                # calculate intersection - # of overlapping 0s
                intersection = (hy0 == y0).sum(-1)
                intersection0 = (hy0.eq(0) * y0.eq(0)).sum(-1)
                correct = (intersection - intersection0).float()
                pi0 = correct / hy.sum(-1).float()
                ri0 = correct / y.sum(-1).float()

                pi[pi != pi] = 0
                pi0[pi0 != pi0] = 0
                if (pi != pi).any():
                    import pdb; pdb.set_trace()
                if (ri != ri).any():
                    import pdb; pdb.set_trace()

                #p += pi.sum().item()
                #r += ri.sum().item()
                p += pi0.sum().item()
                r += ri0.sum().item()
                Nf += Ni
        P = p / Nf
        R = r / Nf
        if P == 0 and R == 0:
            import pdb; pdb.set_trace()
        F1 = 2 * P * R / (P + R)
        return F1

    def tostr(self, x):
        return [self.V.itos[w] for w in x.tolist()] 
