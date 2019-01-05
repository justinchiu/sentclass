from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torch.nn.utils import clip_grad_norm_ as clip_


class Lm(nn.Module):
    PAD = "<pad>"

    def _loop(self, iter, optimizer=None, clip=0, learn=False, re=None):
        context = torch.enable_grad if learn else torch.no_grad

        cum_loss = 0
        cum_ntokens = 0
        cum_rx = 0
        cum_kl = 0
        batch_loss = 0
        batch_ntokens = 0
        states = None
        with context():
            titer = tqdm(iter) if learn else iter
            for i, batch in enumerate(titer):
                if learn:
                    optimizer.zero_grad()
                text, lens = batch.text
                x = text[:-1]
                y = text[1:]
                lens = lens - 1

                e, lene = batch.entities
                t, lent = batch.types
                v, lenv = batch.values
                #rlen, N = e.shape
                #r = torch.stack([e, t, v], dim=-1)
                r = [e, t, v]
                assert (lene == lent).all()
                lenr = lene

                # should i include <eos> in ppl?
                nwords = y.ne(1).sum()
                # assert nwords == lens.sum()
                T, N = y.shape
                #if states is None:
                states = self.init_state(N)
                logits, _ = self(x, states, lens, r, lenr)

                nll = self.loss(logits, y)
                kl = 0
                nelbo = nll + kl
                if learn:
                    nelbo.div(nwords.item()).backward()
                    if clip > 0:
                        gnorm = clip_(self.parameters(), clip)
                        #for param in self.rnn_parameters():
                            #gnorm = clip_(param, clip)
                    optimizer.step()
                cum_loss += nelbo.item()
                cum_ntokens += nwords.item()
                batch_loss += nelbo.item()
                batch_ntokens += nwords.item()
                if re is not None and i % re == -1 % re:
                    titer.set_postfix(loss = batch_loss / batch_ntokens, gnorm = gnorm)
                    batch_loss = 0
                    batch_ntokens = 0
        return cum_loss, cum_ntokens

    def train_epoch(self, iter, optimizer, clip=0, re=None):
        return self._loop(iter=iter, learn=True, optimizer=optimizer, clip=clip, re=re)

    def validate(self, iter):
        return self._loop(iter=iter, learn=False)

    def forward(self):
        raise NotImplementedError

    def rnn_parameters(self):
        raise NotImplementedError

    def init_state(self):
        raise NotImplementedError

    def loss(self, logits, y):
        T, N = y.shape
        yflat = y.view(-1, 1)
        return -(F.log_softmax(logits, dim=-1)
            .view(T*N, -1)
            .gather(-1, yflat)[yflat != 1]
            .sum())


class Lvm(Lm):
    def _loop(self, iter, optimizer=None, clip=0, learn=False, re=None):
        context = torch.enable_grad if learn else torch.no_grad

        cum_loss = 0
        cum_ntokens = 0
        cum_rx = 0
        cum_kl = 0
        batch_loss = 0
        batch_ntokens = 0
        states = None
        with context():
            titer = tqdm(iter) if learn else iter
            for i, batch in enumerate(titer):
                if learn:
                    optimizer.zero_grad()
                text, lens = batch.text
                x = text[:-1]
                y = text[1:]
                lens = lens - 1

                e, lene = batch.entities
                t, lent = batch.types
                v, lenv = batch.values
                #rlen, N = e.shape
                #r = torch.stack([e, t, v], dim=-1)
                r = [e, t, v]
                assert (lene == lent).all()
                lenr = lene

                # should i include <eos> in ppl?
                nwords = y.ne(1).sum()
                # assert nwords == lens.sum()
                T, N = y.shape
                #if states is None:
                states = self.init_state(N)
                logits, _ = self(x, states, lens, r, lenr)
                """
                if learn:
                    states = (
                        [tuple(x.detach() for x in tup) for tup in states[0]],
                        tuple(x.detach() for x in states[1]),
                        states[2].detach(),
                    )
                """
                nll = self.loss(logits, y)
                kl = 0
                nelbo = nll + kl
                if learn:
                    nelbo.div(nwords.item()).backward()
                    if clip > 0:
                        gnorm = clip_(self.parameters(), clip)
                        #for param in self.rnn_parameters():
                            #gnorm = clip_(param, clip)
                    optimizer.step()
                cum_loss += nelbo.item()
                cum_ntokens += nwords.item()
                batch_loss += nelbo.item()
                batch_ntokens += nwords.item()
                if re is not None and i % re == -1 % re:
                    titer.set_postfix(loss = batch_loss / batch_ntokens, gnorm = gnorm)
                    batch_loss = 0
                    batch_ntokens = 0
        return cum_loss, cum_ntokens


    def loss(self, logits, y):
        T, N = y.shape
        yflat = y.view(-1, 1)
        return -(F.log_softmax(logits, dim=-1)
            .view(T*N, -1)
            .gather(-1, yflat)[yflat != 1]
            .sum())


class Ie(nn.Module):
    PAD = "<pad>"

    def _loop_ie(self, iter, optimizer=None, clip=0, learn=False, re=None):
        context = torch.enable_grad if learn else torch.no_grad

        cum_loss = 0
        cum_ntokens = 0
        cum_rx = 0
        cum_kl = 0
        batch_loss = 0
        batch_ntokens = 0
        states = None
        with context():
            t = tqdm(iter) if learn else iter
            for i, batch in enumerate(t):
                if learn:
                    optimizer.zero_grad()
                text, lens = batch.text
                x = text[:-1]
                y = text[1:]
                lens = lens - 1
                # should i include <eos> in ppl?
                nwords = y.ne(1).sum()
                # assert nwords == lens.sum()
                T, N = y.shape
                #if states is None:
                states = self.init_state(N)
                logits, _ = self(x, states, lens)
                #logits, states = self(x, states, lens)
                logprobs = F.log_softmax(logits, dim=-1)
                logp = logprobs.view(T*N, -1).gather(-1, y.view(T*N, 1))
                kl = 0
                nll = -logp[y.view(-1, 1) != 1].sum()
                nelbo = nll + kl
                if learn:
                    nelbo.div(nwords.item()).backward()
                    if clip > 0:
                        gnorm = clip_(self.parameters(), clip)
                        #for param in self.rnn_parameters():
                            #gnorm = clip_(param, clip)
                    optimizer.step()
                cum_loss += nelbo.item()
                cum_ntokens += nwords.item()
                batch_loss += nelbo.item()
                batch_ntokens += nwords.item()
                if re is not None and i % re == -1 % re:
                    t.set_postfix(loss = batch_loss / batch_ntokens, gnorm = gnorm)
                    batch_loss = 0
                    batch_ntokens = 0
        return cum_loss, cum_ntokens

