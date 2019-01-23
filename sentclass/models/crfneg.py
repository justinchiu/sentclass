from .base import Sent
from .. import sentihood as data

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from pyro.ops.contract import ubersum

class CrfNeg(Sent):
    def __init__(
        self,
        V = None,
        L = None,
        A = None,
        S = None,
        emb_sz = 256,
        rnn_sz = 256,
        nlayers = 2,
        dp = 0.3,
        tieweights = True,
    ):
        super(CrfNeg, self).__init__()

        self._N = 0

        self.V = V
        self.L = L
        self.A = A
        self.S = S
        if L is None:
            L = [1]

        self.emb_sz = emb_sz
        self.rnn_sz = rnn_sz
        self.nlayers = nlayers
        self.dp = dp

        self.lut = nn.Embedding(
            num_embeddings = len(V),
            embedding_dim = emb_sz,
            padding_idx = V.stoi[self.PAD],
        )
        self.lut.weight.data.copy_(V.vectors)
        self.lut.weight.data[V.stoi[self.PAD]] = 0
        self.lut.weight.requires_grad = False
        self.lut_la = nn.Embedding(
            num_embeddings = len(L) * len(A),
            embedding_dim = nlayers * 2 * 2 * rnn_sz,
        )
        self.rnn = nn.LSTM(
            input_size    = emb_sz,
            hidden_size   = rnn_sz,
            num_layers    = nlayers,
            bias          = True,
            dropout       = dp,
            bidirectional = True,
            batch_first   = True,
        )
        self.drop = nn.Dropout(dp)

        # Score each sentiment for each location and aspect
        # Store the combined pos, neg, none in a single vector :(
        self.proj_s = nn.Parameter(torch.randn(len(L)*len(A), len(S), emb_sz))
        self.proj_s.data[:,:,self.S.stoi["none"]].mul_(2)
        self.proj_neg = nn.Parameter(torch.randn(len(L)*len(A), 2, 2*rnn_sz))
        self.psi_ys = nn.Parameter(torch.FloatTensor([0.1, 0.1, 0.1]))
        self.flip = nn.Parameter(
            torch.zeros(len(self.S), len(self.S))
        )
        self.flip.requires_grad = False
        self.flip.data[0,0] = 1
        self.flip.data[1,2] = 1
        self.flip.data[2,1] = 1

    def forward(self, x, lens, k, kx):
        words = x

        emb = self.drop(self.lut(x))
        p_emb = pack(emb, lens, True)

        l, a = k
        N = x.shape[0]
        T = x.shape[1]
        y_idx = l * len(self.A) + a if self.L is not None else a
        s = (self.lut_la(y_idx)
            .view(N, 2, 2 * self.nlayers, self.rnn_sz)
            .permute(1, 2, 0, 3)
            .contiguous())
        state = (s[0], s[1])
        x, (h, c) = self.rnn(p_emb, state)
        # h: L * D x N x H
        x = unpack(x, True)[0]
        proj_s = self.proj_s[y_idx.squeeze(-1)]
        phi_s = torch.einsum("nsh,nth->nts", [proj_s, emb])
        proj_neg = self.proj_neg[y_idx.squeeze(-1)]
        phi_neg = torch.einsum("nbh,nth->ntb", [proj_neg, x])

        phi_y = torch.zeros(N, len(self.S)).to(self.lut.weight.device)
        psi_ybs0 = torch.diag(self.psi_ys)
        psi_ybs1 = psi_ybs0 @ self.flip
        psi_ybs = (torch.stack([psi_ybs0, psi_ybs1], 1)
            .view(1, 1, len(self.S), 2, len(self.S))
            .repeat(N, T, 1, 1, 1))
        idxs = torch.arange(0, max(lens)).to(lens.device)
        # mask: N x R
        mask = (idxs.repeat(len(lens), 1) >= lens.unsqueeze(-1))
        phi_s.masked_fill_(mask.unsqueeze(-1), 0)
        psi_ybs.masked_fill_(mask.view(N, T, 1, 1, 1).expand_as(psi_ybs), 0)
        Z, hy = ubersum("nts,ntb,ntybs,ny->n,ny", phi_s, phi_neg, psi_ybs, phi_y, batch_dims="t", modulo_total=True)
        if self.training:
            self._N += 1
        if self._N > 1000 and self.training:
            Zt, hx, hb = ubersum("nts,ntb,ntybs,ny->nt,nts,ntb", phi_s, phi_neg, psi_ybs, phi_y, batch_dims="t", modulo_total=True)
            xp = (hx - Zt.unsqueeze(-1)).exp()
            bp = (hb - Zt.unsqueeze(-1)).exp()
            yp = (hy - Z.unsqueeze(-1)).exp()
            def stuff(i):
                loc = self.L.itos[l[i]]
                asp = self.A.itos[a[i]]
                return self.tostr(words[i]), loc, asp, xp[i], yp[i], bp[i]
            import pdb; pdb.set_trace()
            # wordsi, loc, asp, xpi, ypi, bpi = stuff(10)
        return hy

    def observe(self, x, lens, l, a, y):
        raise NotImplementedError
