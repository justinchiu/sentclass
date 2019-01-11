from .ugm import Ugm
from .. import sentihood as data

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from pyro.ops.contract import ubersum

class CrfSimple(Ugm):
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
        super(CrfSimple, self).__init__()

        self._N = 0

        self.V = V
        self.L = L
        self.A = A
        self.S = S

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
        self.lut.weight.requires_grad = False

        self.rnn = nn.LSTM(
            input_size = emb_sz,
            hidden_size = rnn_sz,
            num_layers = nlayers,
            bias = True,
            dropout = dp,
            bidirectional = True,
            batch_first   = True,
        )
        self.drop = nn.Dropout(dp)

        # Score each sentiment for each location and aspect
        # Store the combined pos, neg, none in a single vector :(
        self.proj_s = nn.Linear(
            in_features = 2 * rnn_sz,
            out_features = len(S) * len(L),
            bias = True,
        )
        self.proj_l = nn.Linear(
            in_features = 2 * rnn_sz,
            out_features = len(L) * 2,
            bias = True,
        )
        self.proj_a = nn.Linear(
            in_features = 2 * rnn_sz,
            out_features = len(A),
            bias = True,
        )
        self.theta = nn.Parameter(torch.Tensor([1.]))
        self.psi_ys = nn.Parameter(
            torch.randn(len(S), len(S)) + torch.eye(len(S))
        )

    def forward(self, x, lens, k, kx, y=None):
        N, T = x.shape
        L, A, S = self.L, self.A, self.S
        emb = self.drop(self.lut(x))
        p_emb = pack(emb, lens, True)

        l, a = k
        x, _ = self.rnn(p_emb)
        x = unpack(x, True)[0]
        psi_ax = self.proj_a(x).view(N, T, len(A))
        psi_sx = self.proj_s(x).view(N, T, len(L), len(S))
        #psi_lx = self.proj_l(x).view(N, T, len(L), 2)
        #psi_yas =
        psi_ysa = self.psi_ysa
        import pdb; pdb.set_trace()

        return self.proj(h)

    def _old_forward(self, x, lens, k):
        emb = self.drop(self.lut(x))
        p_emb = pack(emb, lens, True)
        x, (h, c) = self.rnn(p_emb)
        # h: L * D x N x H
        x = unpack(x, True)[0]
        # y: N x D * H
        #h = h+c
        y = (h
            .view(self.nlayers, 2, -1, self.rnn_sz)[-1]
            .permute(1, 0, 2)
            .contiguous()
            .view(-1, 2 * self.rnn_sz))
        Ys = self.Y_shape
        return self.proj(y).view(-1, Ys[0], Ys[1], Ys[2])
