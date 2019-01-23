from .base import Sent
from .. import sentihood as data

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from pyro.ops.contract import ubersum

class CrfLstmDiag(Sent):
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
    ):
        super(CrfLstmDiag, self).__init__()

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
        self.lut.weight.requires_grad = False
        self.lut_la = nn.Embedding(
            num_embeddings = len(L) * len(A),
            embedding_dim = nlayers * 2 * 2 * rnn_sz,
        )
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
            #out_features = len(S),# + 1,
            out_features = len(S) + 1,
            bias = False,
        )
        #self.psi_ys = nn.Parameter(
            #torch.randn(len(S), len(S)) + torch.eye(len(S))
        #)
        #self.theta = nn.Parameter(torch.FloatTensor([10]))
        self.psi_ys = nn.Parameter(torch.FloatTensor([0.1, 0.1, 0.1]))
        #self.psi_ys.requires_grad = False
        self.proj_y = nn.Linear(
            in_features = 2 * rnn_sz,
            out_features = len(S),
            bias = False,
        )


    def forward(self, x, lens, k, kx):
        # model takes as input the text, aspect, and location
        # runs BLSTM over text using embedding(location, aspect) as
        # the initial hidden state, as opposed to a different lstm for every pair???
        # output sentiment

        # DBG
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
        phi_s = self.proj_s(x)
        idxs = torch.arange(0, max(lens)).to(lens.device)
        # mask: N x R x 1
        mask = (idxs.repeat(len(lens), 1) >= lens.unsqueeze(-1))
        phi_s[:,:,-1].masked_fill_(1-mask, float("-inf"))
        phi_s[:,:,:3].masked_fill_(mask.unsqueeze(-1), float("-inf"))

        phi_y = torch.zeros(N, len(self.S)).to(self.psi_ys.device)
        psi_ys = torch.cat(
            [torch.diag(self.psi_ys), torch.zeros(len(self.S), 1).to(self.psi_ys)],
            dim=-1,
        ).expand(T, len(self.S), len(self.S)+1)
        Z, hy = ubersum("nts,tys,ny->n,ny", phi_s, psi_ys, phi_y, batch_dims="t", modulo_total=True)

        return hy


    def observe(self, x, lens, l, a, y):
        emb = self.drop(self.lut(x))
        p_emb = pack(emb, lens, True)

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
        # Get the last hidden states for both directions, POSSIBLE BUGS
        phi_s = self.proj_s(x)
        idxs = torch.arange(0, max(lens)).to(lens.device)
        # mask: N x R x 1
        mask = (idxs.repeat(len(lens), 1) >= lens.unsqueeze(-1))
        phi_s[:,:,-1].masked_fill_(1-mask, float("-inf"))
        phi_s[:,:,:3].masked_fill_(mask.unsqueeze(-1), float("-inf"))
        phi_y = torch.zeros(N, len(self.S)).to(self.psi_ys.device)
        psi_ys = torch.cat(
            [torch.diag(self.psi_ys), torch.zeros(len(self.S), 1).to(self.psi_ys)],
            dim=-1,
        )
        psi_ys0 = psi_ys[y]
        return phi_s + psi_ys0
