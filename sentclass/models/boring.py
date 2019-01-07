from .base import Sent
from .. import sentihood as data

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

class Boring(Sent):
    def __init__(
        self,
        V = None,
        Y_shape = None,
        emb_sz = 256,
        rnn_sz = 256,
        nlayers = 2,
        dp = 0.3,
        tieweights = True,
    ):
        super(Boring, self).__init__()

        assert(Y_shape is not None)

        self._N = 0

        self.V = V
        self.Y_shape = Y_shape
        self.emb_sz = emb_sz
        self.rnn_sz = rnn_sz
        self.nlayers = nlayers
        self.dp = dp

        self.lut = nn.Embedding(
            num_embeddings = len(V),
            embedding_dim = emb_sz,
            padding_idx = V.stoi[self.PAD],
        )
        self.rnn = nn.LSTM(
            input_size = emb_sz,
            hidden_size = rnn_sz,
            num_layers = nlayers,
            bias = False,
            dropout = dp,
            bidirectional = True,
            batch_first   = True,
        )
        self.drop = nn.Dropout(dp)

        # Score each sentiment for each location and aspect
        self.proj = nn.Linear(
            in_features = 2*rnn_sz,
            out_features = Y_shape[0] * Y_shape[1] * Y_shape[2],
            bias = False,
        )

        # Tie weights???
        #if tieweights:
            #self.proj.weight = self.lut.weight


    def forward(self, x, lens, k):
        emb = self.lut(x)
        p_emb = pack(emb, lens, True)
        x, (h, c) = self.rnn(p_emb)
        # h: L * D x N x H
        x = unpack(x, True)[0]
        # y: N x D * H
        y = (h
            .view(self.nlayers, 2, -1, self.rnn_sz)[-1]
            .permute(1, 0, 2)
            .contiguous()
            .view(-1, 2 * self.rnn_sz))
        Ys = self.Y_shape
        return self.proj(self.drop(y)).view(-1, Ys[0], Ys[1], Ys[2])
        #return self.proj(self.drop(unpack(x)[0])), s


    def init_state(self, N):
        if self._N != N:
            self._N = N
            self._state = (
                torch.zeros(self.nlayers, N, self.rnn_sz).to(self.lut.weight.device),
                torch.zeros(self.nlayers, N, self.rnn_sz).to(self.lut.weight.device),
            )
        return self._state

if __name__ == "__main__":
    print("HI")
