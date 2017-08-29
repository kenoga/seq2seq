# -*- coding: utf-8 -*- 

from chainer import Chain
from chainer import links as L
from chainer import functions as F

class LSTM_Encoder(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size):
        """
        :param vocab_size
        :param embed_size: the size of word embedding
        :param hidden_size
        """
        super(LSTM_Encoder, self).__init__(
            # word embedding layer
            xe = L.EmbedID(vocab_size, embed_size, ignore_label=-1),
            # word vector to hidden vector (4x for lstm)
            eh = L.Linear(embed_size, 4 * hidden_size),
            # hidden vector to next hidden vector (4x for lstm)
            hh = L.Linear(hidden_size, 4 * hidden_size)
        )

    def __call__(self, x, c, h):
        """
        encode
        :param x: one-hot word vector
        :param c: internal memory
        :param h: hidden vector
        :return: next internal memory, next hidden vector
        """
        # word embedding + non-linear function
        e = F.tanh(self.xe(x))
        # lstm
        h1 = F.dropout(self.eh(e), ratio=0.8)
        h2 = F.dropout(self.hh(h), ratio=0.8)
        return F.lstm(c, h1 + h2)
        # return F.lstm(c, self.eh(e) + self.hh(h))