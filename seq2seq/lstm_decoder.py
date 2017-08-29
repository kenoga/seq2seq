# -*- coding: utf-8 -*-

from chainer import Chain
from chainer import links as L
from chainer import functions as F

class LSTM_Decoder(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size):
        """
        :param vocab_size
        :param embed_size: the size of word embedding
        :param hidden_size: the size of hidden vector
        """
        super(LSTM_Decoder, self).__init__(
            # word embedding layer
            ye = L.EmbedID(vocab_size, embed_size, ignore_label=-1),
            # word vector to hidden vector (4x for lstm)
            eh = L.Linear(embed_size, 4 * hidden_size),
            # hidden vector to hidden vector (4x for lstm)
            hh = L.Linear(hidden_size, 4 * hidden_size),
            # hidden vector to word embedding vector
            he = L.Linear(hidden_size, embed_size),
            # word embedding vector to one-hot word vector
            ey = L.Linear(embed_size, vocab_size)
        )

    def __call__(self, y, c, h):
        """
        decode
        :param y: one-hot word vector
        :param c: internal memory in LSTM
        :param h: hidden vecotr
        :return: prediected word distribution, next internal memory, next hidden vector
        """
        # word embedding + non-linear function
        e = F.tanh(self.ye(y))
        # lstm
        # h1 = F.dropout(self.eh(e), ratio=0.8)
        # h2 = F.dropout(self.hh(h), ratio=0.8)
        # c, h = F.lstm(c, h1 + h2)
        c, h = F.lstm(c, self.eh(e) + self.hh(h))
        # caluculate probablity of each word
        t = self.ey(F.tanh(self.he(h)))
        return t, c, h