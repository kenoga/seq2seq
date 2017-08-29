# -*- encoding: utf-8 -*-

from chainer import Chain, Variable, functions
from lstm_encoder import LSTM_Encoder
from lstm_decoder import LSTM_Decoder

class Seq2Seq(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size, flag_gpu=False):
        """
        Initialize model
        :param vocab_size
        :param embed_size
        :param hidden_size
        :param flag_gpu: whether use gpu or not
        """
        super(Seq2Seq, self).__init__(
            encoder = LSTM_Encoder(vocab_size, embed_size, hidden_size),
            decoder = LSTM_Decoder(vocab_size, embed_size, hidden_size)
        )
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        if flag_gpu:
            from chainer import cuda
            self.ARR = cuda.cupy
        else:
            import numpy as np
            self.ARR = np

    def encode(self, words, batch_size):
        """
        This method encodes sentences to a fixed size vector.
        :param words: Variable object, sentence length * batch size
        :return:
        """
        # initialize c and h
        c = Variable(self.ARR.zeros((batch_size, self.hidden_size), dtype='float32'))
        h = Variable(self.ARR.zeros((batch_size, self.hidden_size), dtype='float32'))

        # encode each word
        for w in words:
            c, h = self.encoder(w, c, h)

        # memorize hidden vector
        self.h = h
        # initialize internal memory in LSTM
        self.c = Variable(self.ARR.zeros((batch_size, self.hidden_size), dtype='float32'))

    def decode(self, w):
        """
        This method decode vectors to sentences.
        :param w: word
        :return: vector (vocab size), likelihood of each word
        """
        t, self.c, self.h = self.decoder(w, self.c, self.h)
        return t

    def reset(self, batch_size):
        """
        Initializes h, c, and gradients.
        :return:
        """
        self.h = Variable(self.ARR.zeros((batch_size, self.hidden_size), dtype='float32'))
        self.c = Variable(self.ARR.zeros((batch_size, self.hidden_size), dtype='float32'))

        self.zerograds()


    def forward(self, enc_words, dec_words, vocab):
        """
        Caluculate loss.
        :param enc_words: a minibatch of source sentences
        :param dec_words: a minibatch of tartget sentences
        :param vocab
        :param model
        :param ARR: cuda.cupy or numpy
        :return: total loss
        """
        # batch size
        batch_size = len(enc_words[0])
        # initialize contextsand gradients
        self.reset(batch_size)
        # ndarray to Variable
        enc_words = [Variable(row) for row in enc_words]
        # sentence encoding (1)
        self.encode(enc_words, batch_size)
        # initialize loss
        loss = Variable(self.ARR.zeros((), dtype='float32'))
        # make decoder read <eos> (2)
        t = Variable(self.ARR.array([vocab["<bos>"] for _ in range(batch_size)], dtype="int32"))
        
        for w in dec_words:
            # decoding (3)
            y = self.decode(t)
            # transform the type of a target word, ndarray to Variable
            t = Variable(self.ARR.array(w, dtype='int32'))
            # caluculate loss (4)
            loss += functions.softmax_cross_entropy(y, t)
        return loss
    
    def reverse_dict(self, dict):
        return {v:k for k, v in dict.items()}
        
    def get_sentence(self, enc_words, vocab):
        r_vocab = self.reverse_dict(vocab)
        batch_size = len(enc_words[0])
        self.reset(batch_size=batch_size)
        enc_words = [Variable(row) for row in enc_words]
        self.encode(enc_words, batch_size=batch_size)
        t = Variable(self.ARR.array([vocab["<bos>"] for _ in range(batch_size)], dtype="int32"))
        
        s = []
        while True:
            vector = self.decode(t).data[0]
            sorted_indexes = np.argsort(vector)[::-1]
            print [r_vocab[i] for i in sorted_indexes[:5]]
            word_i = np.argmax(vector)
            t = Variable(np.array([[word_i]], dtype=np.int32))
            word = r_vocab[word_i]
            s.append(word)
            if word == "<eos>":
                break
        return s
    


        
