# -*- encoding: utf-8 -*- 

import os, sys
import time
import json
import numpy as np
import chainer
from chainer import optimizers, links, optimizer, serializers
from seq2seq import Seq2Seq
import utils

DATASET_PATH = "../dataset/dataset.json"
VOCAB_PATH = "../dataset/vocab.json"
MODEL_SAVE_PATH = "./model/seq2seq_epoch%02d.model"
EMBED_SIZE = 200
HIDDEN_SIZE = 200
FLAG_GPU = False
EPOCH_NUM = 20

BATCH_SIZE = 1

dataset = utils.load_dataset(DATASET_PATH)
vocab = utils.load_vocab(VOCAB_PATH)
vocab_size = len(vocab)
test_dataset = dataset["test"]
test_x_list, test_y_list = utils.make_dataset(test_dataset)
test_x_batches, test_y_batches = utils.make_minibatch(test_x_list, test_y_list, BATCH_SIZE, vocab, random=False)
test_batch_num = len(test_x_batches)

train_dataset = dataset["train"]
dev_dataset = dataset["dev"]
train_x_list, train_y_list = utils.make_dataset(train_dataset)
dev_x_list, dev_y_list = utils.make_dataset(dev_dataset)
dev_x_batches, dev_y_batches = utils.make_minibatch(dev_x_list, dev_x_list, BATCH_SIZE, vocab, random=False)
dev_batch_num = len(dev_x_batches)
train_x_batches, train_y_batches = utils.make_minibatch(train_x_list[:100], train_y_list[:100], BATCH_SIZE, vocab, random=False)

# train_x_list = 


# TODO
# beam search
# dropout, mini_batch作成方法
# learning rate
# initialization
# embed size, hidden size
model = Seq2Seq(vocab_size=vocab_size,
                embed_size=EMBED_SIZE,
                hidden_size=HIDDEN_SIZE,
                flag_gpu=FLAG_GPU)
serializers.load_hdf5("./model/seq2seq_epoch05.model", model)

r_vocab = {v:k for k, v in vocab.items()}

for i, batch in enumerate(test_x_batches):
    print "*"*50
    test = model.get_sentence(batch, vocab)
    print "context:" + " ".join([r_vocab[word_id] for word_id in train_x_list[i]])
    print "ans    :" + " ".join([r_vocab[word_id] for word_id in train_y_list[i]])
    print "result :" + " ".join(test)

