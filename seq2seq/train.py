# -*- encoding: utf-8 -*- 

import os, sys
import time
import json
import argparse
import chainer
from chainer import optimizers, links, optimizer, serializers
from seq2seq import Seq2Seq
import utils

DATASET_PATH = "../dataset/dataset.json"
VOCAB_PATH = "../dataset/vocab.json"
MODEL_SAVE_PATH = "./model/seq2seq_epoch%02d.model"
EMBED_SIZE = 200
HIDDEN_SIZE = 200
BATCH_SIZE = 128
EPOCH_NUM = 20


def train(args):
    vocab = utils.load_vocab(VOCAB_PATH)
    vocab_size = len(vocab)
    
    model = Seq2Seq(vocab_size=vocab_size,
                    embed_size=EMBED_SIZE,
                    hidden_size=HIDDEN_SIZE,
                    flag_gpu=args.use_gpu)
    
    # set gpu or not
    if args.use_gpu:
        from chainer import cuda
        ARR = cuda.cupy
        cuda.get_device(args.gpu_id).use()
        model.to_gpu(args.gpu_id)
    else:
        import numpy as np
        ARR = np
        
    # load dataset
    dataset = utils.load_dataset(DATASET_PATH)
    train_dataset = dataset["train"]
    dev_dataset = dataset["dev"]
    train_x_list, train_y_list = utils.make_dataset(train_dataset)
    dev_x_list, dev_y_list = utils.make_dataset(dev_dataset)
    dev_x_batches, dev_y_batches = utils.make_minibatch(dev_x_list, dev_y_list, BATCH_SIZE, vocab, use_gpu=args.use_gpu, gpu_id=args.gpu_id, random=False)
    dev_batch_num = len(dev_x_batches)
    

    # start training
    start_time = time.time()
    for epoch_i in range(EPOCH_NUM):
        # initializing a optimizer in each epoch
        opt = optimizers.Adam()
        opt.setup(model)
        opt.add_hook(optimizer.GradientClipping(5))
        
        train_x_batches, train_y_batches = utils.make_minibatch(train_x_list, train_y_list, BATCH_SIZE, vocab, use_gpu=args.use_gpu, gpu_id=args.gpu_id, random=True)
        
        
        train_batch_num = len(train_x_batches)
        
        train_loss_sum = 0
        for batch_i in range(train_batch_num):
            # caluculating loss
            train_loss = model.forward(enc_words=train_x_batches[batch_i],
                                dec_words=train_y_batches[batch_i],
                                vocab=vocab)
            # back propagation
            train_loss.backward()
            
            train_loss_sum += train_loss.data
            print("loss: %4.1f, epoch: %2d/%2d, batch: %2d/%2d, time: %0.1f" % (train_loss.data, epoch_i, EPOCH_NUM, batch_i, train_batch_num, (time.time()-start_time)))
            
            # update the network
            opt.update()

        
        dev_loss_sum = 0
        for batch_i in range(dev_batch_num):
            dev_loss = model.forward(enc_words=dev_x_batches[batch_i],
                               dec_words=dev_y_batches[batch_i],
                               vocab=vocab
                               )
            dev_loss_sum = dev_loss.data
        
        print("*"*50)
        print("EPOCH: %2d/%2d, TRAIN LOSS: %0.1f, DEV LOSS: %0.1f" % (epoch_i+1, EPOCH_NUM, train_loss_sum, dev_loss_sum))
        serializers.save_hdf5(MODEL_SAVE_PATH % epoch_i, model)
        

parser = argparse.ArgumentParser()
parser.add_argument("--use-gpu", action="store_true")
parser.add_argument("--gpu-id", type=int, default=0)
parser.add_argument("--batch", type=int, default=128)
parser.add_argument("--epoch", type=int, default=20)
parser.add_argument("--emb-size", type=int, default=256)
parser.add_argument("--hid-size", type=int, default=256)
args = parser.parse_args()
train(args)