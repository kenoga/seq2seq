# -*- encoding: utf-8 -*- 

import json

def load_dataset(path):
    with open(path, "r") as f:
        return json.load(f)

def load_vocab(path):
    with open(path, "r") as f:
        return json.load(f)

def extend_lists(lists):
    result = []
    for l in lists:
        result += l
    return result
    
def make_dataset(dials_data):
    '''
    対話データセットからcontextとsystem utteranceのペアを作成する
    '''
    enc_data = []
    dec_data = []
    for dial in dials_data:
        turns = dial["dial"]
        # u1 -> s1 -> u2 -> s2
        # 上のような対話から
        # (u1, s1), (u1+s1+u2, s2)のようなサブ対話データを作成する
        sub_dials = [ \
            (extend_lists([pre_turn["ids"] for pre_turn in turns[0:i]]), turn["ids"]) \
            for i, turn in enumerate(turns) if i%2==1]
        enc_data.extend([sub_dial[0] for sub_dial in sub_dials])
        dec_data.extend([sub_dial[1] for sub_dial in sub_dials])
    return enc_data, dec_data

def get_list_by_idxs(l, idxs):
    '''
    複数のindex idsで指定されたl内の要素を返す
    '''
    return [l[i] for i in idxs]

def uniform_batch_length(batch, ARR):
    '''
    batch内のlistの長さを揃える
    '''
    max_len = max([len(x) for x in batch])
    for i, x in enumerate(batch):
        batch[i] = x + [-1] * (max_len - len(x))
    return batch
    
def make_minibatch(x_list, y_list, batch_size, vocab, ARR, random=True):
    '''
    学習データのミニバッチを作成する関数
    ランダムでミニバッチに分割して，ミニバッチ内のデータのサイズをミニバッチ内の最大長に合わせる
    param x_list: list of list of int (word_id)
    param y_list: list of list of int (word_id)
    param batch_size: int
    return x_batches, y_batches
    '''
    # シャッフルしてからバッチサイズごとに長さを揃える
    # dataはidのリストのリスト
    assert len(x_list) == len(y_list)
    N = len(x_list)
    
    if random:
        perm = ARR.random.permutation(N)
    x_batches = []
    y_batches = []
    
    # targetにeosを追加する
    for y in y_list:
        y.append(vocab["<eos>"])
    
    for i in range(0, N, batch_size):
        if random:
            x_batch = get_list_by_idxs(x_list, perm[i:i+batch_size])
            y_batch = get_list_by_idxs(y_list, perm[i:i+batch_size])
        else:
            x_batch = x_list[i:i+batch_size]
            y_batch = y_list[i:i+batch_size]
        uniformed_x_batch = uniform_batch_length(x_batch, ARR)
        uniformed_y_batch = uniform_batch_length(y_batch, ARR)
        x_batches.append(ARR.array(uniformed_x_batch, dtype=ARR.int32).T)
        y_batches.append(ARR.array(uniformed_y_batch, dtype=ARR.int32).T)
    
    return x_batches, y_batches