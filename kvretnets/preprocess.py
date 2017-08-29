# -*- encoding: utf-8 -*-


import os
import re
import json
import config


def separate_symbol(sent, symbol):
    assert len(symbol) == 1
    # symbolの前には必ず空白がくるようにする
    result = sent.replace(" "+symbol, symbol)
    result = result.replace(symbol, " "+symbol)
    # symbolの後の文字が空白でなければ空白を追加する
    indexes = [i for i, c in enumerate(result) if c == symbol]
    for index in indexes:
        if index+1<len(result) and result[index+1] != " ":
            result = result[:index+1] + " " + result[index+1:]
    return result

def preprocess_sent(sent):
    # 大文字を小文字にする
    after_sent = sent.lower()
    # 前にspace" "を含む時間表現10 pmなどの空白を削除する
    time_strs = re.findall(r"\d{1,2}\sam|\d{1,2}\spm", after_sent)
    for time_str in time_strs:
        after_sent = after_sent.replace(time_str, time_str.replace(" ", ""))
    # 連続する.を削除 ... ..
    multi_ps = re.findall(r"\.\.+", after_sent)
    for multi_p in multi_ps:
        after_sent = after_sent.replace(multi_p, " ")
    # 文と記号の間に空白を挟む a? -> a ?, ?a -> ? a
    after_sent = separate_symbol(after_sent, "?")
    after_sent = separate_symbol(after_sent, "!")
    after_sent = separate_symbol(after_sent, ".")
    after_sent = separate_symbol(after_sent, ",")
    return after_sent

def load_data():
    # データの読み込み
    data = {}
    for typ in config.DATA_TYPES:
        with open(os.path.join(config.DATA_1_BASE_PATH, "%s.json" % typ), "r") as f:
            data[typ] = json.load(f)
    return data

def preprocess(data):
    # データの読み込み
    for typ in data.keys():
        for i, dial in enumerate(data[typ]):
            # 各文に対する共通の前処理
            for j, utterance in enumerate(dial["dial"]):
                data[typ][i]["dial"][j] = preprocess_sent(utterance)
            for j, triplet in enumerate(dial["triplets"]):
                for k, element in enumerate(triplet):
                    data[typ][i]["triplets"][j][k] = preprocess_sent(element)
    return data



def add_vocab(tokens, vocab):
    for w in tokens:
        if not w in vocab:
            assert w != " "
            vocab[w] = len(vocab)
    return vocab

def get_ids(tokens, vocab):
    id_s = []
    for w in tokens:
        assert w in vocab
        id_s.append(vocab[w])
    return id_s

def make_datasets(data):
    vocab = {}
    add_vocab("<bos> <eos> <unk>".split(), vocab)
    # add_vocab("<bos> <eos> <unk>".split(), vocab)
    # sentを[sent, id_sent]にする
    for typ in data.keys():
        for i, dial in enumerate(data[typ]):
            for j, utterance in enumerate(dial["dial"]):
                # utterance = "<bos> %s <eos>" % utterance.strip()
                tokens = [token for token in utterance.split() if token.strip()]
                vocab = add_vocab(tokens, vocab)
                ids = get_ids(tokens, vocab)
                data[typ][i]["dial"][j] = { 
                    "tokens": tokens,
                    "ids": ids
                }
    return data, vocab



data = load_data()
data = preprocess(data)
data, vocab = make_datasets(data)
with open(config.VOCAB_PATH, "w") as f:
    json.dump(vocab, f, indent=2)
with open(config.DATASET_PATH, "w") as f:
    json.dump(data, f, indent=2)
