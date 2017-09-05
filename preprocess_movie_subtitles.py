# -*- encoding: utf-8 -*- 

import json
import numpy as np

CORPUS_FILE = './chat_corpus/movie_subtitles_en.txt'
EN_WHITE_CHAR_SET = set('0123456789abcdefghijklmnopqrstuvwxyz ')

MAX_LEN = 25
MIN_LEN = 2
VOCAB_SIZE = 8000
MAX_UNK_RATE = 0.5

def read_lines(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
    return lines

def filter_char(line, EN_WHITELIST):
    return ''.join(c for c in line if c in EN_WHITE_CHAR_SET)

def tokenize(lines):
    return [[t.strip() for t in line.split() if t] for line in lines]

def separate_q_and_a(lines):
    q_lines = [line for i, line in enumerate(lines) if i % 2 == 0]
    a_lines = [line for i, line in enumerate(lines) if i % 2 == 1]
    return q_lines, a_lines
    
def filter_line(q_lines, a_lines, max_len, min_len, unk_rate, unk_id):
    n_q_lines = []
    n_a_lines = []
    for q_line, a_line in zip(q_lines, a_lines):
        # fileter by length
        if len(q_line) <= max_len and len(q_line) >= min_len \
            and len(a_line) <= max_len and len(a_line) >= min_len:
            # filter by unk rate
            q_unk_count = float(len([t_id for t_id in q_line if t_id == unk_id]))
            a_unk_count = float(len([t_id for t_id in a_line if t_id == unk_id]))
            if q_unk_count/len(q_line) <= unk_rate and a_unk_count/len(a_line) <= unk_rate:
                n_q_lines.append(q_line)
                n_a_lines.append(a_line)
    return n_q_lines, n_a_lines

def make_token_count_dict(tokens):
    token2count = {}
    for token in tokens:
        if token in token2count:
            token2count[token] += 1
        else:
            token2count[token] = 1
    return token2count

def make_vocab_by_vocab_size(token2count, vocab_size):
    vocab = {}
    vocab['<unk>'] = 0
    vocab['<bos>'] = 1
    vocab['<eos>'] = 2
    sorted_t2c = sorted(token2count.items(), key=lambda d: d[1], reverse=True)
    for t2c in sorted_t2c:
        if len(vocab) < vocab_size:
            vocab[t2c[0]] = len(vocab)
    return vocab

def tokens2ids(tokens, vocab):
    return [vocab[token] if token in vocab else vocab['<unk>'] for token in tokens]

def ids2tokens(ids, vocab):
    r_vocab = {v:k for k, v in vocab.items()}
    return [r_vocab[id] for id in ids]

def split_dataset(x, y, rates=[0.8, 0.1, 0.1]):
    dataset = {
        'train': {},
        'dev': {},
        'test': {},
    }
    assert sum([rate*100 for rate in rates]) == 100
    data_len  = len(x)
    lens = [int(data_len*rate) for rate in rates]
    train_x = x[:lens[0]]
    train_y = y[:lens[0]]
    dataset['train']['x'] = x[:lens[0]]
    dataset['train']['y'] = y[:lens[0]]
    dataset['dev']['x'] = x[lens[0]:lens[0]+lens[1]]
    dataset['dev']['y'] = y[lens[0]:lens[0]+lens[1]]
    dataset['test']['x'] = x[lens[0]+lens[1]:]
    dataset['test']['y'] = y[lens[0]+lens[1]:]
    return dataset


lines = read_lines(CORPUS_FILE)
lines = [line.lower() for line in lines]
lines = [filter_char(line, EN_WHITE_CHAR_SET) for line in lines]
tokens_list = tokenize(lines)
# flatten tokens list to make vocab
tokens = []
for ts in tokens_list:
    tokens.extend(ts)

# count token
token2count = make_token_count_dict(tokens)
vocab = make_vocab_by_vocab_size(token2count, VOCAB_SIZE)

unk_count = len([token for token in tokens if token not in vocab])
token_count = len(tokens)
print("unknown token rate: %3.2f" % (unk_count / float(token_count) * 100) + "%")

# token -> id
ids_list = [tokens2ids(tokens, vocab) for tokens in tokens_list]

# separate query and answer
q_ids_list, a_ids_list = separate_q_and_a(ids_list)
assert len(q_ids_list) == len(a_ids_list)

# filter line by length, and unknown token rate
q_ids_list, a_ids_list = filter_line(q_ids_list, a_ids_list, MAX_LEN, MIN_LEN, MAX_UNK_RATE, 0)
assert len(q_ids_list) == len(a_ids_list)

# get splitted datset (train, dev, test)
dataset = split_dataset(q_ids_list, a_ids_list)
assert len(q_ids_list) == (len(dataset['train']['x']) + len(dataset['dev']['x']) + len(dataset['test']['x']))


dataset_fname = './dataset/movie_subtitles_en/dataset.json'
with open(dataset_fname, 'w') as f:
    json.dump(dataset, f, indent=2)

vocab_fname = './dataset/movie_subtitles_en/vocab.json'
with open(vocab_fname, 'w') as f:
    json.dump(vocab, f, indent=2)



# TODO
# save to file
# make_minibatch






