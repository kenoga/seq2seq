# -*- coding: utf-8 -*-

import os, sys
import json
sys.path.append(os.pardir)
import config
from prettyprint import pp


def load_datasets():
    data = None
    with open(config.DATASET_PATH, "r") as f:
        data = json.load(f)
    return data

# 単語idを食ってその単語を含む発話文を返す
# debugに使う
def get_utterance_by_vocab_id(id, data):
    for typ in data.keys():
        for dial in data[typ]:
            for utterance in dial["dial"]:
                if id in utterance[1]:
                    index = utterance[1].index(id)
                    return utterance[0], utterance[0].split(" ")[index], dial
    return None, None
        
def get_utterance_by_phrase(phrase, data):
    for typ in data.keys():
        for dial in data[typ]:
            for utterance in dial["dial"]:
                if phrase in utterance[0]:
                    return utterance[0], dial
    return None, None, None

