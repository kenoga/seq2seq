# -*- coding: utf-8 -*-

DATASET_PATH = "/Users/nogaken/GoogleDrive/Lab/work/kvretnets/dataset/dataset.json"
VOCAB_PATH = "/Users/nogaken/GoogleDrive/Lab/work/kvretnets/dataset/vocab.json"

PREPROCESSED_DATASET_PATH = "./dataset_preprocessed/preprocessed_data.json"

TASK_NAMES = ["schedule"]
# 今回扱う対象はscheduleタスクのみ
TASK_NAME = "schedule"

DATA_TYPES = ["train", "dev", "test"]

DATA_0_BASE_PATH = "./kvret_dataset_public/"
DATA_1_BASE_PATH = "./dataset_split/tasks/%s" % TASK_NAME
