# -*- coding: utf-8 -*-

import os

BASE_PATH = "./"
DATASET_PATH = os.path.join(BASE_PATH, "dataset/dataset.json")
VOCAB_PATH = os.path.join(BASE_PATH, "dataset/vocab.json")

TASK_NAMES = ["schedule", "navigate", "weather"]
# 対象
TASK_NAME = "schedule"

DATA_TYPES = ["train", "dev", "test"]
DATA_0_BASE_PATH = "./kvret_dataset_public/"
DATA_1_BASE_PATH = "./dataset_split/tasks/%s" % TASK_NAME
