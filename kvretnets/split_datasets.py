# -*- coding: utf-8 -*-

from prettyprint import pp
import json

types = ["train", "dev", "test"]
tasks = ["schedule", "navigate", "weather"]

for typ in types:
    with open("./kvret_dataset_public/kvret_%s_public.json" % typ, "r") as f:
        json_data = json.load(f)
    data = {}
    
    # taskごとにデータを整理するために分ける
    for task in tasks:
        data[task] = {}
        data[task][typ] = []
        
    for dial in json_data:
        dic = {}
        dic["dial"] = []
        dic["triplets"] = []
        for turn in dial["dialogue"]:
            dic["dial"].append(turn["data"]["utterance"])
        
        # task名の取得
        task = dial["scenario"]["task"]["intent"]
        cols = dial["scenario"]["kb"]["column_names"]
        items = dial["scenario"]["kb"]["items"]
        if items:
            for item in items:
                if "poi" in cols:
                    for col in [col for col in cols if not col == "poi"]:
                        dic["triplets"].append((item["poi"], col, item[col]))
                elif "event" in cols:
                    for col in [col for col in cols if not col == "event"]:
                        dic["triplets"].append((item["event"], col, item[col]))
                elif "location" in cols:
                    dic["triplets"].append(["day", "today", item["today"]])
                    for col in [col for col in cols if not col in {"location", "today"}]:
                        dic["triplets"].append((item["location"], col, item[col]))
                else:
                    raise Exception()
        data[task][typ].append(dic)
        
    for task in data.keys():
        with open("dataset_split/tasks/%s/%s.json" % (task, typ), "w") as f:
            json.dump(data[task][typ], f, ensure_ascii=False, indent=4)
