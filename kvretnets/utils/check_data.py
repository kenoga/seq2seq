    # -*- coding: utf-8 -*-

    from prettyprint import pp
    import json

    train = None
    with open("./kvret_dataset_public/kvret_train_public.json") as f:
        train = json.load(f)


    for dial in train:
        for turn in dial["dialogue"]:
            pp(turn["data"]["utterance"])
        break
        