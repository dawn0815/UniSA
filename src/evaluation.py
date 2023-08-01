import json
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, "pycocoevalcap")

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider


# based on visual-comet
def use_same_id(sent):
    r_sent = sent.replace("'", " '")
    r_sent = ' '.join([g if not g.isdigit() else '1' for g in r_sent.split()]).strip()
    r_sent = r_sent.replace(" '", "'")
    return r_sent


# based on visual-comet
def compute_metric_inference(gens_list, refs_list, calculate_diversity=False, train_file=None):
    scorers = [
        (Bleu(4), ["BLEU1", "BLEU2", "BLEU3", "BLEU4"]),
        (Meteor(), "METEOR"),
        (Cider(), "CIDEr")
    ]
    tokenizer = PTBTokenizer()

    refs = {}
    preds = {}
    output = {}
    cnt = 0

    for i, gens in tqdm(enumerate(gens_list)):
        ref_index = gens['index']
        relation = gens['task_type']
        ref = refs_list[ref_index][relation]
        if len(ref) > 0:
            for pred in gens['generations']:
                preds[cnt] = [{'caption': pred}]
                refs[cnt] = [{'caption': r} for r in ref]
                cnt += 1

    refs = tokenizer.tokenize(refs)
    preds = tokenizer.tokenize(preds)

    if calculate_diversity:
        unique_sents = []
        novel_sents = []

        # store train sentence to calculate novelty
        train_sents = json.load(open(train_file))
        ts = set()
        for d in train_sents:
            for r in ['intent', 'before', 'after']:
                if r in d:
                    for sent in d[r]:
                        r_sent = use_same_id(sent)
                        ts.add(r_sent)

        for pred in preds.values():
            pred_same_id = use_same_id(pred[0])
            unique_sents.append(pred_same_id)
            novel_sents.append(pred_same_id not in ts)

        print(len(unique_sents))
        unique = len(set(unique_sents)) / len(unique_sents)
        output['Unique'] = unique
        print('Unique Inferences:', unique)

        novel = np.mean(novel_sents)
        output['Novel'] = novel
        print('Novel Inferences:', novel)

    for scorer, method in scorers:
        score, scores = scorer.compute_score(refs, preds)
        if type(method) == list:
            for m in range(len(method)):
                output[method[m]] = score[m]
                print(method[m], score[m])
        else:
            output[method] = score
            print(method, score)

    return output
