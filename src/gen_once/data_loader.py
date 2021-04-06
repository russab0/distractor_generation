import csv
import os
import pickle
from collections import defaultdict
from random import choice

import numpy as np
from torch.utils import data
from tqdm import tqdm
from transformers import AutoTokenizer, BertTokenizer
from utility.tok import *


class loadOnceDataset(data.Dataset):
    def __init__(self, fpath, pretrained, maxlen=510, cache=False):
        sample = []
        if 'albert_chinese' in pretrained:
            tokenizer = BertTokenizer.from_pretrained(pretrained)
        else:
            tokenizer = AutoTokenizer.from_pretrained(pretrained)
        cache_path = fpath + "_maxlen" + str(maxlen) + "_" + pretrained.replace("/", "_") + ".cache"
        if os.path.isfile(cache_path) and cache:
            with open(cache_path, "rb") as cf:
                sample = pickle.load(cf)
        else:
            total_data = 0
            data_exceed_maxlen = 0
            for i in get_data_from_file(fpath):
                tasks, task, input, target = i
                feature = get_feature_from_data(tokenizer, maxlen, input, target)
                if len(feature['input']) == len(feature['target']) == len(feature['ntarget']) <= maxlen:
                    sample.append(feature)
                else:
                    data_exceed_maxlen += 1
                total_data += 1

            print("Processed " + str(total_data) + " data, removed " + str(
                data_exceed_maxlen) + " data that exceed the maximum length.")

            if cache:
                with open(cache_path, 'wb') as cf:
                    pickle.dump(sample, cf)

        self.sample = sample

    def increase_with_sampling(self, total):
        inc_samp = [choice(self.sample) for i in range(total - len(self.sample))]
        self.sample.extend(inc_samp)

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        self.sample[idx].update((k, np.asarray(v)) for k, v in self.sample[idx].items())
        return self.sample[idx]


def get_data_from_file(fpath):
    tasks = defaultdict(list)
    task = 'default'
    tasks[task] = []
    with open(fpath, encoding='utf') as csvfile:
        for i in tqdm(list(csv.reader(csvfile))):
            source_text = i[0]
            target_text = i[1]
            input = source_text
            target = target_text
            yield tasks, task, input, target


def get_feature_from_data(tokenizer, maxlen, input, target=None, ntarget=None):
    row_dict = dict()
    tokenized_input = [tok_begin(tokenizer)] + tokenizer.tokenize(input) + [tok_sep(tokenizer)]
    mask_id = [1] * len(tokenized_input)
    type_id = [0] * len(tokenized_input)

    row_dict['target'] = [-1] * maxlen
    row_dict['ntarget'] = [-1] * maxlen

    tokenized_input_id = tokenizer.convert_tokens_to_ids(tokenized_input)
    target_start = len(tokenized_input_id)
    if target is not None:
        tokenized_target = tokenizer.tokenize(target)
        tokenized_target += [tok_sep(tokenizer)]
        tokenized_target_id = [-1] * len(tokenized_input)
        tokenized_target_id.extend(tokenizer.convert_tokens_to_ids(tokenized_target))
        tokenized_target_id.extend([-1] * (maxlen - len(tokenized_target_id)))
        row_dict['target'] = tokenized_target_id

    if ntarget is not None:
        tokenized_ntarget = tokenizer.tokenize(ntarget)
        tokenized_ntarget_id = [-1] * target_start
        tokenized_ntarget_id.extend(tokenizer.convert_tokens_to_ids(tokenized_ntarget))
        tokenized_ntarget_id.extend([-1] * (maxlen - len(tokenized_ntarget_id)))
        row_dict['ntarget'] = tokenized_ntarget_id

    tokenized_input_id.extend([tokenizer.mask_token_id] * (maxlen - len(tokenized_input_id)))
    mask_id.extend([0] * (maxlen - len(mask_id)))
    type_id.extend([1] * (maxlen - len(type_id)))

    row_dict['input'] = tokenized_input_id
    row_dict['type'] = type_id
    row_dict['mask'] = mask_id
    row_dict['start'] = target_start

    # if True:
    #     print("*** Example ***")
    #     print("tokenized_input",input,tokenized_input)
    #     print(f"input: {len(row_dict['input'])}, {row_dict['input']} ")
    #     print(f"type: {len(row_dict['type'])}, {row_dict['type']} ")
    #     print(f"mask: {len(row_dict['mask'])}, {row_dict['mask']} ")
    #     if target is not None:
    #         print(f"target: {len(row_dict['target'])}, {row_dict['target']} ")
    #     if ntarget is not None:
    #         print("POS", target_start, len(tokenized_ntarget))
    #         print("STR", tokenized_target, tokenized_ntarget)
    #         print("ANS", tokenized_target[target_start], tokenized_ntarget_id)
    #         print(f"ntarget: {len(tokenized_ntarget_id)}, {row_dict['ntarget']} ")

    return row_dict
