import csv
import os
import pickle
from collections import defaultdict
from random import choice

import numpy as np
from torch.utils import data
from transformers import AutoTokenizer, BertTokenizer
from utility.tok import *
import gen_once
from tqdm import tqdm
import datasets
import hashlib
import nlp2

COLUMNS = ['source_text', 'target_text', 'negative_text']


def loadOneByOneDataset(fpath, pretrained_config, maxlen=510, cache=False, likelihood='', pos_ratio=1, neg_ratio=1,
                        **kwargs):
    if 'albert_chinese' in pretrained_config:
        tokenizer = BertTokenizer.from_pretrained(pretrained_config)
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_config)

    dataset = datasets.load_dataset('csv',
                                    data_files=fpath,
                                    names=COLUMNS,
                                    split='train',
                                    cache_dir=fpath[:fpath.rfind('/')])
    print('loaded', dataset)

    fun_args = nlp2.function_get_all_arg(get_feature_from_data)
    fingerprint_name = fpath + 'transformed' + str('handle_exceed_' in fun_args or 'handle_exceed' in fun_args)
    dataset = dataset.map(
        lambda x: mapping(
            item=x,
            tokenizer=tokenizer,
            maxlen=maxlen,
            likelihood=likelihood,
            pos_ratio=pos_ratio,
            neg_ratio=neg_ratio,
        ),
        batched=True,
        batch_size=1,
        remove_columns=COLUMNS,
        new_fingerprint=hashlib.sha224(bytearray(fingerprint_name, 'utf8')).hexdigest()
    )
    # dataset.set_format('torch')

    return dataset


def mapping(item, tokenizer, maxlen, likelihood, pos_ratio, neg_ratio):
    total_data = 0
    data_invalid = 0

    input = item['source_text'][0]
    target = item['target_text'][0].strip().split(" ")
    negative_text = item['negative_text'][0].strip()

    input = input.strip()
    tokenized_target = tokenizer.tokenize(" ".join(target))

    # each word in sentence
    sample = []
    for j in range(1, len(tokenized_target) + 1):
        feature = get_feature_from_data(tokenizer, maxlen, input, tokenized_target[:j - 1],
                                        tokenized_target[:j])
        if "neg" in likelihood or 'both' in likelihood:
            # formatting neg data in csv
            if negative_text is None:
                ntext_arr = [tokenizer.convert_tokens_to_string(tokenized_target[:j - 1])]
            elif "[SEP]" in negative_text:
                ntext_arr = [ntext.strip() for ntext in negative_text.split("[SEP]")]
            else:
                ntext_arr = [negative_text.strip()]
            # adding neg data
            for neg_text in ntext_arr:
                feature_neg = gen_once.data_loader.get_feature_from_data(tokenizer, maxlen, input,
                                                                         ntarget=neg_text)
                feature['ntarget'] = feature_neg['ntarget']
                if check_feature_valid(feature, maxlen):
                    print(feature, type(feature))
                    sample.append(feature)
                else:
                    data_invalid += 1
                total_data += 1
        else:
            if check_feature_valid(feature, maxlen):
                sample.append(feature)
            else:
                data_invalid += 1
            total_data += 1

    # end of the last word
    feature = get_feature_from_data(tokenizer, maxlen, input, tokenized_target, [tok_sep(tokenizer)])
    if "neg" in likelihood or 'both' in likelihood:
        # formatting neg data in csv
        if negative_text is None:
            ntext_arr = [tokenizer.convert_tokens_to_string(tokenized_target[:j - 1])]
        elif "[SEP]" in negative_text:
            ntext_arr = [ntext.strip() for ntext in negative_text.split("[SEP]")]
        else:
            ntext_arr = [negative_text.strip()]
        # adding neg data
        for neg_text in ntext_arr:
            feature_neg = gen_once.data_loader.get_feature_from_data(tokenizer, maxlen, input,
                                                                     ntarget=neg_text)
            feature['ntarget'] = feature_neg['ntarget']
            if check_feature_valid(feature, maxlen):
                sample.append(feature)
            else:
                data_invalid += 1
            total_data += 1
    else:
        if check_feature_valid(feature, maxlen):
            sample.append(feature)
        else:
            data_invalid += 1
        total_data += 1

        # whole sentence masking
        if 'pos' in likelihood or 'both' in likelihood:
            feature = gen_once.data_loader.get_feature_from_data(tokenizer, maxlen, input,
                                                                 " ".join(target))
            if check_feature_valid(feature, maxlen):
                for _ in range(int(pos_ratio)):
                    sample.append(feature)
            else:
                data_invalid += 1
            total_data += 1

    return {k: np.array([dic[k] for dic in sample]) for k in sample[0]}  # from list of dicts to dict of lists


def check_feature_valid(feature, maxlen):
    if len(feature['input']) == len(feature['target']) == len(feature['ntarget']) == maxlen:
        if feature['target'][feature['start']] == feature['ntarget'][feature['start']]:
            feature['ntarget'][feature['start']] = -1
        return True
    else:
        return False


def get_data_from_file(fpath):
    tasks = defaultdict(list)
    task = 'default'
    tasks[task] = []
    with open(fpath, encoding='utf') as csvfile:
        for i in tqdm(list(csv.reader(csvfile))):
            source_text = i[0]
            target_text = i[1].strip().split(" ")
            negative_text = i[2].strip() if len(i) > 2 else None
            input = source_text
            target = target_text
            yield tasks, task, input, target, negative_text


# new version
def handle_exceed(tokenizer, seq, maxlen, mode=['noop', 'remove', 'slide', 'start_slice', 'end_slice'],
                  keep_after_sep=True):
    mode = mode[0] if isinstance(mode, list) else mode
    seq = seq.replace("[MASK]", tok_mask(tokenizer)).replace("[SEP]", tok_sep(tokenizer)).replace("[CLS]",
                                                                                                  tok_begin(tokenizer))
    sep_split = seq.split(tok_sep(tokenizer))
    ext_seq = [tok_sep(tokenizer)] + tokenizer.tokenize(tok_sep(tokenizer).join(sep_split[1:])) \
        if len(sep_split) > 1 and keep_after_sep else []
    t_seq = tokenizer.tokenize(sep_split[0])
    if mode == 'noop':
        return [t_seq + ext_seq], [[0, len(t_seq + ext_seq)]]
    if mode == 'remove':
        if len(t_seq + ext_seq) <= maxlen:
            return [t_seq + ext_seq], [[0, len(t_seq + ext_seq)]]
        else:
            return [], [[0, 0]]
    if mode == 'slide':
        return nlp2.sliding_windows(t_seq, maxlen - len(ext_seq), append_seq=ext_seq)
    if mode == 'start_slice':
        slices = t_seq[:maxlen - len(ext_seq)]
        slices.extend(ext_seq)
        return [slices], [[0, maxlen - len(ext_seq)]]
    if mode == 'end_slice':  # seems like it does not work
        start_pos = len(t_seq) + len(ext_seq) - maxlen
        slices = t_seq[start_pos:]
        slices.extend(ext_seq)
        return [slices], [[max(0, start_pos), len(t_seq)]]


# new version
def get_feature_from_data(tokenizer, maxlen, input, previous, target=None, ntarget=None, reserved_len=0,
                          handle_exceed_='noop', **kwargs):  # TODO was noop
    feature_dict_list = []
    t_input_list, _ = handle_exceed(tokenizer, input, maxlen - 2 - len(previous) - 1,
                                    handle_exceed_)  # -2 for cls and sep
    #print(t_input_list[0])
    for t_input in t_input_list:
        row_dict = dict()
        t_input = [tok_begin(tokenizer)] + \
                  t_input[:maxlen - reserved_len - 2] + \
                  [tok_sep(tokenizer)]
        t_input.extend(previous)
        t_input.append(tok_mask(tokenizer))
        t_input_id = tokenizer.convert_tokens_to_ids(t_input)
        mask_id = [1] * len(t_input)
        target_start = len(t_input_id) - 1
        target_end = maxlen
        t_input_id.extend([0] * (maxlen - len(t_input_id)))
        row_dict['target'] = [-1] * maxlen
        row_dict['ntarget'] = [-1] * maxlen
        tokenized_target_id = None
        if target is not None:
            tokenized_target_id = [-1] * target_start
            tokenized_target_id.append(tokenizer.convert_tokens_to_ids(target)[-1])
            target_end = len(tokenized_target_id) - 1
            tokenized_target_id.extend([-1] * (maxlen - len(tokenized_target_id)))
            row_dict['target'] = tokenized_target_id
        if ntarget is not None and len(tokenizer.tokenize(ntarget)) > 0:
            tokenized_ntarget = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ntarget))
            tokenized_ntarget_id = [-1] * target_start
            tokenized_ntarget_id.extend(tokenized_ntarget)
            tokenized_ntarget_id.extend([-1] * (maxlen - len(tokenized_ntarget_id)))
            if len(tokenized_ntarget_id) <= maxlen:
                row_dict['ntarget'] = tokenized_ntarget_id

        mask_id.extend([0] * (maxlen - len(mask_id)))
        type_id = [0] * len(t_input)
        type_id.extend([1] * (maxlen - len(type_id)))
        row_dict['input'] = t_input_id
        # row_dict['type'] = type_id
        row_dict['mask'] = mask_id
        row_dict['start'] = target_start
        row_dict['end'] = target_end
        # ntarget - negative target
        # target - label
        feature_dict_list.append(row_dict)
    return feature_dict_list[0]
