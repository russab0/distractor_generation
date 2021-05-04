import json
import sys
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(dir_path, os.pardir)))

from gen_onebyone.data_loader import get_feature_from_data
from itertools import combinations
from torch.nn.functional import softmax
from math import log
from utility.loss import *
from utility.tok import *
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from src.eval import load_predict_parameter

label_map = {label: i
             for i, label in enumerate(["A", "B", "C", "D"])}
LABELS = ["0", "1", "2", "3"]


class OneByOne(nn.Module):
    def __init__(self, tokenizer, pretrained, maxlen=512, **kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        self.pretrained = pretrained
        self.model = nn.Linear(self.pretrained.config.hidden_size, self.tokenizer.__len__())
        self.device = 'cuda' if torch.cuda.is_available() and not kwargs['force_cpu'] else 'cpu'
        self.qa_weight = kwargs['qa_weight']
        self.maxlen = maxlen
        print('Using device:', self.device)
        self.model.to(self.device)

    def forward(self, batch_data, eval=False):
        inputs = batch_data['input']
        targets = batch_data['target']
        negative_targets = batch_data['ntarget']
        masks = batch_data['mask']

        tokens_tensor = torch.as_tensor(inputs).to(self.device)
        mask_tensors = torch.as_tensor(masks).to(self.device)

        outputs = self.pretrained(tokens_tensor, attention_mask=mask_tensors)
        prediction_scores = self.model(outputs[0])

        if eval:
            result_dict = {
                'label_prob_all': [],
                'label_map': [],
                'prob_list': []
            }
            start = batch_data['start'][0]
            logit_prob = softmax(prediction_scores[0][start], dim=0).data.tolist()
            prob_result = {self.tokenizer.convert_ids_to_tokens(id): prob for id, prob in enumerate(logit_prob)}
            prob_result = sorted(prob_result.items(), key=lambda x: x[1], reverse=True)
            result_dict['prob_list'].append(sorted(logit_prob, reverse=True))
            result_dict['label_prob_all'].append(prob_result)
            result_dict['label_map'].append(prob_result[0])
            outputs = result_dict
        else:
            loss_tensors = torch.as_tensor(targets).to(self.device)
            negativeloss_tensors = torch.as_tensor(negative_targets).to(self.device)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)  # -1 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.pretrained.config.vocab_size),
                                      loss_tensors.view(-1))
            negative_loss_fct = NegativeCElLoss(ignore_index=-1).to(self.device)
            negative_loss = negative_loss_fct(prediction_scores.view(-1, self.pretrained.config.vocab_size),
                                              negativeloss_tensors.view(-1))
            masked_lm_loss += negative_loss
            outputs = masked_lm_loss
        return outputs

    def _jaccard_similarity(self, list1, list2):
        s1 = set(list1)
        s2 = set(list2)
        return len(s1.intersection(s2)) / len(s1.union(s2))

    def _isSimilar(self, s, t):
        return self._jaccard_similarity(s, t) > 0.5

    def _filterSimilar(self, d, topP):
        while True:
            filteredOne = False
            for s, t in combinations(d, 2):
                if self._isSimilar(s[0], t[0]) and len(d) - 1 >= topP:
                    d.remove(t)
                    filteredOne = True
                    break
            if not filteredOne:
                break

    def predict(self, input='', topK=1, topP=0.85, mode=None, decodenum=1, filtersim=True,
                reserved_len=0, task=None):
        if mode is None:
            mode = ['greedy', 'topK', 'topP']
        filtersim = json.loads(str(filtersim).lower())
        topK = int(topK)
        topP = float(topP)
        decodenum = int(decodenum)
        mode = mode[0] if isinstance(mode, list) else mode.lower()

        self.eval()
        sequences = [[[], 1.0]]
        with torch.no_grad():
            while True:
                all_candidates = list()
                exceed = False
                for seq in sequences:
                    if tok_sep(self.tokenizer) not in seq[0]:
                        tokens, score = seq
                        feature_dict = get_feature_from_data(self.tokenizer, self.maxlen, input, tokens,
                                                             reserved_len=reserved_len)
                        # check input exceed
                        if len(feature_dict['input']) > self.maxlen:
                            exceed = True
                            all_candidates.append(seq)
                            continue

                        for k, v in feature_dict.items():
                            feature_dict[k] = [v]
                        predictions = self.forward(feature_dict, eval=True)
                        token_prob_list = predictions['label_prob_all'][0]

                        # topK topP
                        if 'top' in mode:
                            prob_list = [prob for tok, prob in token_prob_list]
                            if 'topk' in mode:
                                sample_list = prob_list[:topK]
                                decode_range = max(decodenum, topK)
                                prob_norm = [float(i) / sum(sample_list) for i in sample_list]
                                choice_list = np.random.choice(sample_list, p=prob_norm,
                                                               size=decode_range,
                                                               replace=False)
                            else:
                                topP_list = np.cumsum(prob_list)
                                index_overP = [i for i, x in enumerate(topP_list) if x > topP]
                                index_overP = 0 if len(index_overP) < 1 else index_overP[0]
                                sample_list = prob_list[:index_overP + 1]
                                prob_norm = [float(i) / sum(sample_list) for i in sample_list]
                                choice_list = np.random.choice(sample_list, p=prob_norm,
                                                               size=decodenum)
                            for idx in range(decodenum):
                                sampling_index = prob_list.index(choice_list[idx])
                                k, v = token_prob_list[sampling_index]
                                candidate = [tokens + [k], score + -log(v)]
                                all_candidates.append(candidate)

                        # greedy / beam search
                        else:
                            for k, v in token_prob_list[:50]:
                                if len(tokens) > 0 and tokens[-1] == k or len(k) < 1:
                                    continue
                                candidate = [tokens + [k], score + -log(v)]
                                all_candidates.append(candidate)
                    else:
                        all_candidates.append(seq)

                ordered = sorted(all_candidates, key=lambda tup: tup[1])
                if filtersim:
                    self._filterSimilar(ordered, decodenum)
                sequences = ordered[:decodenum]
                stop = 0
                for i in sequences:
                    # i[0] - sequence,i[1] - sequence score
                    if tok_sep(self.tokenizer) in i[0] \
                            or len(i[0]) > 3 and i[0][-1] == i[0][-2] == i[0][-3] \
                            or i[1] > 300:
                        stop += 1
                if stop == len(sequences) or exceed:
                    break

            for i in range(len(sequences)):
                if tok_sep(self.tokenizer) in sequences[i][0]:  # remove sep token
                    sequences[i][0] = sequences[i][0][:sequences[i][0].index(tok_sep(self.tokenizer))]
                sequences[i][0] = "".join(self.tokenizer.convert_tokens_to_string(sequences[i][0]))
            result_dict = {
                'label_map': sequences
            }
            return [i[0] for i in sequences], result_dict


class OneByOne_QA(OneByOne):
    """
    Three options are available:
    - correct answer
    - gold distractor
    - generated distractor
    """

    def __init__(self, tokenizer, pretrained, maxlen=512, **kwargs):
        qa_model_path = kwargs.pop('qa_model', '')
        super().__init__(tokenizer, pretrained, maxlen, **kwargs)
        print('qa_model_path', qa_model_path, bool(qa_model_path))
        if qa_model_path:
            raise NotImplementedError('Not implemented')
        else:
            qa_tokenizer = AutoTokenizer.from_pretrained("russab0/distilbert-qa")
            qa_model = AutoModelForMultipleChoice.from_pretrained("russab0/distilbert-qa")
            self.qa_model = qa_model
            self.qa_tokenizer = qa_tokenizer

        self.qa_model.eval()
        self.qa_model.to(self.device)

        """self.predict_parameter = load_predict_parameter(self, False)
        self.predict_parameter.pop('input')
        if 'task' not in self.predict_parameter:
            self.predict_parameter.update({'task': 'default'})"""

    def qa(self, context, question, options, label_example):
        choices_inputs = []
        for ending_idx, (_, ending) in enumerate(zip(context, options)):
            question_option = question + " " + ending  # simple question (not fill-the-blank)

            inputs = self.qa_tokenizer(
                context,
                question_option,
                add_special_tokens=True,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_overflowing_tokens=False,
            )
            choices_inputs.append(inputs)

        label = torch.LongTensor([label_map[label_example]])
        input_ids = torch.LongTensor([
            [x["input_ids"] for x in choices_inputs]
        ])
        attention_mask = (
            torch.Tensor([[x["attention_mask"] for x in choices_inputs]])
            # as the sentences follow the same structure, just one of them is necessary to check
            if "attention_mask" in choices_inputs[0]
            else None
        )

        example_encoded = {
            # "example_id": example_id,
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "labels": label.to(self.device),
        }

        output = self.qa_model(**example_encoded)
        return output

    def get_qa_loss(self, inputs, starts):
        losses = []
        for i in range(len(inputs)):  # batch
            result, result_dict = self.predict(
                self.tokenizer.convert_tokens_to_string(
                    self.tokenizer.convert_ids_to_tokens(
                        inputs[i]
                    )
                ),
            )
            gen_distractor = result[0]

            cqa = self.tokenizer.convert_ids_to_tokens(inputs[i])[1:starts[i]]
            cqa = self.tokenizer.convert_tokens_to_string(cqa)
            context, question, answer, true_distractor = cqa.split('[SEP]')

            out = self.qa(context, question, [answer, true_distractor, gen_distractor], "A")
            losses.append(out.loss)

        return torch.Tensor(losses).mean()

    def forward(self, batch_data, eval=False):
        inputs = batch_data['input']
        targets = batch_data['target']
        negative_targets = batch_data['ntarget']
        masks = batch_data['mask']
        starts = batch_data['start']

        tokens_tensor = torch.as_tensor(inputs).to(self.device)
        mask_tensors = torch.as_tensor(masks).to(self.device)

        outputs = self.pretrained(tokens_tensor, attention_mask=mask_tensors)
        prediction_scores = self.model(outputs[0])  # == self.model(outputs.last_hidden_state)

        if eval:
            result_dict = {
                'label_prob_all': [],
                'label_map': [],
                'prob_list': []
            }
            start = batch_data['start'][0]
            topK = torch.topk(softmax(prediction_scores[0][start], dim=0), 50)
            logit_prob = softmax(prediction_scores[0][start], dim=0).data.tolist()
            prob_result = [(self.tokenizer.convert_ids_to_tokens(id), prob) for prob, id in
                           zip(topK.values.data.tolist(), topK.indices.data.tolist())]
            result_dict['prob_list'].append(logit_prob)
            result_dict['label_prob_all'].append(prob_result)
            result_dict['label_map'].append(prob_result[0])
            outputs = result_dict
        else:  # if train
            loss_tensors = torch.as_tensor(targets).to(self.device)
            negativeloss_tensors = torch.as_tensor(negative_targets).to(self.device)

            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)  # -1 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.pretrained.config.vocab_size),
                                      loss_tensors.view(-1))

            negative_loss_fct = NegativeCElLoss(ignore_index=-1).to(self.device)
            negative_loss = negative_loss_fct(prediction_scores.view(-1, self.pretrained.config.vocab_size),
                                              negativeloss_tensors.view(-1))

            if self.tokenizer.sep_token_id in targets:  # ending sample in the
                qa_loss = self.qa_weight * self.get_qa_loss(inputs, starts)
            else:
                qa_loss = 0

            # masked_lm_loss += negative_loss
            outputs = masked_lm_loss + negative_loss + qa_loss

        return outputs


MODEL_CONFIG_MAPPING = {
    'standard': OneByOne,
    'QA': OneByOne_QA
}
