def select_field(features, field):
    return [
        [choice[field] for choice in feature.choices_features]
        for feature in features
    ]


def prepare_for_qa(data, tokenizer, evaluate=False, test=False):
    assert not (evaluate and test)

    label_list = processor.get_labels()

    examples = [

    ]

    features = convert_examples_to_features(
        examples,
        label_list,
        args.max_seq_length,
        tokenizer,
        pad_on_left=False,
        pad_token_segment_id=0,
    )
    all_input_ids = torch.tensor(select_field(features, "input_ids"), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(features, "input_mask"), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(features, "segment_ids"), dtype=torch.long)
    all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset

from transformers import RobertaTokenizer
from transformers import RobertaForMultipleChoice


from transformers import BertForMultipleChoice
import torch
from transformers import RobertaTokenizer
from transformers import RobertaForMultipleChoice
from src.gen_onebyone.onebyone import OneByOne


class Model_QA_pretrained_triplet(OneByOne):
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
            # self.qa_tokenizer = self.tokenizer # shouldn't tokenizer be loaded as well?
            # self.qa_model = BertForMultipleChoice.from_pretrained('roberta-base-openai-detector')
            # self.qa_model.load_state_dict(torch.load(qa_model_path))
        else:
            # pass
            qa_tokenizer = RobertaTokenizer.from_pretrained(
                "LIAMF-USP/roberta-large-finetuned-race")
            qa_model = RobertaForMultipleChoice.from_pretrained(
                "LIAMF-USP/roberta-large-finetuned-race")
            self.qa_model = qa_model
            self.qa_tokenizer = qa_tokenizer

        self.qa_model.eval()
        self.qa_model.to(self.device)

    def qa(self, context, question, options, label_example):
        choices_inputs = []
        for ending_idx, (_, ending) in enumerate(
                zip(context, options)):

            if question.find("_") != -1:  # fill in the blanks questions
                question_option = question.replace("_", ending)
            else:
                question_option = question + " " + ending
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
            # as the senteces follow the same structure, just one of them is necessary to check
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
        for i in range(len(inputs)):
            result, result_dict = self.predict(
                tokenizer.convert_tokens_to_string(
                    tokenizer.convert_ids_to_tokens(
                        inputs[i]
                    )
                ),
                **self.predict_parameter
            )
            gen_distractor = result[0]

            # print(tokenizer.convert_ids_to_tokens(inputs[i]))
            cqa = tokenizer.convert_ids_to_tokens(inputs[i])[1:starts[i]]
            cqa = tokenizer.convert_tokens_to_string(cqa)
            # print(cqa.split('[SEP]'), len( cqa.split('[SEP]') ))
            print(cqa)
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

            qa_loss = self.get_qa_loss(inputs, starts)

            # masked_lm_loss += negative_loss
            outputs = masked_lm_loss + negative_loss + qa_loss

        return outputs

