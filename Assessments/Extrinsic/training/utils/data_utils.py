import json
import copy
import torch
import dataclasses
import transformers
from torch import Tensor
from typing import Sequence
from torch.utils.data import Dataset


def check_tensor_all_equal(a : torch.Tensor, b : torch.Tensor) :
    return len(a) == len(b) and torch.all(a == b)


def chatml2str(tokenizer, chatml : list[str], data_args) -> str :
    messages = []
    def process_input(input, data_args) :
        if data_args.SFT_format == "mathematics" :
            return input + "\n" + "Please reason step by step, and put your final answer within \\boxed{}."
        else :
            raise NotImplementedError("Unknown SFT format = {}".format(data_args.SFT_format))
    if len(chatml) == 1 :
        messages.append({"role" : "user", "content" : process_input(chatml[0], data_args)})
        return tokenizer.apply_chat_template(messages, add_generation_prompt = True, tokenize = False)
    elif len(chatml) == 2 :
        messages.append({"role" : "user", "content" : process_input(chatml[0], data_args)})
        messages.append({"role" : "assistant", "content" : chatml[1]})
        return tokenizer.apply_chat_template(messages, add_generation_prompt = False, tokenize = False)
    else :
        raise NotImplementedError


def _tokenize_fn(strings : Sequence[str], tokenizer : transformers.PreTrainedTokenizer) -> list[Tensor] :
    tokenized_list = [
        tokenizer(
            text,
            return_tensors = "pt",
            max_length = tokenizer.model_max_length,
            truncation = True,
            add_special_tokens = False,
        )
        for text in strings
    ]
    input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
    return input_ids


@dataclasses.dataclass
class DataCollatorForVanillaDataset() :
    tokenizer : transformers.PreTrainedTokenizer

    def __call__(self, instances : Sequence[dict]) -> dict[str, Tensor] :
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first = True, padding_value = self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first = True, padding_value = -100)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()

        return dict(
            input_ids = input_ids,
            labels = labels,
            attention_mask = attention_mask,
        )


class FinetuneDataset(Dataset) :
    def __init__(self, raw_data, tokenizer, data_args) :
        super().__init__()

        sources = [tokenizer.bos_token + instance["input"] for instance in raw_data]
        examples = [(tokenizer.bos_token + instance["input"]) + (instance["output"] + tokenizer.eos_token) for instance in raw_data]
        examples_tokenized, sources_tokenized = _tokenize_fn(examples, tokenizer), _tokenize_fn(sources, tokenizer)

        input_ids = examples_tokenized
        labels = copy.deepcopy(input_ids)
        for label, source in zip(labels, sources_tokenized) :
            source = source[: source.ne(tokenizer.pad_token_id).sum().item()]
            assert (source != tokenizer.pad_token_id).all()
            assert check_tensor_all_equal(label[: len(source)], source)
            label[: len(source)] = -100
        
        self.input_ids, self.labels = input_ids, labels

    def __len__(self) :
        return len(self.input_ids)

    def __getitem__(self, i) -> dict[str, Tensor] :
        return dict(input_ids = self.input_ids[i], labels = self.labels[i])

def make_Finetunedata_module(raw_data, tokenizer, data_args) :
    train_dataset = FinetuneDataset(
        raw_data = raw_data,
        tokenizer = tokenizer,
        data_args = data_args,
    )
    data_collator = DataCollatorForVanillaDataset(tokenizer = tokenizer)
    return dict(train_dataset = train_dataset, data_collator = data_collator)


class SFTDataset(Dataset) :
    def __init__(self, raw_data, tokenizer, data_args) :
        super().__init__()

        sources = [chatml2str(tokenizer, [instance["input"], ], data_args) for instance in raw_data]
        examples = [chatml2str(tokenizer, [instance["input"], instance["output"]], data_args) for instance in raw_data]
        examples_tokenized, sources_tokenized = _tokenize_fn(examples, tokenizer), _tokenize_fn(sources, tokenizer)

        input_ids = examples_tokenized
        labels = copy.deepcopy(input_ids)
        for label, source in zip(labels, sources_tokenized) :
            source = source[: source.ne(tokenizer.pad_token_id).sum().item()]
            assert (source != tokenizer.pad_token_id).all()
            assert check_tensor_all_equal(label[: len(source)], source)
            label[: len(source)] = -100
        
        self.input_ids, self.labels = input_ids, labels

    def __len__(self) :
        return len(self.input_ids)

    def __getitem__(self, i) -> dict[str, Tensor] :
        return dict(input_ids = self.input_ids[i], labels = self.labels[i])

def make_SFTdata_module(raw_data, tokenizer, data_args) :
    train_dataset = SFTDataset(
        raw_data = raw_data,
        tokenizer = tokenizer,
        data_args = data_args,
    )
    data_collator = DataCollatorForVanillaDataset(tokenizer = tokenizer)
    return dict(train_dataset = train_dataset, data_collator = data_collator)


def make_data_module(tokenizer, data_args, training_args) :
    with open(data_args.dataset_path, "r") as fin :
        raw_data = json.load(fin)

    def split_train_into_train_and_eval(train_dataset, eval_size : int, seed : int) -> tuple[Dataset, Dataset] :
        assert eval_size in range(1, len(train_dataset))
        new_train_size = len(train_dataset) - eval_size
        def _get_generator(seed : int) -> torch.Generator :
            rng = torch.Generator()
            rng.manual_seed(seed)
            return rng
        train_dataset, eval_dataset = torch.utils.data.random_split(
            train_dataset, [new_train_size, eval_size], generator =_get_generator(seed)
        )
        return train_dataset, eval_dataset

    if data_args.dataset_type == "Finetune" :
        raw_data = list(filter(lambda instance : len(_tokenize_fn([instance["input"] + instance["output"]], tokenizer)[0]) < training_args.model_max_length, raw_data))

        data_module = make_Finetunedata_module(raw_data, tokenizer, data_args)
        data_module["train_dataset"], data_module["eval_dataset"] = split_train_into_train_and_eval(
            train_dataset = data_module["train_dataset"],
            eval_size = data_args.eval_size,
            seed = training_args.seed,
        )
    elif data_args.dataset_type == "SFT" :
        raw_data = list(filter(lambda instance : len(_tokenize_fn([chatml2str(tokenizer, [instance["input"], ], data_args)], tokenizer)[0]) < training_args.model_max_length, raw_data))

        data_module = make_SFTdata_module(raw_data, tokenizer, data_args)
        data_module["train_dataset"], data_module["eval_dataset"] = split_train_into_train_and_eval(
            train_dataset = data_module["train_dataset"],
            eval_size = data_args.eval_size,
            seed = training_args.seed,
        )
    else :
        raise NotImplementedError

    return data_module