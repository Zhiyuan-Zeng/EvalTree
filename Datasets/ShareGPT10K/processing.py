# Download https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json
import os
import json
import random
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", token = os.getenv("HF_TOKEN"))

with open("ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json", "r") as fin :
    dataset = json.load(fin)
    def check_instance(instance) :
        return \
            instance["id"].endswith("_0") and \
            len(instance["conversations"]) >= 2 and \
            instance["conversations"][0]["from"] == "human" and instance["conversations"][1]["from"] == "gpt" and \
            len(instance["conversations"][0]["value"].strip()) and len(instance["conversations"][1]["value"].strip()) and \
            instance["conversations"][1]["value"] != "\u200b" and \
            len(tokenizer.tokenize(instance["conversations"][0]["value"]) + tokenizer.tokenize(instance["conversations"][1]["value"])) <= 4096
    dataset = list(filter(check_instance, dataset))
    basket = {instance["conversations"][0]["value"].strip().lower() : instance["id"] for instance in dataset}
    dataset = list(filter(lambda instance : basket[instance["conversations"][0]["value"].strip().lower()] == instance["id"], dataset))
    dataset = [{"instruction" : instance["conversations"][0]["value"], "response" : instance["conversations"][1]["value"]} for instance in dataset]
print(len(dataset))
assert len(dataset) >= 10000

random.seed(42)
random.shuffle(dataset)
with open("dataset.json", "w") as fout :
    json.dump(dataset[: 10000], fout, indent = 2)