import os
import json
import random
import datasets
from transformers import AutoTokenizer
random.seed(42)


dataset = datasets.load_dataset("allenai/WildChat")["train"].to_list()
dataset = random.sample(dataset, 50000)


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", token = os.getenv("HF_TOKEN"))
def check_instance(instance) :
    return len(instance["conversation"]) >= 2 and \
            instance["conversation"][0]["language"] == "English" and instance["conversation"][1]["language"] == "English" and \
            len(instance["conversation"][0]["content"].strip()) and len(instance["conversation"][1]["content"].strip()) and \
            len(tokenizer.tokenize(instance["conversation"][0]["content"]) + tokenizer.tokenize(instance["conversation"][1]["content"])) <= 4096
dataset = list(filter(check_instance, dataset))
basket = {instance["conversation"][0]["content"].strip().lower()[: 512] : instance["conversation_id"] for instance in dataset}
dataset = list(filter(lambda instance : basket[instance["conversation"][0]["content"].strip().lower()[: 512]] == instance["conversation_id"], dataset))
dataset = [{"instruction" : instance["conversation"][0]["content"], "response" : instance["conversation"][1]["content"]} for instance in dataset]
print(len(dataset))
assert len(dataset) >= 10000


random.shuffle(dataset)
with open("dataset.json", "w") as fout :
    json.dump(dataset[: 10000], fout, indent = 2)