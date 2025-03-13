import os
import json
import random
import datasets
import argparse
from utils.common import manual_seed

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, required = True, choices = ("MATH", "DS-1000", ))
parser.add_argument("--config", type = str, required = True)
parser.add_argument("--seed", type = int, default = True)
parser.add_argument("--instance_num", type = int, default = 128)
args = parser.parse_args()
manual_seed(args.seed)


with open("Assessments/Extrinsic/data/generate_data/configs/{}/{}.json".format(args.dataset, args.config), "r") as fin :
    config = json.load(fin)


data = []
if config["input"].startswith("[input-generation=") and config["input"].endswith("]") :
    with open("Assessments/Extrinsic/data/pools/{}/{}.json".format(args.dataset, config["input"]), "r") as fin :
        cache = json.load(fin)
    with open(config["capability_path"], "r") as fin :
        inputs = [capability_input for capability in json.load(fin) for capability_input in cache[capability["capability"]]]
elif config["input"] == "original" :
    if args.dataset == "MATH" :
        dataset = datasets.load_dataset("lighteval/MATH")["test"].to_list()
        with open("Datasets/MATH/splits/4k-1k.json", "r") as fin :
            dataset = [dataset[index] for index in json.load(fin)]
        inputs = [instance["problem"] for instance in dataset]
        dataset = None
    elif args.dataset == "DS-1000" :
        dataset = datasets.load_dataset("xlangai/DS-1000")["test"].to_list()
        with open("Datasets/DS-1000/splits/600-400.json", "r") as fin :
            dataset = [dataset[index] for index in json.load(fin)]
        inputs = [instance["prompt"] for instance in dataset]
        dataset = None
    else :
        raise NotImplementedError("dataset = {}".format(args.dataset))
else :
    raise NotImplementedError("config[\"input\"] = {}".format(config["input"]))

with open("Assessments/Extrinsic/data/pools/{}/{}.json".format(args.dataset, config["output"]), "r") as fin :
    input2output = json.load(fin)


assert len(inputs) >= args.instance_num

for input in random.sample(inputs, args.instance_num) :
    def processing(output) :
        if args.dataset == "DS-1000" :
            output = "\n" + output
        else :
            pass
        return output
    data.append({
        "input" : input,
        "output" : processing(input2output[input]),
    })


os.makedirs("Assessments/Extrinsic/training/data/{}".format(args.dataset), exist_ok = True)
with open("Assessments/Extrinsic/training/data/{}/{}_[seed={}].json".format(args.dataset, args.config, args.seed), "w") as fout :
    json.dump(data, fout, indent = 2)