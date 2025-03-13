import os
import json
import random
import argparse
from utils.common import manual_seed

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, required = True, choices = ("MATH", "WildChat10K", ))
parser.add_argument("--synthetic", type = str, default = "synthetic")
parser.add_argument("--seed", type = int, default = 0)
parser.add_argument("--base_prob", type = float, default = 0.7)
parser.add_argument("--prob_drate", type = float, required = True)
args = parser.parse_args()
manual_seed(args.seed)

if args.dataset == "MATH" :
    results_type = "accuracy"
elif args.dataset == "WildChat10K" :
    results_type = "win-rate"
else :
    raise NotImplementedError("dataset = {}".format(args.dataset))


with open("Datasets/{}/eval_results/{}/ground-truth.json".format(args.dataset, args.synthetic), "r") as fin :
    capabilities = json.load(fin)
with open("Datasets/{}/AssociatedInstances_[gpt-4o-mini].json".format(args.dataset), "r") as fin :
    cache = json.load(fin)


attacks = None
for index, capability in enumerate(capabilities) :
    if attacks is None :
        attacks = [([index] if label == "YES" else []) for label in cache[capability]]
    else :
        for attack, label in zip(attacks, cache[capability]) :
            if label == "YES" :
                attack.append(index)

results = []
for attack in attacks :
    prob = args.base_prob * (args.prob_drate ** len(attack))
    if results_type == "accuracy" :
        results.append(random.choices([1, 0], weights = [prob, 1 - prob], k = 1)[0])
    elif results_type == "win-rate" :
        results.append([
            random.choices([1, 2], weights = [prob, 1 - prob], k = 1)[0],
            random.choices([1, 2], weights = [prob, 1 - prob], k = 1)[0],
        ])
    else :
        raise NotImplementedError("results_type = {}".format(results_type))


os.makedirs("Datasets/{}/eval_results/{}/[base={}]_[drate={}]_[seed={}]".format(args.dataset, args.synthetic, args.base_prob, args.prob_drate, args.seed), exist_ok = True)
with open("Datasets/{}/eval_results/{}/[base={}]_[drate={}]_[seed={}]/results.json".format(args.dataset, args.synthetic, args.base_prob, args.prob_drate, args.seed), "w") as fout :
    json.dump(results, fout, indent = 2)