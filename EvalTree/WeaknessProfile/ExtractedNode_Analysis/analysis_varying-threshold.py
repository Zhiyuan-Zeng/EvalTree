import os
import json
import torch
import argparse
import numpy as np
from EvalTree.WeaknessProfile.extract_subtrees import extract_subtrees

parser = argparse.ArgumentParser()
parser.add_argument("--tree_dataset", type = str, required = True, choices = ("MATH", "MMLU", "DS-1000", "WildChat10K", ))
parser.add_argument("--tree_path", type = str, required = True)

parser.add_argument("--embedding_dataset", type = str, required = True, choices = ("MATH", "CollegeMath", ) + ("MMLU", ) + ("DS-1000", ) + ("WildChat10K", "ShareGPT10K", "Chatbot-Arena", ))
parser.add_argument("--embedding_path", type = str, default = "stage2-CapabilityEmbedding/[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]", choices = ("stage2-CapabilityEmbedding/[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]", ))
parser.add_argument("--embedding_split", type = str, required = True)

parser.add_argument("--results_path", type = str, required = True)

parser.add_argument("--direction", type = str, required = True, choices = ("higher", "lower"))
parser.add_argument("--alpha", type = float, default = 0.05)
args = parser.parse_args()


output_path = args.tree_path.split("/")
assert len(output_path) == 2
output_path = "EvalTree/TREE=[{}]_{}".format(output_path[0], output_path[1])
with open(os.path.join("Datasets/{}/eval_results".format(args.tree_dataset), args.results_path, output_path, "confidence_interval.json"), "r") as fin :
    TREE_RESULTS = json.load(fin)

output_path = args.tree_path.split("/")
assert len(output_path) == 2
output_path = "[dataset={}]_[{}]_{}".format(args.tree_dataset, output_path[0], output_path[1])
with open(os.path.join("Datasets/{}/EvalTree/{}_[located-split={}]".format(args.embedding_dataset, args.embedding_path, args.embedding_split), "{}.json".format(output_path)), "r") as fin :
    INSTANCE2PATH = json.load(fin)

with open(os.path.join("Datasets/{}/eval_results".format(args.embedding_dataset), args.results_path, "results.json"), "r") as fin :
    if args.embedding_dataset in ("MATH", "CollegeMath", ) + ("MMLU", ) + ("DS-1000", ) :
        results_type = "accuracy"
        RESULTS = np.array(json.load(fin))
    elif args.embedding_dataset in ("WildChat10K", "ShareGPT10K", "Chatbot-Arena", ) :
        results_type = "win-rate"
        RESULTS = np.array([((metrics[0] == 1) + (metrics[1] == 1)) / 2.0 for metrics in json.load(fin)])
    else :
        raise NotImplementedError("dataset = {}".format(args.dataset))

def load_the_split(split) :
    with open("Datasets/{}/splits/{}.json".format(args.embedding_dataset, split), "r") as fin :
        return json.load(fin)
if args.embedding_split == "full" :
    RANGE = np.arange(len(RESULTS))
    assert args.tree_dataset != args.embedding_dataset
else :
    if args.embedding_split.startswith("[exclusion]") :
        RANGE = np.array(sorted(list(set(range(len(RESULTS))) - set(load_the_split(args.embedding_split[len("[exclusion]") :])))))
    else :
        assert False
        RANGE = np.array(load_the_split(args.embedding_split))
assert set([int(instance) for instance in INSTANCE2PATH.keys()]) == set(RANGE)


Threshold = []
Instance_Number, Performance = [], []
delta = 0.0001
if args.direction == "lower" :
    threshold = TREE_RESULTS["confidence_interval"][str(args.alpha)][1]
    for subtree_results in TREE_RESULTS["subtrees"].values() if isinstance(TREE_RESULTS["subtrees"], dict) else TREE_RESULTS["subtrees"] :
        if subtree_results["size"] >= 20 :
            threshold = max(threshold, subtree_results["confidence_interval"][str(args.alpha)][1])
    threshold += delta
elif args.direction == "higher" :
    threshold = TREE_RESULTS["confidence_interval"][str(args.alpha)][0]
    for subtree_results in TREE_RESULTS["subtrees"].values() if isinstance(TREE_RESULTS["subtrees"], dict) else TREE_RESULTS["subtrees"] :
        if subtree_results["size"] >= 20 :
            threshold = min(threshold, subtree_results["confidence_interval"][str(args.alpha)][0])
    threshold -= delta
else :
    raise NotImplementedError("direction = {}".format(args.direction))

while True :
    print(threshold)
    extract_subtrees(TREE_RESULTS, args.alpha, threshold, args.direction)
    ALL_NODES = []
    REMOVE = []

    for instance, path in INSTANCE2PATH.items() :
        tree_results = TREE_RESULTS
        extracted = False
        for cluster in path :
            if extracted :
                assert not tree_results["extracted"]
            if tree_results["extracted"] :
                assert not extracted
                extracted = True
            tree_results = tree_results["subtrees"][str(cluster)]
        if extracted :
            ALL_NODES.append(int(instance))
        else :
            REMOVE.append(instance)
    
    for remove in REMOVE :
        INSTANCE2PATH.pop(remove)
    
    if len(ALL_NODES) == 0 :
        break
    assert set(ALL_NODES) <= set(RANGE.tolist())
    
    Threshold.append(threshold)
    Instance_Number.append(len(ALL_NODES))
    Performance.append(float(RESULTS[ALL_NODES].mean()))
        
    if args.direction == "lower" :
        threshold -= delta
    elif args.direction == "higher" :
        threshold += delta
    else :
        raise NotImplementedError("direction = {}".format(args.direction))


assert args.results_path.startswith("real/")
os.makedirs("EvalTree/WeaknessProfile/ExtractedNode_Analysis/results/{}->{}".format(args.tree_dataset, args.embedding_dataset), exist_ok = True)
with open("EvalTree/WeaknessProfile/ExtractedNode_Analysis/results/{}->{}/[direction={}]{}.json".format(args.tree_dataset, args.embedding_dataset, args.direction, args.results_path[len("real/") :]), "w") as fout :
    json.dump(dict(THRESHOLD = Threshold, PERFORMANCE = Performance, all_mean = float(RESULTS[RANGE].mean()), INSTANCE_NUMBER = Instance_Number), fout, indent = 2)