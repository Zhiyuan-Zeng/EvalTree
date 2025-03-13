import os
import json
import torch
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--tree_dataset", type = str, required = True, choices = ("MATH", "MMLU", "DS-1000", "WildChat10K", ))
parser.add_argument("--tree_path", type = str, required = True)

parser.add_argument("--embedding_dataset", type = str, required = True, choices = ("MATH", "CollegeMath", ) + ("MMLU", ) + ("DS-1000", ) + ("WildChat10K", "ShareGPT10K", "Chatbot-Arena", ))
parser.add_argument("--embedding_path", type = str, choices = ("stage2-CapabilityEmbedding/[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]", ), default = "stage2-CapabilityEmbedding/[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]")
parser.add_argument("--embedding_split", type = str, required = True)
args = parser.parse_args()


PATH = []
instance2path = {}
def locate(tree, instances : np.ndarray) :
    if isinstance(tree, int) or tree["kmeans"] is None :
        for instance in instances :
            instance2path[instance.item()] = PATH.copy()
        return
    matrix = MATRIX[instances]
    clusters = tree["kmeans"].predict(matrix)
    for cluster in tree["subtrees"].keys() :
        subtree_instances = instances[clusters == cluster]
        if len(subtree_instances) :
            PATH.append(cluster)
            locate(tree["subtrees"][cluster], subtree_instances)
            PATH.pop()


TREE = torch.load(os.path.join("Datasets/{}/EvalTree".format(args.tree_dataset), "{}.bin".format(args.tree_path)), weights_only = False)
MATRIX = torch.stack(torch.load("Datasets/{}/EvalTree/{}.bin".format(args.embedding_dataset, args.embedding_path), weights_only = True)).numpy()
def load_the_split(split) :
    with open("Datasets/{}/splits/{}.json".format(args.embedding_dataset, split), "r") as fin :
        return json.load(fin)
if args.embedding_split == "full" :
    RANGE = np.arange(MATRIX.shape[0])
else :
    if args.embedding_split.startswith("[exclusion]") :
        RANGE = np.array(list(set(range(len(MATRIX))) - set(load_the_split(args.embedding_split[len("[exclusion]") :]))))
    else :
        assert False
        RANGE = np.array(load_the_split(args.embedding_split))
locate(TREE, np.array(RANGE))


os.makedirs("Datasets/{}/EvalTree/{}_[located-split={}]".format(args.embedding_dataset, args.embedding_path, args.embedding_split), exist_ok = True)
output_path = args.tree_path.split("/")
assert len(output_path) == 2
output_path = "[dataset={}]_[{}]_{}".format(args.tree_dataset, output_path[0], output_path[1])
with open(os.path.join("Datasets/{}/EvalTree/{}_[located-split={}]".format(args.embedding_dataset, args.embedding_path, args.embedding_split), "{}.json".format(output_path)), "w") as fout :
    json.dump(instance2path, fout, indent = 2)