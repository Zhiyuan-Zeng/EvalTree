# python -m EvalTree.Chatbot-Arena_ranking
# python -m EvalTree.Chatbot-Arena_ranking --dataset Chatbot-Arena_NEW
import os
import json
import torch
import argparse
import pandas as pd
from utils.compute_elo import compute_mle_elo, preety_print_model_ratings

import warnings
warnings.filterwarnings("error", category = RuntimeWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, default = "Chatbot-Arena", choices = ("Chatbot-Arena", "Chatbot-Arena_NEW", ))
parser.add_argument("--tree_path", type = str, default = "stage3-RecursiveClustering/[split=full]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10]")
parser.add_argument("--results_path", type = str, default = "battleground")
args = parser.parse_args()


TREE = torch.load(os.path.join("Datasets/{}/EvalTree".format(args.dataset), "{}.bin".format(args.tree_path)), weights_only = False)
with open(os.path.join("Datasets/{}/eval_results".format(args.dataset), args.results_path, "results.json"), "r") as fin :
    RESULTS = json.load(fin)


def ranking(tree) :
    battles = []
    tree_results =  {}
    if not isinstance(tree, int) :
        if isinstance(tree["subtrees"], list) :
            assert ("kmeans" not in tree) or (tree["kmeans"] is None)

            tree_results["subtrees"] = []
            for subtree in tree["subtrees"] :
                subtree_results, subtree_battles = ranking(subtree)
                tree_results["subtrees"].append(subtree_results)
                battles += subtree_battles
        else :
            assert isinstance(tree["subtrees"], dict)
            assert ("kmeans" in tree) and (tree["kmeans"] is not None)

            tree_results["subtrees"] = {}
            for cluster, subtree in tree["subtrees"].items() :
                subtree_results, subtree_battles = ranking(subtree)
                tree_results["subtrees"][cluster] = subtree_results
                battles += subtree_battles
    else :
        tree_results["subtrees"] = tree
        battles = RESULTS[tree]
    
    try :
        result = preety_print_model_ratings(compute_mle_elo(pd.DataFrame(battles)))
        tree_results["ranking"] = list(zip(result["Model"].tolist(), result["Elo rating"].tolist()))
    except :
        tree_results["ranking"] = None
    
    return tree_results, battles


TREE_RESULTS, battles = ranking(TREE)
with open(os.path.join("Datasets/{}/eval_results".format(args.dataset), args.results_path, "EvalTree_ranking.json"), "w") as fout :
    json.dump(TREE_RESULTS, fout, indent = 2)