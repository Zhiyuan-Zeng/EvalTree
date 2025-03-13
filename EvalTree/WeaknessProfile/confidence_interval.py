import os
import json
import torch
import argparse
import statsmodels.api as sm

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, required = True, choices = ("MATH", "WildChat10K", "DS-1000", "MMLU", ))
parser.add_argument("--tree_path", type = str, required = True)
parser.add_argument("--results_path", type = str, required = True)
args = parser.parse_args()


TREE = torch.load(os.path.join("Datasets/{}/EvalTree".format(args.dataset), "{}.bin".format(args.tree_path)), weights_only = False)
with open(os.path.join("Datasets/{}/eval_results".format(args.dataset), args.results_path, "results.json"), "r") as fin :
    RESULTS = json.load(fin)

if args.dataset in ("MATH", "DS-1000", "MMLU", ) :
    results_type = "accuracy"
elif args.dataset == "WildChat10K" :
    results_type = "win-rate"
else :
    raise NotImplementedError("dataset = {}".format(args.dataset))


def calculate(tree) :
    if not isinstance(tree, int) :
        tree_results =  {
            "size" : 0,
            "sum_metrics" : 0,
        }
        if isinstance(tree["subtrees"], list) :
            assert ("kmeans" not in tree) or (tree["kmeans"] is None)

            tree_results["subtrees"] = []
            for subtree in tree["subtrees"] :
                subtree_results = calculate(subtree)
                tree_results["subtrees"].append(subtree_results)
                tree_results["size"] += subtree_results["size"]
                tree_results["sum_metrics"] += subtree_results["sum_metrics"]
        else :
            assert isinstance(tree["subtrees"], dict)
            assert ("kmeans" in tree) and (tree["kmeans"] is not None)

            tree_results["subtrees"] = {}
            for cluster, subtree in tree["subtrees"].items() :
                subtree_results = calculate(subtree)
                tree_results["subtrees"][cluster] = subtree_results
                tree_results["size"] += subtree_results["size"]
                tree_results["sum_metrics"] += subtree_results["sum_metrics"]
    else :
        metrics = RESULTS[tree]
        if results_type == "accuracy" :
            assert metrics in (0, 1)
            tree_results = {
                "size" : 1,
                "sum_metrics" : metrics,
            }
        elif results_type == "win-rate" :
            assert isinstance(metrics, list) and len(metrics) == 2
            assert metrics[0] in (1, 2) and metrics[1] in (1, 2)
            tree_results = {
                "size" : 1,
                "sum_metrics" : int(metrics[0] == 1) + int(metrics[1] == 1),
            }
        else :
            raise NotImplementedError("results_type = {}".format(results_type))
        tree_results["subtrees"] = tree
    if tree_results["size"] < 5 :
        tree_results["confidence_interval"] = None
    else :
        tree_results["confidence_interval"] = {}
        for alpha in (0.01, 0.05) :
            if results_type == "accuracy" :
                lower_bound, upper_bound = sm.stats.proportion_confint(tree_results["sum_metrics"], tree_results["size"], alpha = alpha, method = "beta")
            elif results_type == "win-rate" :
                lower_bound, upper_bound = sm.stats.proportion_confint(tree_results["sum_metrics"], tree_results["size"] * 2, alpha = alpha, method = "beta")
            else :
                raise NotImplementedError("results_type = {}".format(results_type))
            tree_results["confidence_interval"][alpha] = (lower_bound, upper_bound)
    return tree_results


TREE_RESULTS = calculate(TREE)
output_path = args.tree_path.split("/")
assert len(output_path) == 2
output_path = "EvalTree/TREE=[{}]_{}".format(output_path[0], output_path[1])
os.makedirs(os.path.join("Datasets/{}/eval_results".format(args.dataset), args.results_path, output_path), exist_ok = True)
with open(os.path.join("Datasets/{}/eval_results".format(args.dataset), args.results_path, output_path, "confidence_interval.json"), "w") as fout :
    json.dump(TREE_RESULTS, fout, indent = 2)