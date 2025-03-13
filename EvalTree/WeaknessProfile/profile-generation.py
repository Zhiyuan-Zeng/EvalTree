import os
import json
import argparse
from EvalTree.WeaknessProfile.extract_subtrees import extract_subtrees

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, required = True, choices = ("MATH", "DS-1000", ))
parser.add_argument("--tree_path", type = str, required = True)
parser.add_argument("--results_path", type = str, required = True)
parser.add_argument("--description_model", type = str, default = "gpt-4o-mini", choices = ("gpt-4o-mini", ))
parser.add_argument("--direction", type = str, default = "lower", choices = ("higher", "lower"))
parser.add_argument("--alpha", type = float, default = 0.05)
parser.add_argument("--threshold", type = float, required = True)
args = parser.parse_args()


with open(os.path.join("Datasets/{}/EvalTree".format(args.dataset), "{}_[stage4-CapabilityDescription-model={}].json".format(args.tree_path, args.description_model)), "r") as fin :
    TREE_DESCRIPTION = json.load(fin)

output_path = args.tree_path.split("/")
assert len(output_path) == 2
output_path = "EvalTree/TREE=[{}]_{}".format(output_path[0], output_path[1])
with open(os.path.join("Datasets/{}/eval_results".format(args.dataset), args.results_path, output_path, "confidence_interval.json"), "r") as fin :
    TREE_RESULTS = json.load(fin)


extract_subtrees(TREE_RESULTS, args.alpha, args.threshold, args.direction)


Total = -1
OUTPUTS = []
def traverse_tree(tree_results, tree_description, extracted : int) :
    global Total
    if extracted != -1 :
        assert not tree_results["extracted"]
    if tree_results["extracted"] :
        assert extracted == -1
        Total += 1
        extracted = Total
        assert Total == len(OUTPUTS)
        OUTPUTS.append({"capability" : tree_description["description"], "instances" : []})
    
    if isinstance(tree_results["subtrees"], int) :
        assert isinstance(tree_description["subtrees"], int)
        assert tree_results["subtrees"] == tree_description["subtrees"]
        if extracted != -1 :
            OUTPUTS[extracted]["instances"].append(tree_results["subtrees"])
    else :
        if isinstance(tree_results["subtrees"], list) :
            assert isinstance(tree_description["subtrees"], list)
            assert len(tree_results["subtrees"]) == len(tree_description["subtrees"])
            for subtree_results, subtree_description in zip(tree_results["subtrees"], tree_description["subtrees"]) :
                traverse_tree(subtree_results, subtree_description, extracted)
        elif isinstance(tree_results["subtrees"], dict) :
            assert isinstance(tree_description["subtrees"], dict)
            assert len(tree_results["subtrees"]) == len(tree_description["subtrees"])
            for subtree in tree_results["subtrees"].keys() :
                traverse_tree(tree_results["subtrees"][subtree], tree_description["subtrees"][subtree], extracted)
        else :
            assert False
traverse_tree(TREE_RESULTS, TREE_DESCRIPTION, -1)


os.makedirs(os.path.join("Datasets/{}/eval_results".format(args.dataset), args.results_path, output_path, "weakness-profiles"), exist_ok = True)
with open(os.path.join("Datasets/{}/eval_results".format(args.dataset), args.results_path, output_path, "weakness-profiles/[description={}]_[direction={}]_[alpha={}]_[threshold={}]_[with-instances].json".format(args.description_model, args.direction, args.alpha, args.threshold)), "w") as fout :
    json.dump(OUTPUTS, fout, indent = 2)