import os
import json
import argparse
from EvalTree.WeaknessProfile.extract_subtrees import extract_subtrees


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, required = True, choices = ("MATH", "WildChat10K", ))
parser.add_argument("--tree_path", type = str, required = True)
parser.add_argument("--results_path", type = str, required = True)
parser.add_argument("--description_model", type = str, default = "gpt-4o-mini", choices = ("gpt-4o-mini", ))
parser.add_argument("--direction", type = str, default = "lower", choices = ("higher", "lower"))
parser.add_argument("--alpha", type = float, default = 0.05)
parser.add_argument("--max_profile_size", type = int, default = None)
args = parser.parse_args()


with open(os.path.join("Datasets/{}/EvalTree".format(args.dataset), "{}_[stage4-CapabilityDescription-model={}].json".format(args.tree_path, args.description_model)), "r") as fin :
    TREE_DESCRIPTION = json.load(fin)

output_path = args.tree_path.split("/")
assert len(output_path) == 2
output_path = "EvalTree/TREE=[{}]_{}".format(output_path[0], output_path[1])
with open(os.path.join("Datasets/{}/eval_results".format(args.dataset), args.results_path, output_path, "confidence_interval.json"), "r") as fin :
    TREE_RESULTS = json.load(fin)


delta = 0.0001
if args.direction == "lower" :
    threshold = TREE_RESULTS["confidence_interval"][str(args.alpha)][1]
    for subtree_results in TREE_RESULTS["subtrees"].values() if isinstance(TREE_RESULTS["subtrees"], dict) else TREE_RESULTS["subtrees"] :
        if subtree_results["size"] >= 20 :
            threshold = max(threshold, subtree_results["confidence_interval"][str(args.alpha)][1])
    threshold += delta
else :
    raise NotImplementedError("direction = {}".format(args.direction))


if args.max_profile_size is None :
    # We keep all possible weakness profiles.
    all_profiles = set()
    SEPARATOR = "[###***###]"
else :
    # We keep one weakness profile for each size in [1, args.max_profile_size].
    size2profile = {}


while True :
    extract_subtrees(TREE_RESULTS, args.alpha, threshold, args.direction)

    weakness_profile = []
    def traverse_tree(tree_results, tree_description, extracted : bool) :
        if extracted :
            assert not tree_results["extracted"]
        if tree_results["extracted"] :
            assert not extracted
            weakness_profile.append(tree_description["description"])
            extracted = True
        
        if isinstance(tree_results["subtrees"], int) :
            assert isinstance(tree_description["subtrees"], int)
            assert tree_results["subtrees"] == tree_description["subtrees"]
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
    traverse_tree(TREE_RESULTS, TREE_DESCRIPTION, False)

    if len(weakness_profile) == 0 :
        break

    if args.max_profile_size is None :
        weakness_profile = sorted(weakness_profile)
        weakness_profile = SEPARATOR.join(weakness_profile)
        all_profiles.add(weakness_profile)
    else :
        if len(weakness_profile) <= args.max_profile_size :
            size2profile[len(weakness_profile)] = weakness_profile.copy()
    
    if args.direction == "lower" :
        threshold -= delta
    else :
        raise NotImplementedError("direction = {}".format(args.direction))


os.makedirs(os.path.join("Datasets/{}/eval_results".format(args.dataset), args.results_path, output_path, "weakness-profiles_varying_threshold"), exist_ok = True)
if args.max_profile_size is None :
    all_profiles = sorted(list(all_profiles))
    for index, profile in enumerate(all_profiles) :
        with open(os.path.join("Datasets/{}/eval_results".format(args.dataset), args.results_path, output_path, "weakness-profiles_varying_threshold/[description={}]_[direction={}]_[alpha={}]_[index={}].json".format(args.description_model, args.direction, args.alpha, index)), "w") as fout :
            json.dump(profile.split(SEPARATOR), fout, indent = 2)
else :
    for size in range(1, args.max_profile_size + 1) :
        if size not in size2profile :
            continue
        with open(os.path.join("Datasets/{}/eval_results".format(args.dataset), args.results_path, output_path, "weakness-profiles_varying_threshold/[description={}]_[direction={}]_[alpha={}]_[size={}].json".format(args.description_model, args.direction, args.alpha, size)), "w") as fout :
            json.dump(size2profile[size], fout, indent = 2)