import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, required = True, choices = ("MATH", "WildChat10K", "DS-1000", ))
parser.add_argument("--results_path", type = str, required = True)
parser.add_argument("--path", type = str, default = "[negative_instance=50]_[positive_instance=50]_[maximum=20]_[seed=0]")
parser.add_argument("--split", type = str, required = True)
parser.add_argument("--output_instances", type = int, default = None)
args = parser.parse_args()


if args.dataset in ("MATH", "DS-1000", ) :
    results_type = "accuracy"
elif args.dataset == "WildChat10K" :
    results_type = "win-rate"
else :
    raise NotImplementedError("dataset = {}".format(args.dataset))
with open(os.path.join("Datasets/{}/eval_results".format(args.dataset), args.results_path, "results.json"), "r") as fin :
    RESULTS = json.load(fin)

output_path = os.path.join("Datasets/{}/eval_results".format(args.dataset), args.results_path, "TextDiff", args.path)
with open(os.path.join(output_path, "weakness-profile.json"), "r") as fin :
    weakness_profile = json.load(fin)

def load_the_split(split) :
    with open("Datasets/{}/splits/{}.json".format(args.dataset, split), "r") as fin :
        return json.load(fin)
if args.split == "full" :
    RANGE = list(range(len(RESULTS)))
elif args.split.startswith("[exclusion]") :
    assert False
    RANGE = sorted(list(set(range(len(RESULTS))) - set(load_the_split(args.split[len("[exclusion]") :]))))
else :
    RANGE = load_the_split(args.split)
def get_RANGE(LIST : list) :
    return [LIST[index] for index in RANGE]

with open("Datasets/{}/AssociatedInstances_[gpt-4o-mini].json".format(args.dataset), "r") as fin :
    cache = json.load(fin)


capability2performance = {}
for capability in weakness_profile :
    total, success = 0, 0
    for requirement, result in zip(get_RANGE(cache[capability]), get_RANGE(RESULTS)) :
        if requirement != "YES" :
            continue
        if results_type == "accuracy" :
            assert result in (0, 1)
            total += 1
            success += result
        elif results_type == "win-rate" :
            assert isinstance(result, list) and len(result) == 2
            assert result[0] in (1, 2) and result[1] in (1, 2)
            total += 2
            success += (result[0] == 1) + (result[1] == 1)
        else :
            raise NotImplementedError("results_type = {}".format(results_type))
    capability2performance[capability] = success / total if total else 1.0


weakness_profile = sorted(weakness_profile, key = lambda capability : capability2performance[capability])
profiles_with_instances = []
for index in range(1, len(weakness_profile) + 1) :
    capability = weakness_profile[index - 1]
    instances = []
    for i in RANGE :
        if cache[capability][i] == "YES" :
            instances.append(i)
    profiles_with_instances.append(dict(capability = capability, instances = instances))
    
    if args.output_instances is None or args.output_instances == index :
        with open(os.path.join(output_path, "[split={}]weakness-profiles_[size={}].json".format(args.split, index)), "w") as fout :
            json.dump(weakness_profile[: index], fout, indent = 2)
    if args.output_instances == index :
        with open(os.path.join(output_path, "[split={}]weakness-profiles_[size={}]_[with-instances].json".format(args.split, index)), "w") as fout :
            json.dump(profiles_with_instances, fout, indent = 2)