import os
import json
import argparse
from Baselines.QualEval.WeaknessProfile.performance_under_capabilities import get_capability2performance_split

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, required = True, choices = ("MATH", "WildChat10K", "DS-1000", ))
parser.add_argument("--results_path", type = str, required = True)

parser.add_argument("--chunk_size", type = int, default = 20)
parser.add_argument("--model", type = str, default = "gpt-4o-mini", choices = ("gpt-4o-mini", ))
parser.add_argument("--num_capabilities", type = int, default = 20)
parser.add_argument("--shrink_factor", type = int, default = 4)
parser.add_argument("--round", type = int, required = True)

parser.add_argument("--split", type = str, required = True)

parser.add_argument("--output_instances", type = int, default = None)
args = parser.parse_args()


with open("Datasets/{}/QualEval/stage1-CapabilityDiscovery/[chunk={}]_[model={}]/{}.json".format(args.dataset, args.chunk_size, args.model, "initialize" if args.round == 0 else "[num={}]_[factor={}]_[round={}]".format(args.num_capabilities, args.shrink_factor, args.round)), "r") as fin :
    capabilities = [capability for chunk in json.load(fin) for capability in chunk]

if args.dataset in ("MATH", "DS-1000", ) :
    results_type = "accuracy"
elif args.dataset in ("WildChat10K", ) :
    results_type = "win-rate"
else :
    raise NotImplementedError("dataset = {}".format(args.dataset))
with open(os.path.join("Datasets/{}/eval_results".format(args.dataset), args.results_path, "results.json"), "r") as fin :
    RESULTS = json.load(fin)

with open("Datasets/{}/QualEval/stage2-CapabilityAssignment/[chunk={}]_[model={}]_{}.json".format(args.dataset, args.chunk_size, args.model, "initialize" if args.round == 0 else "[num={}]_[factor={}]_[round={}]".format(args.num_capabilities, args.shrink_factor, args.round)), "r") as fin :
    assignments = json.load(fin)
    assert len(assignments) == len(RESULTS)

if args.split == "full" :
    RANGE = list(range(len(assignments)))
else :
    with open("Datasets/{}/splits/{}.json".format(args.dataset, args.split), "r") as fin :
        RANGE = json.load(fin)


capability2performance = get_capability2performance_split(
    capabilities = capabilities,
    assignments = assignments,
    results = RESULTS,
    results_type = results_type,
    split = RANGE,
)


capability2instances = {capability : [] for capability in capabilities}
for index in RANGE :
    assignment = assignments[index]
    for capability in assignment["assignment"] :
        capability = capabilities[int(capability) - 1]
        capability2instances[capability].append(index)


capability2performance = list(capability2performance.items())
capability2performance.sort(key = lambda x : x[1])


output_path = os.path.join("Datasets/{}/eval_results".format(args.dataset), args.results_path, "QualEval/[chunk={}]_[model={}]_{}_[direction={}]_[split={}]".format(args.chunk_size, args.model, "initialize" if args.round == 0 else "[num={}]_[factor={}]_[round={}]".format(args.num_capabilities, args.shrink_factor, args.round), "lower", args.split))
os.makedirs(output_path, exist_ok = True)
with open(os.path.join(output_path, "capability2performance.json"), "w") as fout :
    json.dump(capability2performance, fout, indent = 2)


profiles, profiles_with_instances, profiles_with_instances_statistics = [], [], []
for index, capability_performance in enumerate(capability2performance) :
    capability = capability_performance[0]
    profiles.append(capability_performance[0])
    profiles_with_instances.append(dict(capability = capability, instances = capability2instances[capability]))
    # profiles_with_instances_statistics.append(dict(capability = capability, instance_number = len(capability2instances[capability])))
    if args.output_instances is None or args.output_instances == index + 1 :
        with open(os.path.join(output_path, "weakness-profiles_[size={}].json".format(index + 1)), "w") as fout :
            json.dump(profiles, fout, indent = 2)
    if args.output_instances == index + 1 :
        with open(os.path.join(output_path, "weakness-profiles_[size={}]_[with-instances].json".format(index + 1)), "w") as fout :
            json.dump(profiles_with_instances, fout, indent = 2)