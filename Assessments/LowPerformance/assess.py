import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, required = True, choices = ("MATH", "WildChat10K", ))
parser.add_argument("--results_path", type = str, required = True)
parser.add_argument("--method", type = str, required = True, choices = ("TextDiff", "QualEval", "EvalTree", ))
parser.add_argument("--predictor", type = str, required = True)
parser.add_argument("--max_index", type = int, default = 120)
parser.add_argument("--max_profile_size", type = int, default = 20)
parser.add_argument("--split", type = str, required = True)
args = parser.parse_args()


if args.dataset == "MATH" :
    results_type = "accuracy"
    assert args.split == "[exclusion]4k-1k"
elif args.dataset == "WildChat10K" :
    results_type = "win-rate"
    assert args.split == "[exclusion]8k-2k"
else :
    raise NotImplementedError("dataset = {}".format(args.target_dataset))

with open(os.path.join("Datasets/{}/eval_results".format(args.dataset), args.results_path, "results.json"), "r") as fin :
    RESULTS = json.load(fin)

def load_the_split(split) :
    with open("Datasets/{}/splits/{}.json".format(args.dataset, split), "r") as fin :
        return json.load(fin)
if args.split == "full" :
    assert False
    RANGE = list(range(len(RESULTS)))
else :
    if args.split.startswith("[exclusion]") :
        RANGE = sorted(list(set(range(len(RESULTS))) - set(load_the_split(args.split[len("[exclusion]") :]))))
    else :
        assert False

with open("Datasets/{}/AssociatedInstances_[gpt-4o-mini].json".format(args.dataset), "r") as fin :
    cache = json.load(fin)


def get_performance(index_set : set) :
    total, success = 0, 0
    for index in index_set :
        result = RESULTS[index]
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
    return success / total if total else 1.0


size2val1, num2val2 = {}, {}
for index in range(1, args.max_index + 1) :
    path = os.path.join("Datasets/{}/eval_results".format(args.dataset), args.results_path, args.method, args.predictor.replace(r"{PLACEHOLDER}", str(index)) + ".json")
    try :
        with open(path, "r") as fin :
            weakness_profile = json.load(fin)
    except FileNotFoundError :
        continue
    except :
        assert False

    Sum_performance = 0.0
    all_index_set = set()
    for weakness in weakness_profile :
        for index in RANGE :
            assert isinstance(cache[weakness][index], str)
        index_set = set([index for index in RANGE if cache[weakness][index] == "YES"])
        all_index_set |= index_set
        Sum_performance += get_performance(index_set)
    
    if len(weakness_profile) <= args.max_profile_size :
        val1 = Sum_performance / len(weakness_profile)
        if len(weakness_profile) not in size2val1 :
            size2val1[len(weakness_profile)] = val1
        else :
            size2val1[len(weakness_profile)] = min(size2val1[len(weakness_profile)], val1)

    val2 = get_performance(all_index_set)
    if len(all_index_set) not in num2val2 :
        num2val2[len(all_index_set)] = val2
    else :
        num2val2[len(all_index_set)] = min(num2val2[len(all_index_set)], val2)


output_path = os.path.join("Assessments/LowPerformance/results/{}".format(args.dataset), args.results_path)
os.makedirs(output_path, exist_ok = True)
with open(os.path.join(output_path, "[split={}][method={}]size2val1.json".format(args.split, args.method)), "w") as fout :
    size, val1 = zip(*sorted(size2val1.items()))
    json.dump(dict(size = size, val1 = val1), fout, indent = 2)
with open(os.path.join(output_path, "[split={}][method={}]num2val2.json".format(args.split, args.method)), "w") as fout :
    num, val2 = zip(*sorted(num2val2.items()))
    json.dump(dict(num = num, val2 = val2), fout, indent = 2)