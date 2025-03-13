import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, required = True, choices = ("MATH", "WildChat10K", ))
parser.add_argument("--synthetic", type = str, default = "synthetic")
parser.add_argument("--results_path", type = str, required = True)
parser.add_argument("--method", type = str, required = True, choices = ("TextDiff", "QualEval", "EvalTree", ))
parser.add_argument("--predictor", type = str, required = True)
parser.add_argument("--size", type = int, required = True)
args = parser.parse_args()

with open("Datasets/{}/eval_results/{}/ground-truth.json".format(args.dataset, args.synthetic), "r") as fin :
    groundtruths = json.load(fin)
with open(os.path.join("Datasets/{}/eval_results/{}".format(args.dataset, args.synthetic), args.results_path, args.method, args.predictor.replace(r"{PLACEHOLDER}", str(args.size)) + ".json"), "r") as fin :
    predictions = json.load(fin)

with open("Datasets/{}/AssociatedInstances_[gpt-4o-mini].json".format(args.dataset), "r") as fin :
    cache = json.load(fin)


def get_set(capability : str) :
    return set([index for index, label in enumerate(cache[capability]) if label == "YES"])
def get_union(sets) :
    union = set()
    for s in sets :
        union |= s
    return union
groundtruths = {capability : get_set(capability) for capability in groundtruths}
predictions = {capability : get_set(capability) for capability in predictions}
GROUNDTRUTH_set, PREDICTION_set = get_union(groundtruths.values()), get_union(predictions.values())


evaluation_results = {
    "Recall" : {"per-capability" : {}, "average" : None},
    "Precision" : {"per-capability" : {}, "average" : None},
    "harmonic mean (F1)" : None,
}

# How many instances in A are also covered by B?
def calculate_intersection_ratio(a : set, b : set) :
    a, b = len(a & b), len(a)
    return "{} / {} = {:.5f}".format(a, b, a / b if b else 0.0), a / b if b else 0.0

Sum = 0.0
for groundtruth, groundtruth_set in groundtruths.items() :
    evaluation_results["Recall"]["per-capability"][groundtruth], ratio = calculate_intersection_ratio(groundtruth_set, PREDICTION_set)
    Sum += ratio
evaluation_results["Recall"]["average"] = Sum / len(groundtruths)

Sum = 0.0
for prediction, prediction_set in predictions.items() :
    evaluation_results["Precision"]["per-capability"][prediction], ratio = calculate_intersection_ratio(prediction_set, GROUNDTRUTH_set)
    Sum += ratio
evaluation_results["Precision"]["average"] = Sum / len(predictions)

def calculate_F1(Recall, Precision) :
    return 2 * Recall * Precision / (Recall + Precision) if Recall + Precision > 0 else 0.0
evaluation_results["harmonic mean (F1)"] = calculate_F1(evaluation_results["Recall"]["average"], evaluation_results["Precision"]["average"])


output_path = os.path.join("Assessments/Synthetic/results/{}".format(args.dataset), args.results_path)
os.makedirs(output_path, exist_ok = True)
with open(os.path.join("Assessments/Synthetic/results/{}".format(args.dataset), args.results_path, "[method={}][size={}].json".format(args.method, args.size)), "w") as fout :
    json.dump(evaluation_results, fout, indent = 2)