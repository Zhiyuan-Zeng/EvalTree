import json
import argparse
import numpy as np
from scipy.optimize import linprog

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, required = True, choices = ("MATH", "WildChat10K", "DS-1000", ))
parser.add_argument("--model", type = str, default = "gpt-4o-mini", choices = ("gpt-4o-mini", ))
parser.add_argument("--chunk_size", type = int, default = 20)
parser.add_argument("--num_capabilities", type = int, default = 20)
parser.add_argument("--shrink_factor", type = int, default = 4)
parser.add_argument("--round", type = int, required = True)
args = parser.parse_args()


with open("Datasets/{}/QualEval/stage1-CapabilityDiscovery/[chunk={}]_[model={}]/{}.json".format(args.dataset, args.chunk_size, args.model, "initialize" if args.round == 0 else "[num={}]_[factor={}]_[round={}]".format(args.num_capabilities, args.shrink_factor, args.round)), "r") as fin :
    num_capabilities = sum([len(chunk) for chunk in json.load(fin)])
with open("Datasets/{}/QualEval/stage2-CapabilityAssignment/[chunk={}]_[model={}]_{}.json".format(args.dataset, args.chunk_size, args.model, "initialize" if args.round == 0 else "[num={}]_[factor={}]_[round={}]".format(args.num_capabilities, args.shrink_factor, args.round)), "r") as fin :
    scores = json.load(fin)
prior_probabilities = {str(capability) : 0 for capability in range(1, num_capabilities + 1)}


capabiltiy_gt_scores_np = []
for score in scores :
    if isinstance(score["scoring"], str) :
        score["scoring"] = {str(capability) : dict(score = 1) for capability in range(1, num_capabilities + 1)}
    for capability in range(1, num_capabilities + 1) :
        s = score["scoring"].get(str(capability), dict(score = 1))
        s = 1 if not isinstance(s, dict) else s.get("score", 1)
        prior_probabilities[str(capability)] += s
        capabiltiy_gt_scores_np.append(s)
assert len(capabiltiy_gt_scores_np) == len(scores) * num_capabilities
sum_prior_probabilities = sum(prior_probabilities.values())


A = np.zeros(
    (
        len(scores) + num_capabilities + num_capabilities,
        len(scores) * num_capabilities,
    )
)
b = np.zeros(len(scores) + num_capabilities + num_capabilities)
# constraint 1
for i in range(len(scores)) :
    A[i, i * num_capabilities : (i + 1) * num_capabilities] = 1
    b[i] = 2

# constraint 2 -- upper bound
for j in range(num_capabilities) :
    A[len(scores) + j, j :: num_capabilities] = 1
    b[len(scores) + j] = 2 * len(scores) * (prior_probabilities[str(j + 1)] / sum_prior_probabilities) * (1 + 0.1)

# constraint 2 -- lower bound
for j in range(num_capabilities):
    A[len(scores) + num_capabilities + j, j :: num_capabilities] = -1
    b[len(scores) + num_capabilities + j] = -2 * len(scores) * (prior_probabilities[str(j + 1)] / sum_prior_probabilities) * (1 - 0.1)

# solve the linear program
res = linprog(-np.array(capabiltiy_gt_scores_np), A_ub = A, b_ub = b, bounds = (0, 1), integrality = 1)
reshaped_assignments = res.x.reshape(len(scores), num_capabilities)
assert np.all(np.logical_or(reshaped_assignments == 0, reshaped_assignments == 1))
for instance, assignment in zip(scores, reshaped_assignments) :
    instance["assignment"] = [str(index + 1) for index, assigned in enumerate(assignment) if assigned]
    assert len(instance["assignment"]) == 2


with open("Datasets/{}/QualEval/stage2-CapabilityAssignment/[chunk={}]_[model={}]_{}.json".format(args.dataset, args.chunk_size, args.model, "initialize" if args.round == 0 else "[num={}]_[factor={}]_[round={}]".format(args.num_capabilities, args.shrink_factor, args.round)), "w") as fout :
    json.dump(scores, fout, indent = 2)