import os
import json
import torch
import random
import argparse
import concurrent.futures
from utils.common import manual_seed
from utils.api_inference import create_OpenAIclient, openai_completion, prompt_to_chatml

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, required = True, choices = ("MATH", "WildChat10K", "DS-1000", ))
parser.add_argument("--tree_path", type = str, required = True)

parser.add_argument("--description_model", type = str, default = "gpt-4o-mini", choices = ("gpt-4o-mini", ))
parser.add_argument("--num_procs", type = int, default = 512)
args = parser.parse_args()


TREE = torch.load(os.path.join("Datasets/{}/EvalTree".format(args.dataset), "{}.bin".format(args.tree_path)), weights_only = False)
with open("Datasets/{}/EvalTree/stage1-CapabilityAnnotation/[annotation={}].json".format(args.dataset, args.description_model), "r") as fin :
    CAPABILITIES = json.load(fin)

if args.dataset == "MATH" :
    PROMPT = "mathematics"
elif args.dataset in ("WildChat10K", "Chatbot-Arena", ) :
    PROMPT = "instruction-following"
elif args.dataset == "DS-1000" :
    PROMPT = "ds-1000"
elif args.dataset == "MMLU" :
    PROMPT = "mmlu"
else :
    raise NotImplementedError("dataset = {}".format(args.dataset))
with open("EvalTree/stage4-CapabilityDescription/prompts/{}.txt".format(PROMPT), "r") as fin :
    PROMPT = fin.read()


def initialize_description(tree) :
    tree_description = {
        "description" : None,
    }
    if isinstance(tree, dict) :
        assert "subtrees" in tree
        if isinstance(tree["subtrees"], list) :
            tree_description["subtrees"] = [initialize_description(subtree) for subtree in tree["subtrees"]]
        elif isinstance(tree["subtrees"], dict) :
            tree_description["subtrees"] = {key : initialize_description(subtree) for key, subtree in tree["subtrees"].items()}
        else :
            raise NotImplementedError
    else :
        assert isinstance(tree, int)
        tree_description["subtrees"] = tree
    return tree_description


EXECUTORS = {}
OPENAI_KWARGS = {
    "model" : args.description_model,
    "max_tokens" : 1024,
    "temperature" : 0.0,
    "seed" : 0,
}
def describe(tree_description, depth) :
    cost = 0.0
    if not isinstance(tree_description["subtrees"], int) :
        assert isinstance(tree_description["subtrees"], list) or isinstance(tree_description["subtrees"], dict)

        if depth not in EXECUTORS :
            EXECUTORS[depth] = concurrent.futures.ThreadPoolExecutor(max_workers = args.num_procs)
        executor = EXECUTORS[depth]
        cost += sum(list(executor.map(lambda subtree: describe(subtree, depth + 1), (tree_description["subtrees"] if isinstance(tree_description["subtrees"], list) else tree_description["subtrees"].values()))))

        skills = [subtree["description"] for subtree in (tree_description["subtrees"] if isinstance(tree_description["subtrees"], list) else tree_description["subtrees"].values())]
        manual_seed(42)
        random.shuffle(skills)
        skills = ["### Skill #{}\n{}\n".format(index + 1, skill) for index, skill in enumerate(skills)]
        
        chatml = prompt_to_chatml(prompt = PROMPT.format_map(dict(group_number = len(skills), skill_descriptions = "\n".join(skills))))
        client = create_OpenAIclient(dict(api_key = os.getenv("OpenAI_API_KEY")))
        response = openai_completion(client, chatml, OPENAI_KWARGS)
        tree_description["description"] = response["response"].strip()
        cost += response["cost"]
        print(tree_description["description"])
    else :
        tree_description["description"] = CAPABILITIES[tree_description["subtrees"]]
    return cost


try :
    TREE_DESCRIPTION = initialize_description(TREE)
    print("cost = {}".format(describe(TREE_DESCRIPTION, 0)))
finally :
    for executor in EXECUTORS.values() :
        executor.shutdown(wait = True)


with open(os.path.join("Datasets/{}/EvalTree".format(args.dataset), "{}_[stage4-CapabilityDescription-model={}].json".format(args.tree_path, args.description_model)), "w") as fout :
    json.dump(TREE_DESCRIPTION, fout, indent = 2)