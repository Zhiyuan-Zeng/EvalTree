import os
import json
import datasets
import argparse
import functools
import multiprocessing
from tqdm import tqdm
from utils.api_inference import create_OpenAIclient, openai_completion, prompt_to_chatml

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, required = True, choices = ("MATH", "WildChat10K", "DS-1000", ))
parser.add_argument("--capability_path", type = str, required = True)
parser.add_argument("--num_procs", type = int, default = 512)
parser.add_argument("--annotation_model", type = str, default = "gpt-4o-mini", choices = ("gpt-4o-mini", ))
parser.add_argument("--split", type = str, required = True)
args = parser.parse_args()

if args.dataset == "MATH" :
    PROMPT = "mathematics"
    INPUT_KEY, OUTPUT_KEY = "problem", "solution"
    dataset = datasets.load_dataset("lighteval/MATH")["test"].to_list()
elif args.dataset == "WildChat10K" :
    PROMPT = "instruction-following"
    INPUT_KEY, OUTPUT_KEY = "instruction", "response"
    with open("Datasets/{}/dataset.json".format(args.dataset), "r") as fin :
        dataset = json.load(fin)
elif args.dataset == "DS-1000" :
    PROMPT = "ds-1000"
    INPUT_KEY, OUTPUT_KEY = "prompt", "reference_code"
    dataset = datasets.load_dataset("xlangai/DS-1000")["test"].to_list()
else :
    raise NotImplementedError("dataset = {}".format(args.dataset))
with open("AssociatedInstances/prompts/{}.txt".format(PROMPT), "r") as fin :
    PROMPT = fin.read()

with open(args.capability_path, "r") as fin :
    capabilities = json.load(fin)

def load_the_split(split) :
    with open("Datasets/{}/splits/{}.json".format(args.dataset, split), "r") as fin :
        return json.load(fin)
if args.split == "full" :
    RANGE = list(range(len(dataset)))
elif args.split.startswith("[exclusion]") :
    RANGE = sorted(list(set(range(len(dataset))) - set(load_the_split(args.split[len("[exclusion]") :]))))
else :
    RANGE = load_the_split(args.split)


OPENAI_KWARGS = {
    "model" : args.annotation_model,
    "max_tokens" : 128,
    "temperature" : 0.0,
    "seed" : 0,
}
def Process(instance, capability) :
    chatml = prompt_to_chatml(PROMPT.format_map(dict(input = instance[INPUT_KEY], output = instance[OUTPUT_KEY], capability = capability)))

    client = create_OpenAIclient(dict(api_key = os.getenv("OpenAI_API_KEY")))
    return openai_completion(client, chatml, OPENAI_KWARGS)


COST = 0.0
for capability in capabilities :
    try :
        with open("Datasets/{}/AssociatedInstances_[{}].json".format(args.dataset, args.annotation_model), "r") as fin :
            cache = json.load(fin)
    except FileNotFoundError :
        cache = {}
    except :
        assert False
    
    annotations = cache.get(capability, [None] * len(dataset))
    to_be_annotated = []
    for index in RANGE :
        if annotations[index] not in ("YES", "NO", ) :
            to_be_annotated.append(dataset[index])
    
    with multiprocessing.Pool(args.num_procs) as p :
        _Process = functools.partial(Process, capability = capability)
        outputs = list(
            tqdm(
                p.imap(_Process, to_be_annotated),
                desc = "to_be_annotated",
                total = len(to_be_annotated),
            )
        )
    COST += sum([output["cost"] for output in outputs])

    output_index = 0
    for index in RANGE :
        if annotations[index] not in ("YES", "NO", ) :
            annotations[index] = outputs[output_index]["response"]
            output_index += 1
    cache[capability] = annotations

    assert len(outputs) == output_index
    assert len(to_be_annotated) == output_index

    with open("Datasets/{}/AssociatedInstances_[{}].json".format(args.dataset, args.annotation_model), "w") as fout :
        json.dump(cache, fout, indent = 2)
print("cost = {}".format(COST))