import os
import json
import argparse
import datasets
import functools
import multiprocessing
from tqdm import tqdm
from utils.api_inference import create_OpenAIclient, openai_completion, prompt_to_chatml

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, required = True, choices = ("MATH", "DS-1000", ))
parser.add_argument("--source", type = str, required = True)
parser.add_argument("--num_procs", type = int, default = 64)
parser.add_argument("--model", type = str, default = "gpt-4o-mini", choices = ("gpt-4o-mini", ))
args = parser.parse_args()


if args.source.startswith("[input-generation=") and args.source.endswith("]") :
    with open("Assessments/Extrinsic/data/pools/{}/{}.json".format(args.dataset, args.source), "r") as fin :
        inputs = [capability_input for capability_inputs in json.load(fin).values() for capability_input in capability_inputs]
elif args.source == "original" :
    if args.dataset == "MATH" :
        dataset = datasets.load_dataset("lighteval/MATH")["test"].to_list()
        with open("Datasets/MATH/splits/4k-1k.json", "r") as fin :
            dataset = [dataset[index] for index in json.load(fin)]
        inputs = [instance["problem"] for instance in dataset]
        dataset = None
    elif args.dataset == "DS-1000" :
        dataset = datasets.load_dataset("xlangai/DS-1000")["test"].to_list()
        with open("Datasets/DS-1000/splits/600-400.json", "r") as fin :
            dataset = [dataset[index] for index in json.load(fin)]
        inputs = [instance["prompt"] for instance in dataset]
        dataset = None
    else :
        raise NotImplementedError("dataset = {}".format(args.dataset))
else :
    raise NotImplementedError("source = {}".format(args.source))
with open("Assessments/Extrinsic/data/prompts/generate_output/{}.txt".format(args.dataset), "r") as fin :
    PROMPT = fin.read()

try :
    with open("Assessments/Extrinsic/data/pools/{}/[output-generation={}].json".format(args.dataset, args.model), "r") as fin :
        cache = json.load(fin)
except FileNotFoundError :
    cache = {}
except :
    assert False


OPENAI_KWARGS = {
    "model" : args.model,
    "max_tokens" : 4096,
    "temperature" : 0.0,
    "seed" : 0,
}
def Process(input) :
    if input in cache :
        return dict(response = cache[input], cost = 0.0)
    try :
        prompt = PROMPT.format_map(dict(input = input))
    except :
        prompt = PROMPT.replace("{input}", input)
    chatml = prompt_to_chatml(prompt)
    client = create_OpenAIclient(dict(api_key = os.getenv("OpenAI_API_KEY")))
    return openai_completion(client, chatml, OPENAI_KWARGS)

with multiprocessing.Pool(args.num_procs) as p :
    _Process = functools.partial(Process)
    outputs = list(
        tqdm(
            p.imap(_Process, inputs),
            desc = "inputs",
            total = len(inputs),
        )
    )
print("cost = {}".format(sum([output["cost"] for output in outputs])))
for input, output in zip(inputs, outputs) :
    cache[input] = output["response"]


with open("Assessments/Extrinsic/data/pools/{}/[output-generation={}].json".format(args.dataset, args.model), "w") as fout :
    json.dump(cache, fout, indent = 2)