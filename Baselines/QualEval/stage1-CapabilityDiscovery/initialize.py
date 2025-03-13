import os
import json
import random
import datasets
import argparse
import functools
import multiprocessing
from tqdm import tqdm
from utils.common import manual_seed
from utils.api_inference import create_OpenAIclient, openai_completion, prompt_to_chatml

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, required = True, choices = ("MATH", "WildChat10K", "DS-1000", ))
parser.add_argument("--num_procs", type = int, default = 64)
parser.add_argument("--chunk_size", type = int, default = 20)
parser.add_argument("--model", type = str, default = "gpt-4o-mini", choices = ("gpt-4o-mini", ))
args = parser.parse_args()
manual_seed(0)


if args.dataset == "MATH" :
    PROMPT = "mathematics"
    INPUT_KEY, OUTPUT_KEY = "problem", "solution"
    INPUT_NAME, OUTPUT_NAME = "Question", "Solution"
    dataset = datasets.load_dataset("lighteval/MATH")["test"].to_list()
elif args.dataset == "WildChat10K" :
    PROMPT = "instruction-following"
    INPUT_KEY, OUTPUT_KEY = "instruction", "response"
    INPUT_NAME, OUTPUT_NAME = "Instruction", "Response"
    with open("Datasets/{}/dataset.json".format(args.dataset), "r") as fin :
        dataset = json.load(fin)
elif args.dataset == "DS-1000" :
    PROMPT = "ds-1000"
    INPUT_KEY, OUTPUT_KEY = "prompt", "reference_code"
    INPUT_NAME, OUTPUT_NAME = "Problem", "Implementation"
    dataset = datasets.load_dataset("xlangai/DS-1000")["test"].to_list()
else :
    raise NotImplementedError("dataset = {}".format(args.dataset))
random.shuffle(dataset)
dataset = [dataset[start_index : start_index + args.chunk_size] for start_index in range(0, len(dataset), args.chunk_size)]

with open("Baselines/QualEval/stage1-CapabilityDiscovery/prompts/initialize/{}.txt".format(PROMPT), "r") as fin :
    PROMPT = fin.read()


OPENAI_KWARGS = {
    "model" : args.model,
    "max_tokens" : 4096,
    "temperature" : 0.0,
    "seed" : 0,
}
def Process(chunk) :
    inputs_and_outputs = ["### {} #{}\n{}\n\n### {} #{}\n{}\n".format(INPUT_NAME, index + 1, instance[INPUT_KEY], OUTPUT_NAME, index + 1, instance[OUTPUT_KEY]) for index, instance in enumerate(chunk)]
    prompt = PROMPT.format_map(dict(instance_num = len(inputs_and_outputs), inputs_and_outputs = "\n".join(inputs_and_outputs)))
    chatml = prompt_to_chatml(prompt = prompt)
    client = create_OpenAIclient(dict(api_key = os.getenv("OpenAI_API_KEY")))
    return openai_completion(client, chatml, OPENAI_KWARGS)
with multiprocessing.Pool(args.num_procs) as p :
    _Process = functools.partial(Process)
    outputs = list(
        tqdm(
            p.imap(_Process, dataset),
            desc = "dataset",
            total = len(dataset),
        )
    )


print("cost = {}".format(sum([output["cost"] for output in outputs])))
os.makedirs("Datasets/{}/QualEval/stage1-CapabilityDiscovery/[chunk={}]_[model={}]".format(args.dataset, args.chunk_size, args.model), exist_ok = True)
with open("Datasets/{}/QualEval/stage1-CapabilityDiscovery/[chunk={}]_[model={}]/initialize.json".format(args.dataset, args.chunk_size, args.model), "w") as fout :
    json.dump([[response.strip() for response in output["response"].split("\n") if response.strip()] for output in outputs], fout, indent = 2)