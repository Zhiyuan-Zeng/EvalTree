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
parser.add_argument("--num_procs", type = int, default = 64)
parser.add_argument("--model", type = str, default = "gpt-4o-mini", choices = ("gpt-4o-mini", ))
parser.add_argument("--chunk_size", type = int, default = 20)
parser.add_argument("--num_capabilities", type = int, default = 20)
parser.add_argument("--shrink_factor", type = int, default = 4)
parser.add_argument("--round", type = int, required = True)
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
with open("Baselines/QualEval/stage2-CapabilityAssignment/prompts/{}.txt".format(PROMPT), "r") as fin :
    PROMPT = fin.read()
with open("Datasets/{}/QualEval/stage1-CapabilityDiscovery/[chunk={}]_[model={}]/{}.json".format(args.dataset, args.chunk_size, args.model, "initialize" if args.round == 0 else "[num={}]_[factor={}]_[round={}]".format(args.num_capabilities, args.shrink_factor, args.round)), "r") as fin :
    capabilities = [capability for chunk in json.load(fin) for capability in chunk]


OPENAI_KWARGS = {
    "model" : args.model,
    "max_tokens" : 4096,
    "temperature" : 0.0,
    "seed" : 0,
}
def Process(instance) :
    def format_prompt(prompt, dict) :
        for key, value in dict.items() :
            prompt = prompt.replace("{" + key + "}", str(value))
        return prompt
    prompt = format_prompt(PROMPT, dict(num_capabilities = len(capabilities), capability_list = "\n".join(["({}) {}\n".format(index + 1, capability) for index, capability in enumerate(capabilities)]), input = instance[INPUT_KEY], output = instance[OUTPUT_KEY]))
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
for index, output in enumerate(outputs) :
    start_index, end_index = output["response"].find("{"), output["response"].rfind("}")
    try :
        outputs[index] = dict(scoring = json.loads(output["response"][start_index : end_index + 1]))
    except :
        outputs[index] = dict(scoring = output["response"])
os.makedirs("Datasets/{}/QualEval/stage2-CapabilityAssignment".format(args.dataset), exist_ok = True)
with open("Datasets/{}/QualEval/stage2-CapabilityAssignment/[chunk={}]_[model={}]_{}.json".format(args.dataset, args.chunk_size, args.model, "initialize" if args.round == 0 else "[num={}]_[factor={}]_[round={}]".format(args.num_capabilities, args.shrink_factor, args.round)), "w") as fout :
    json.dump(outputs, fout, indent = 2)