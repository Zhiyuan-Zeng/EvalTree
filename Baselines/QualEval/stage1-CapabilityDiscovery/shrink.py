import os
import json
import argparse
import functools
import multiprocessing
from tqdm import tqdm
from utils.api_inference import create_OpenAIclient, openai_completion, prompt_to_chatml

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, required = True, choices = ("MATH", "WildChat10K", "DS-1000", ))
parser.add_argument("--num_procs", type = int, default = 20)
parser.add_argument("--chunk_size", type = int, default = 20)
parser.add_argument("--model", type = str, default = "gpt-4o-mini", choices = ("gpt-4o-mini", ))
parser.add_argument("--num_capabilities", type = int, default = 20)
parser.add_argument("--shrink_factor", type = int, default = 4)
parser.add_argument("--round", type = int, required = True)
args = parser.parse_args()
assert args.round >= 1


if args.dataset == "MATH" :
    PROMPT = "mathematics"
elif args.dataset == "WildChat10K" :
    PROMPT = "instruction-following"
elif args.dataset == "DS-1000" :
    PROMPT = "ds-1000"
else :
    raise NotImplementedError("dataset = {}".format(args.dataset))
with open("Baselines/QualEval/stage1-CapabilityDiscovery/prompts/shrink/{}.txt".format(PROMPT), "r") as fin :
    PROMPT = fin.read()

with open("Datasets/{}/QualEval/stage1-CapabilityDiscovery/[chunk={}]_[model={}]/{}.json".format(args.dataset, args.chunk_size, args.model, "initialize" if args.round == 1 else "[num={}]_[factor={}]_[round={}]".format(args.num_capabilities, args.shrink_factor, args.round - 1)), "r") as fin :
    capabilities = [capability for chunk in json.load(fin) for capability in chunk]
if len(capabilities) <= args.num_capabilities :
    print("No need to shrink capabilities")
    exit(0)
CHUNK_SIZE = args.num_capabilities * args.shrink_factor
capabilities = [capabilities[start_index : start_index + CHUNK_SIZE] for start_index in range(0, len(capabilities), CHUNK_SIZE)]


OPENAI_KWARGS = {
    "model" : args.model,
    "max_tokens" : 4096,
    "temperature" : 0.0,
    "seed" : 0,
}
def Process(chunk) :
    if len(chunk) <= args.num_capabilities :
        return dict(cost = 0, response = "\n".join(chunk))
    capabilities = ["({}) {}\n".format(index + 1, capability) for index, capability in enumerate(chunk)]
    prompt = PROMPT.format_map(dict(current_num_capabilities = len(capabilities), num_capabilities = args.num_capabilities, capability_list = "\n".join(capabilities)))
    chatml = prompt_to_chatml(prompt = prompt)
    client = create_OpenAIclient(dict(api_key = os.getenv("OpenAI_API_KEY")))
    return openai_completion(client, chatml, OPENAI_KWARGS)
with multiprocessing.Pool(args.num_procs) as p :
    _Process = functools.partial(Process)
    outputs = list(
        tqdm(
            p.imap(_Process, capabilities),
            desc = "capabilities",
            total = len(capabilities),
        )
    )


print("cost = {}".format(sum([output["cost"] for output in outputs])))
with open("Datasets/{}/QualEval/stage1-CapabilityDiscovery/[chunk={}]_[model={}]/{}.json".format(args.dataset, args.chunk_size, args.model, "[num={}]_[factor={}]_[round={}]".format(args.num_capabilities, args.shrink_factor, args.round)), "w") as fout :
    json.dump([[response.strip() for response in output["response"].split("\n") if response.strip()] for output in outputs], fout, indent = 2)