import os
import json
import random
import argparse
import datasets
import functools
import multiprocessing
from tqdm import tqdm
from utils.common import manual_seed
from utils.api_inference import create_OpenAIclient, openai_completion, prompt_to_chatml

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, required = True, choices = ("MATH", "DS-1000", ))
parser.add_argument("--capability_path", type = str, required = True)
parser.add_argument("--data_size", type = int, default = 64)
parser.add_argument("--num_procs", type = int, default = 32)
parser.add_argument("--model", type = str, default = "gpt-4o-mini", choices = ("gpt-4o-mini", ))
args = parser.parse_args()
manual_seed(0)


if args.dataset == "MATH" :
    PROMPT, INPUT_NAME = "mathematics", "Question"
    INPUT_KEY = "problem"
    dataset = datasets.load_dataset("lighteval/MATH")["test"].to_list()
elif args.dataset == "DS-1000" :
    PROMPT, INPUT_NAME = "ds-1000", "Problem"
    INPUT_KEY = "prompt"
    dataset = datasets.load_dataset("xlangai/DS-1000")["test"].to_list()
else :
    raise NotImplementedError("dataset = {}".format(args.dataset))
with open("Assessments/Extrinsic/data/prompts/generate_input/{}.txt".format(PROMPT), "r") as fin :
    PROMPT = fin.read()

with open(args.capability_path, "r") as fin :
    capabilities_with_instances = json.load(fin)


OPENAI_KWARGS = {
    "model" : args.model,
    "max_tokens" : 4096,
    "temperature" : 1.0,
    "seed" : 0,
}
def Process(args, capability, inputs) :
    sampled_inputs = random.sample(inputs, min(5, len(inputs)))
    random.shuffle(sampled_inputs)
    sampled_inputs = ["### {} #{}\n{}\n".format(INPUT_NAME, index + 1, sampled_input) for index, sampled_input in enumerate(sampled_inputs)]
    chatml = prompt_to_chatml(PROMPT.format_map(dict(capability = capability, instance_num = len(sampled_inputs), example_inputs = "\n".join(sampled_inputs))))

    client = create_OpenAIclient(dict(api_key = os.getenv("OpenAI_API_KEY")))
    return openai_completion(client, chatml, OPENAI_KWARGS)


COST = 0.0
for capability_with_instance in capabilities_with_instances :
    capability = capability_with_instance["capability"]
    try :
        with open("Assessments/Extrinsic/data/pools/{}/[input-generation={}].json".format(args.dataset, args.model), "r") as fin :
            cache = json.load(fin)
    except FileNotFoundError :
        cache = {}
    except :
        assert False
    if capability in cache and len(cache[capability]) >= args.data_size :
        continue
    else :
        if capability not in cache :
            cache[capability] = []
    
    inputs = [dataset[index][INPUT_KEY] for index in capability_with_instance["instances"]]

    with multiprocessing.Pool(args.num_procs) as p :
        _Process = functools.partial(Process, capability = capability, inputs = inputs)
        outputs = list(
            tqdm(
                p.imap(_Process, [args] * (args.data_size - len(cache[capability]))),
                desc = "generate synthetic data for capability = {}".format(capability),
                total = args.data_size - len(cache[capability]),
            )
        )
    COST += sum([output["cost"] for output in outputs])
    cache[capability] += [output["response"] for output in outputs]

    os.makedirs("Assessments/Extrinsic/data/pools/{}".format(args.dataset), exist_ok = True)
    with open("Assessments/Extrinsic/data/pools/{}/[input-generation={}].json".format(args.dataset, args.model), "w") as fout :
        json.dump(cache, fout, indent = 2)
print("cost = {}".format(COST))