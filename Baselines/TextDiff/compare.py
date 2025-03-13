import os
import json
import random
import datasets
import argparse
from utils.common import manual_seed
from utils.api_inference import create_OpenAIclient, openai_completion, prompt_to_chatml

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, required = True, choices = ("MATH", "WildChat10K", "DS-1000", ))
parser.add_argument("--results_path", type = str, required = True)
parser.add_argument("--negative_instance_num", type = int, default = 50)
parser.add_argument("--positive_instance_num", type = int, default = 50)
parser.add_argument("--maximum_size", type = int, default = 20)
parser.add_argument("--seed", type = int, default = 0)
args = parser.parse_args()
manual_seed(args.seed)

if args.dataset == "MATH" :
    results_type = "accuracy"
    PROMPT = "mathematics"
    INPUT_KEY, OUTPUT_KEY = "problem", "solution"
    INPUT_NAME, OUTPUT_NAME = "Question", "Solution"
    dataset = datasets.load_dataset("lighteval/MATH")["test"].to_list()
elif args.dataset == "WildChat10K" :
    results_type = "win-rate"
    PROMPT = "instruction-following"
    INPUT_KEY, OUTPUT_KEY = "instruction", "response"
    INPUT_NAME, OUTPUT_NAME = "Instruction", "Response"
    with open("Datasets/{}/dataset.json".format(args.dataset), "r") as fin :
        dataset = json.load(fin)
elif args.dataset == "DS-1000" :
    results_type = "accuracy"
    PROMPT = "ds-1000"
    INPUT_KEY, OUTPUT_KEY = "prompt", "reference_code"
    INPUT_NAME, OUTPUT_NAME = "Problem", "Implementation"
    dataset = datasets.load_dataset("xlangai/DS-1000")["test"].to_list()
else :
    raise NotImplementedError("dataset = {}".format(args.dataset))
with open(os.path.join("Datasets/{}/eval_results".format(args.dataset), args.results_path, "results.json"), "r") as fin :
    RESULTS = json.load(fin)
assert len(RESULTS) == len(dataset)

with open("Baselines/TextDiff/prompts/{}.txt".format(PROMPT), "r") as fin :
    PROMPT = fin.read()


NEGATIVE_INSTANCES, POSITIVE_INSTANCES = [], []
for result, instance in zip(RESULTS, dataset) :
    if results_type == "accuracy" :
        assert result in (0, 1)
        if result == 0 :
            NEGATIVE_INSTANCES.append(instance)
        elif result == 1 :
            POSITIVE_INSTANCES.append(instance)
    elif results_type == "win-rate" :
        assert isinstance(result, list) and len(result) == 2 and result[0] in (1, 2) and result[1] in (1, 2)
        if result[0] == 2 and result[1] == 2 :
            NEGATIVE_INSTANCES.append(instance)
        elif result[0] == 1 and result[1] == 1 :
            POSITIVE_INSTANCES.append(instance)
    else :
        raise NotImplementedError("results_type = {}".format(results_type))
random.shuffle(NEGATIVE_INSTANCES)
random.shuffle(POSITIVE_INSTANCES)
NEGATIVE_INSTANCES, POSITIVE_INSTANCES = NEGATIVE_INSTANCES[: args.negative_instance_num], POSITIVE_INSTANCES[: args.positive_instance_num]



OPENAI_KWARGS = {
    "model" : "gpt-4o",
    "max_tokens" : 4096,
    "temperature" : 0.0,
    "seed" : 0,
}
negative_inputs_and_outputs = ["### {} #{}\n{}\n\n### {} #{}\n{}\n".format(INPUT_NAME, index + 1, instance[INPUT_KEY], OUTPUT_NAME, index + 1, instance[OUTPUT_KEY]) for index, instance in enumerate(NEGATIVE_INSTANCES)]
positive_inputs_and_outputs = ["### {} #{}\n{}\n\n### {} #{}\n{}\n".format(INPUT_NAME, index + 1, instance[INPUT_KEY], OUTPUT_NAME, index + 1, instance[OUTPUT_KEY]) for index, instance in enumerate(POSITIVE_INSTANCES)]
prompt = PROMPT.format_map(dict(negative_instance_num = len(negative_inputs_and_outputs), negative_inputs_and_outputs = "\n".join(negative_inputs_and_outputs), positive_instance_num = len(positive_inputs_and_outputs), positive_inputs_and_outputs = "\n".join(positive_inputs_and_outputs), capability_num = args.maximum_size))
chatml = prompt_to_chatml(prompt = prompt)
client = create_OpenAIclient(dict(api_key = os.getenv("OpenAI_API_KEY")))
completion = openai_completion(client, chatml, OPENAI_KWARGS)
print("cost = {}".format(completion["cost"]))


output_path = os.path.join("Datasets/{}/eval_results".format(args.dataset), args.results_path, "TextDiff/[negative_instance={}]_[positive_instance={}]_[maximum={}]_[seed={}]".format(args.negative_instance_num, args.positive_instance_num, args.maximum_size, args.seed))
os.makedirs(output_path, exist_ok = True)
with open(os.path.join(output_path, "inputs.json"), "w") as fout :
    json.dump(dict(negative = negative_inputs_and_outputs, positive = positive_inputs_and_outputs), fout, indent = 2)
with open(os.path.join(output_path, "raw.txt"), "w") as fout :
    fout.write(completion["response"])
capabilities = [response.strip() for response in completion["response"].split("\n") if response.strip()]
with open(os.path.join(output_path, "weakness-profile.json"), "w") as fout :
    json.dump(capabilities, fout, indent = 2)