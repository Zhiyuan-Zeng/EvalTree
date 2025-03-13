import os
import json
import datasets
import argparse
import functools
import multiprocessing
from tqdm import tqdm
from utils.api_inference import create_OpenAIclient, openai_completion, prompt_to_chatml


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, required = True, choices = ("MATH", "WildChat10K", "DS-1000", ) + ("Chatbot-Arena", "ShareGPT10K", "MMLU", "CollegeMath", ))
parser.add_argument("--num_procs", type = int, default = 64)
parser.add_argument("--annotation_model", type = str, default = "gpt-4o-mini", choices = ("gpt-4o-mini", ))
args = parser.parse_args()


if args.dataset == "MATH" :
    PROMPT = "mathematics"
    INPUT_KEY, OUTPUT_KEY = "problem", "solution"
    dataset = datasets.load_dataset("lighteval/MATH")["test"].to_list()
elif args.dataset in ("WildChat10K", "Chatbot-Arena", "ShareGPT10K", ) :
    PROMPT = "instruction-following"
    INPUT_KEY, OUTPUT_KEY = "instruction", "response"
    with open("Datasets/{}/dataset.json".format(args.dataset), "r") as fin :
        dataset = json.load(fin)
elif args.dataset == "DS-1000" :
    PROMPT = "ds-1000"
    INPUT_KEY, OUTPUT_KEY = "prompt", "reference_code"
    dataset = datasets.load_dataset("xlangai/DS-1000")["test"].to_list()
elif args.dataset == "MMLU" :
    PROMPT = "mmlu"
    INPUT_KEY, OUTPUT_KEY = "question", "[gpt-4o-mini]_answer"
    with open("Datasets/MMLU/dataset.json", "r") as fin :
        dataset = json.load(fin)
elif args.dataset in ("CollegeMath", ) :
    PROMPT = "mathematics"
    INPUT_KEY, OUTPUT_KEY = "question", "[gpt-4o-mini]_solution"
    with open("Datasets/{}/dataset.json".format(args.dataset), "r") as fin :
        dataset = json.load(fin)
else :
    raise NotImplementedError("dataset = {}".format(args.dataset))
with open("EvalTree/stage1-CapabilityAnnotation/prompts/{}.txt".format(PROMPT), "r") as fin :
    PROMPT = fin.read()


OPENAI_KWARGS = {
    "model" : args.annotation_model,
    "max_tokens" : 1024,
    "temperature" : 0.0,
    "seed" : 0,
}
def Process(instance) :
    chatml = prompt_to_chatml(PROMPT.format_map(dict(input = instance[INPUT_KEY], output = instance[OUTPUT_KEY])))
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
os.makedirs("Datasets/{}/EvalTree/stage1-CapabilityAnnotation".format(args.dataset), exist_ok = True)
with open("Datasets/{}/EvalTree/stage1-CapabilityAnnotation/[annotation={}].json".format(args.dataset, args.annotation_model), "w") as fout :
    json.dump([output["response"] for output in outputs], fout, indent = 2)