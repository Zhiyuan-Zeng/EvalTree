import os
import json
import torch
import argparse
import functools
import multiprocessing
from tqdm import tqdm
from utils.api_inference import create_OpenAIclient, openai_embedding

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, required = True, choices = ("MATH", "WildChat10K", "DS-1000", ) + ("Chatbot-Arena", "ShareGPT10K", "MMLU", "CollegeMath", ))
parser.add_argument("--num_procs", type = int, default = 64)
parser.add_argument("--annotation_model", type = str, default = "gpt-4o-mini", choices = ("gpt-4o-mini", ))
parser.add_argument("--embedding_model", type = str, default = "text-embedding-3-small", choices = ("text-embedding-3-small", ))
args = parser.parse_args()


with open("Datasets/{}/EvalTree/stage1-CapabilityAnnotation/[annotation={}].json".format(args.dataset, args.annotation_model), "r") as fin :
    CAPABILITIES = json.load(fin)


if args.dataset in ("MATH", "DS-1000", "CollegeMath", ) :
    PREFIX = "The model has the following skill or capability: "
elif args.dataset in ("WildChat10K", "Chatbot-Arena", "ShareGPT10K", "MMLU", ) :
    PREFIX = "The model has the following capability: "
else :
    raise NotImplementedError("dataset = {}".format(args.dataset))


def Process(capability) :
    client = create_OpenAIclient(dict(api_key = os.getenv("OpenAI_API_KEY")))
    return openai_embedding(client, text = PREFIX + capability, model = args.embedding_model)
with multiprocessing.Pool(args.num_procs) as p :
    _Process = functools.partial(Process)
    outputs = list(
        tqdm(
            p.imap(_Process, CAPABILITIES),
            desc = "CAPABILITIES",
            total = len(CAPABILITIES),
        )
    )


print("cost = {}".format(sum([output["cost"] for output in outputs])))
os.makedirs("Datasets/{}/EvalTree/stage2-CapabilityEmbedding".format(args.dataset), exist_ok = True)
torch.save([torch.tensor(output["embedding"]) for output in outputs], "Datasets/{}/EvalTree/stage2-CapabilityEmbedding/[annotation={}]_[embedding={}].bin".format(args.dataset, args.annotation_model, args.embedding_model))