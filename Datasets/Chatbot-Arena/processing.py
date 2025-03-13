import json
import datasets

ranked_models = ['gpt-4-1106-preview', 'gpt-4-0125-preview', 'gpt-4-0314', 'gpt-4-0613', 'qwen1.5-72b-chat', 'mistral-medium', 'claude-1', 'gemini-pro-dev-api', 'claude-2.0', 'gemini-pro', 'mixtral-8x7b-instruct-v0.1', 'yi-34b-chat', 'gpt-3.5-turbo-0613', 'starling-lm-7b-alpha', 'wizardlm-70b', 'llama-2-70b-chat', 'claude-2.1', 'nous-hermes-2-mixtral-8x7b-dpo', 'gpt-3.5-turbo-0125', 'gpt-3.5-turbo-0314', 'claude-instant-1', 'pplx-70b-online', 'vicuna-33b', 'tulu-2-dpo-70b', 'solar-10.7b-instruct-v1.0', 'llama2-70b-steerlm-chat', 'deepseek-llm-67b-chat', 'openchat-3.5', 'openhermes-2.5-mistral-7b', 'mistral-7b-instruct-v0.2', 'dolphin-2.2.1-mistral-7b', 'wizardlm-13b', 'openchat-3.5-0106', 'gpt-3.5-turbo-1106', 'qwen1.5-7b-chat', 'codellama-34b-instruct', 'llama-2-13b-chat', 'zephyr-7b-beta', 'llama-2-7b-chat', 'pplx-7b-online', 'guanaco-33b', 'zephyr-7b-alpha', 'mpt-30b-chat', 'vicuna-13b', 'stripedhyena-nous-7b', 'falcon-180b-chat', 'qwen-14b-chat', 'palm-2', 'mistral-7b-instruct', 'qwen1.5-4b-chat', 'vicuna-7b', 'koala-13b', 'chatglm3-6b', 'gpt4all-13b-snoozy', 'mpt-7b-chat', 'chatglm2-6b', 'RWKV-4-Raven-14B', 'alpaca-13b', 'oasst-pythia-12b', 'fastchat-t5-3b', 'chatglm-6b', 'stablelm-tuned-alpha-7b', 'dolly-v2-12b', 'llama-13b']


dataset = list(datasets.load_dataset("potsawee/chatbot-arena-llm-judges")["train"])
instruction2responses = {instance["question"] : {} for instance in dataset}
for instance in dataset :
    for key in ("a", "b") :
        instruction2responses[instance["question"]][instance["model_{}".format(key)]] = instance["answer_{}".format(key)]


dataset = []
for instruction, responses in instruction2responses.items() :
    if "<|im_start|>" in instruction :
        continue
    picked_response = None
    for model in ranked_models :
        if model in responses :
            picked_response = responses[model]
            break
    if picked_response is None :
        continue
    dataset.append({"instruction" : instruction, "response" : picked_response})
with open("dataset.json", "w") as fout :
    json.dump(dataset, fout, indent = 2)