import json
import argparse
import datasets
import statsmodels.api as sm

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, required = True, choices = ("MATH", "WildChat10K", "Chatbot-Arena", "Chatbot-Arena_NEW", "MMLU", "DS-1000", ))
args = parser.parse_args()


with open("Datasets/{}/EvalTree/stage3-RecursiveClustering/[split=full]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10]_[stage4-CapabilityDescription-model=gpt-4o].json".format(args.dataset), "r") as fin :
    TREE_DESCRIPTION = json.load(fin)

model2results, TREE_RANKING = None, None
if args.dataset == "MATH" :
    inputs = [instance["problem"] for instance in datasets.load_dataset("lighteval/MATH")["test"]]
    model2results = {}
    for model in ("gpt-4o-mini-2024-07-18", "Llama-3.1-8B-Instruct", "deepseek-math-7b-instruct", "Llama-3-8B-Instruct", "dart-math-llama3-8b-uniform", ) :
        with open("Datasets/MATH/eval_results/{}/results.json".format(model), "r") as fin :
            model2results[model] = json.load(fin)
            assert len(model2results[model]) == len(inputs)
elif args.dataset in ("WildChat10K", "Chatbot-Arena", "Chatbot-Arena_NEW", ) :
    with open("Datasets/{}/dataset.json".format(args.dataset), "r") as fin :
        inputs = [instance["instruction"] for instance in json.load(fin)]
    if args.dataset == "WildChat10K" :
        model2results = {}
        model1, model2 = "llama3.2-3b-instruct", "gemma2-2b-it"
        with open("Datasets/WildChat10K/eval_results/[{}]BEAT[{}]/results.json".format(model1, model2), "r") as fin :
            model2results[(model1, model2)] = json.load(fin)
            assert len(model2results[(model1, model2)]) == len(inputs)
    elif args.dataset in ("Chatbot-Arena", "Chatbot-Arena_NEW", ) :
        with open("Datasets/{}/eval_results/battleground/EvalTree_ranking.json".format(args.dataset), "r") as fin :
            TREE_RANKING = json.load(fin)
    else :
        raise NotImplementedError("dataset = {}".format(args.dataset))
elif args.dataset == "MMLU" :
    with open("Datasets/MMLU/dataset.json", "r") as fin :
        inputs = [instance["question"] for instance in json.load(fin)]
    model2results = {}
    for model in ("gpt-4o-mini-2024-07-18", "gpt-3.5-turbo", "claude-3.5-haiku", "Llama-3.1-8B-Instruct", "Llama-3.1-70B-Instruct", "Llama-3.1-Tulu-3-8B", "Llama-3.1-Tulu-3-70B", "Qwen2.5-7B-Instruct", "Qwen2.5-72B-Instruct", ) :
        with open("Datasets/MMLU/eval_results/{}/results.json".format(model), "r") as fin :
            model2results[model] = json.load(fin)
            assert len(model2results[model]) == len(inputs)
elif args.dataset == "DS-1000" :
    inputs = [instance["prompt"] for instance in datasets.load_dataset("xlangai/DS-1000")["test"]]
    model2results = {}
    for model in ("gpt-4o-2024-08-06", "gpt-3.5-turbo-0613", "deepseek-coder-6.7b-base", ) :
        with open("Datasets/DS-1000/eval_results/{}/results.json".format(model), "r") as fin :
            model2results[model] = json.load(fin)
            assert len(model2results[model]) == len(inputs)
else :
    raise NotImplementedError("dataset = {}".format(args.dataset))


def calculate_results(tree_description) :
    tree_results = {
        "size" : 0,
        "model2sum_metrics" : {},
    }
    if not isinstance(tree_description["subtrees"], int) :
        if isinstance(tree_description["subtrees"], list) :
            tree_results["subtrees"] = []
            for subtree_description in tree_description["subtrees"] :
                tree_results["subtrees"].append(calculate_results(subtree_description))
        elif isinstance(tree_description["subtrees"], dict) :
            tree_results["subtrees"] = {}
            for cluster, subtree_description in tree_description["subtrees"].items() :
                tree_results["subtrees"][cluster] = calculate_results(subtree_description)
        else :
            assert False
        
        tree_results["size"] = 0
        for model in model2results.keys() :
            tree_results["model2sum_metrics"][model] = 0
        for subtree_results in tree_results["subtrees"] if isinstance(tree_results["subtrees"], list) else tree_results["subtrees"].values() :
            tree_results["size"] += subtree_results["size"]
            for model, sum_metrics in subtree_results["model2sum_metrics"].items() :
                tree_results["model2sum_metrics"][model] += sum_metrics
    else :
        tree_results["size"] = 1
        for model, results in model2results.items() :
            metrics = results[tree_description["subtrees"]]
            if args.dataset in ("MATH", "DS-1000", "MMLU", ) :
                assert metrics in (0, 1)
                tree_results["model2sum_metrics"][model] = metrics
            elif args.dataset in ("WildChat10K", ) :
                assert isinstance(metrics, list) and len(metrics) == 2
                assert metrics[0] in (1, 2) and metrics[1] in (1, 2)
                tree_results["model2sum_metrics"][model] = (int(metrics[0] == 1) + int(metrics[1] == 1)) / 2.0
            else :
                raise NotImplementedError("dataset = {}".format(args.dataset))
        tree_results["subtrees"] = tree_description["subtrees"]
    return tree_results

if model2results is not None :
    assert TREE_RANKING is None
    TREE_RESULTS = calculate_results(TREE_DESCRIPTION)
    model2results = None
else :
    assert TREE_RANKING is not None
    TREE_RESULTS = TREE_RANKING


def merge(tree_results, tree_description, depth) :
    tree_data = {"capability" : tree_description["description"], "size" : 0, "depth" : depth}
    if not isinstance(tree_results["subtrees"], int) :
        tree_data["subtrees"] = []
        if isinstance(tree_results["subtrees"], list) :
            assert isinstance(tree_description["subtrees"], list)
            assert len(tree_results["subtrees"]) == len(tree_description["subtrees"])
            for subtree_results, subtree_description in zip(tree_results["subtrees"], tree_description["subtrees"]) :
                tree_data["subtrees"].append(merge(subtree_results, subtree_description, depth + 1))
        else :
            assert isinstance(tree_description["subtrees"], dict) and isinstance(tree_results["subtrees"], dict)
            assert set(tree_results["subtrees"].keys()) == set(tree_description["subtrees"].keys())
            for cluster, subtree_results in tree_results["subtrees"].items() :
                tree_data["subtrees"].append(merge(subtree_results, tree_description["subtrees"][cluster], depth + 1))
        for subtree_data in tree_data["subtrees"] :
            tree_data["size"] += subtree_data["size"]
        
        distinctions = tree_description["distinguishing"].split("\n")
        if len(distinctions) == len(tree_data["subtrees"]) :
            for distinction, subtree_data in zip(distinctions, tree_data["subtrees"]) :
                subtree_data["distinction"] = distinction.strip()
    else :
        assert isinstance(tree_description["subtrees"], int)
        assert tree_results["subtrees"] == tree_description["subtrees"]
        tree_data["size"] = 1
        tree_data["input"] = inputs[tree_results["subtrees"]]
        if len(tree_data["input"]) > 384 :
            tree_data["input"] = tree_data["input"][: 384] + " ..."
        tree_data["subtrees"] = tree_results["subtrees"]
    
    if TREE_RANKING is not None :
        tree_data["ranking"] = tree_results["ranking"]
    else :
        assert tree_results["size"] == tree_data["size"]

        if tree_data["size"] >= 5 :
            tree_data["CI"] = {}

        if args.dataset in ("MATH", "DS-1000", "MMLU", ) :
            tree_data["ranking"] = [[model, sum_metrics / tree_results["size"]] for model, sum_metrics in tree_results["model2sum_metrics"].items()]
            
            if "CI" in tree_data :
                for model, sum_metrics in tree_results["model2sum_metrics"].items() :
                    lower_bound, upper_bound = sm.stats.proportion_confint(sum_metrics, tree_results["size"], alpha = 0.05, method = "beta")
                    tree_data["CI"][model] = [lower_bound, upper_bound]
        elif args.dataset in ("WildChat10K", ) :
            assert len(tree_results["model2sum_metrics"]) == 1
            model1, model2 = list(tree_results["model2sum_metrics"].keys())[0]
            model1_winrate = list(tree_results["model2sum_metrics"].values())[0] / tree_results["size"]
            tree_data["ranking"] = [[model1, model1_winrate], [model2, 1.0 - model1_winrate]]
            
            if "CI" in tree_data :
                lower_bound, upper_bound = sm.stats.proportion_confint(int(list(tree_results["model2sum_metrics"].values())[0] * 2.0), tree_results["size"] * 2, alpha = 0.05, method = "beta")
                tree_data["CI"][model1] = [lower_bound, upper_bound]
                lower_bound, upper_bound = sm.stats.proportion_confint(tree_results["size"] * 2 - int(list(tree_results["model2sum_metrics"].values())[0] * 2.0), tree_results["size"] * 2, alpha = 0.05, method = "beta")
                tree_data["CI"][model2] = [lower_bound, upper_bound]
        else :
            raise NotImplementedError("dataset = {}".format(args.dataset))
        tree_data["ranking"].sort(key = lambda x : x[1], reverse = True)
    
    return tree_data


with open("../data/{}.json".format(args.dataset), "w") as fout :
    json.dump(merge(TREE_RESULTS, TREE_DESCRIPTION, depth = 1), fout, indent = 2)