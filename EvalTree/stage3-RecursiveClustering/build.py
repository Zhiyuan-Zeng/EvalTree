import os
import json
import torch
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type = str, required = True, choices = ("MATH", "WildChat10K", "DS-1000", "MMLU", ))
parser.add_argument("--annotation_model", type = str, default = "gpt-4o-mini", choices = ("gpt-4o-mini", ))
parser.add_argument("--embedding_model", type = str, default = "text-embedding-3-small", choices = ("text-embedding-3-small", ))
parser.add_argument("--max_children", type = int, default = 10)
parser.add_argument("--split", type = str, default = "full")
args = parser.parse_args()

assert args.max_children >= 2
MATRIX = torch.stack(torch.load("Datasets/{}/EvalTree/stage2-CapabilityEmbedding/[annotation={}]_[embedding={}].bin".format(args.dataset, args.annotation_model, args.embedding_model), weights_only = True)).numpy()


def build_tree(instances : np.ndarray) :
    if len(instances) == 1 :
        return instances[0].item()
    if len(instances) == 2 :
        return {"subtrees" : instances.tolist(), "kmeans" : None}
    matrix = MATRIX[instances]
    
    all_labels, all_scores, all_kmeans = [], [], []
    for n_clusters in range(2, min(args.max_children + 1, len(instances))) :
        kmeans = KMeans(
            n_clusters = n_clusters,
            init = "k-means++",
            random_state = 42,
        ).fit(matrix)
        labels : np.ndarray = kmeans.labels_

        if labels.max() == 0 :
            assert len(instances) > 1
            return {"subtrees" : instances.tolist(), "kmeans" : None}

        all_labels.append(labels)
        all_scores.append(silhouette_score(matrix, labels, metric = "cosine"))
        all_kmeans.append(kmeans)
    
    if max(all_scores) <= 0.0 :
        return {"subtrees" : instances.tolist(), "kmeans" : None}
    
    picked_index = np.argmax(all_scores)
    labels = all_labels[picked_index]
    kmeans = all_kmeans[picked_index]

    cluster2subtree = {}
    for cluster in range(args.max_children) :
        subtree_instances = instances[labels == cluster]
        if len(subtree_instances) :
            cluster2subtree[cluster] = subtree_instances
    assert len(cluster2subtree)

    if len(cluster2subtree) == 1 : # There is only one cluster, and the instance number is more than 1.
        assert len(instances) > 1
        return {"subtrees" : instances.tolist(), "kmeans" : None}

    return {
        "subtrees" : {cluster : build_tree(subtree) for cluster, subtree in cluster2subtree.items()},
        "kmeans" : kmeans,
    }


os.makedirs("Datasets/{}/EvalTree/stage3-RecursiveClustering".format(args.dataset), exist_ok = True)
if args.split == "full" :
    RANGE = np.arange(MATRIX.shape[0])
else :
    with open("Datasets/{}/splits/{}.json".format(args.dataset, args.split), "r") as fin :
        RANGE = np.array(json.load(fin))
torch.save(build_tree(RANGE), "Datasets/{}/EvalTree/stage3-RecursiveClustering/[split={}]_[annotation={}]_[embedding={}]_[max-children={}].bin".format(args.dataset, args.split, args.annotation_model, args.embedding_model, args.max_children))