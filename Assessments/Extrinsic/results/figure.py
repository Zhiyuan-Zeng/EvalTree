import os
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


method_names = [
    r"Initial LM",
    r"Generic-Capability-Guided \textbf{Synthetic Data}",
    r"\textsc{TextDiff}-Guided \textbf{Synthetic Data}",
    r"\textsc{QualEval}-Guided \textbf{Synthetic Data}",
    r"\textsc{EvalTree}-Guided \textbf{Synthetic Data}",
    r"Directly-Sampled Data",
]

def get_methodlist(dataset) :
    methodlist = [
        {
            "path" : "Generic-Capability",
            "seed" : 5,
        },
        {
            "path" : "TextDiff",
            "seed" : 5,
        },
        {
            "path" : "QualEval",
            "seed" : 5,
        },
        {
            "path" : "EvalTree",
            "seed" : 5,
        },
        {
            "path" : "Directly-Sampled",
            "seed" : 5,
        },
    ]
    
    if dataset == "MATH" :
        methodlist = [dict(path = "Llama-3.1-8B-Instruct", seed = None)] + methodlist
    elif dataset == "DS-1000" :
        methodlist = [dict(path = "deepseek-coder-6.7b-base", seed = None)] + methodlist
    else :
        raise ValueError("Unknown dataset: {}".format(dataset))
    return methodlist

def get_performance(dataset, path) :
    if dataset == "MATH" :
        with open(os.path.join(path, "metrics.json"), "r") as fin :
            return json.load(fin)["exact_match_flex"] * 100.0
    elif dataset == "DS-1000" :
        with open(os.path.join(path, "metrics.txt"), "r") as fin :
            text = fin.read()
            return float(text.split("mean")[1].split("score")[0].strip()) * 100.0
    else :
        raise ValueError("Unknown dataset: {}".format(dataset))


plt.rcParams["font.family"] = "Palatino"
mpl.rcParams["text.usetex"] = True
mpl.rcParams["mathtext.default"] = "regular"
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14

fig, axes = plt.subplots(1, 2, figsize = (12, 4))

datasets = ["MATH", "DS-1000"]

legend_bars = []
legend_labels = method_names

for idx, dataset in enumerate(datasets) :
    path_prefix = "Assessments/Extrinsic/results/{}".format(dataset)
    method_list = get_methodlist(dataset)

    means = []
    stderrs = []
    for item in method_list : 
        path = item["path"]
        seed = item["seed"]
        if seed is None :
            performance = get_performance(dataset, os.path.join(path_prefix, path))
            means.append(performance)
            stderrs.append(None)
        else :
            performance_list = [
                get_performance(dataset, os.path.join(
                    path_prefix, path + "_[seed={}]".format(s))
                )
                for s in range(seed)
            ]
            means.append(np.mean(performance_list))
            stderrs.append(np.std(performance_list) / np.sqrt(len(performance_list)))

    x = [i for i in range(len(method_list))]
    x[-1] += 0.5

    colors = ["#92C5E0", "#A569BD", "#C86862", "#4575B4", "#D9A441", "#5A5A5A"]
    hatches = ['////', '////', '////', '////', '////', '////']

    bars = []
    for i in range(len(method_list)) :
        bar = axes[idx].bar(
            x[i],
            means[i],
            yerr = stderrs[i] if stderrs[i] is not None else 0,
            capsize = 6,
            color = colors[i % len(colors)],
            edgecolor = 'white',
            hatch = hatches[i % len(hatches)],
            width = 0.8,
            error_kw = dict(lw = 4, capthick = 2, ecolor = 'black'),
        )
        bars.append(bar)
    
    if idx == 0 :
        legend_bars = [b[0] for b in bars]

    for i in range(len(x)) :
        label = "{:.2f}".format(means[i])
        if "EvalTree" in method_names[i] :
            label = r"\textbf{" + label + r"}"
        y_position = means[i] + (stderrs[i] if stderrs[i] is not None else 0) + 0.02
        axes[idx].text(
            x[i],
            y_position,
            label,
            ha = "center",
            va = "bottom",
            fontsize = 18,
        )

    axes[idx].set_ylim([min(means) * 0.97, max(means) * (1.04 if dataset == "DS-1000" else 1.02)])
    axes[idx].set_xticks(x)
    axes[idx].set_xticklabels([""] * len(method_list))
    axes[idx].set_ylabel(r"Accuracy (\%)", fontsize = 15)
    axes[idx].set_title(r"\textbf{" + "{}".format(dataset) + r"}", fontsize = 24)
    axes[idx].yaxis.set_major_locator(mticker.MaxNLocator(nbins = 6))
    axes[idx].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    axes[idx].grid(True, linestyle = "--", alpha = 0.6)

plt.tight_layout()
fig.subplots_adjust(bottom = 0.2, left = 0.06)

fig.legend(
    legend_bars,
    legend_labels,
    loc = "lower center",
    ncol = 3,
    framealpha = 0.9,
    fontsize = 13,
)

plt.savefig("Assessments/Extrinsic/results/figure.pdf")