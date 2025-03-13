import os
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker


method_names_list = [r"\textsc{TextDiff}", r"\textsc{QualEval}", r"\textsc{EvalTree}"]

method_paths_row0_MATH = [
    "[split=[exclusion]4k-1k][method=TextDiff]size2val1.json",
    "[split=[exclusion]4k-1k][method=QualEval]size2val1.json",
    "[split=[exclusion]4k-1k][method=EvalTree]size2val1.json",
]
method_paths_row0_WILD = [
    "[split=[exclusion]8k-2k][method=TextDiff]size2val1.json",
    "[split=[exclusion]8k-2k][method=QualEval]size2val1.json",
    "[split=[exclusion]8k-2k][method=EvalTree]size2val1.json",
]

method_paths_row1_MATH = [
    "[split=[exclusion]4k-1k][method=TextDiff]num2val2.json",
    "[split=[exclusion]4k-1k][method=QualEval]num2val2.json",
    "[split=[exclusion]4k-1k][method=EvalTree]num2val2.json",
]
method_paths_row1_WILD = [
    "[split=[exclusion]8k-2k][method=TextDiff]num2val2.json",
    "[split=[exclusion]8k-2k][method=QualEval]num2val2.json",
    "[split=[exclusion]8k-2k][method=EvalTree]num2val2.json",
]


all_models = [
    ("Llama-3.1-8B-Instruct", True),
    ("dart-math-llama3-8b-uniform", True),
    ("[llama3.2-3b-instruct]BEAT[gemma2-2b-it]", False),
]


def get_paths_for(col, row) :
    is_math = all_models[col][1]
    if row == 0 :
        return method_paths_row0_MATH if is_math else method_paths_row0_WILD
    else :
        return method_paths_row1_MATH if is_math else method_paths_row1_WILD


Averages = []
math_models = ["Llama-3.1-8B-Instruct", "dart-math-llama3-8b-uniform"]
for model in math_models :
    prefix = "Datasets/MATH/eval_results/real/{}".format(model)
    with open("Datasets/MATH/eval_results/real/{}/results.json".format(model)) as fin :
        RESULTS = np.array(json.load(fin))
    with open("Datasets/MATH/splits/4k-1k.json".format(model)) as fin :
        split = sorted(list(set(range(len(RESULTS))) - set(json.load(fin))))
    Averages.append(np.mean(RESULTS[split]))
prefix = "Datasets/WildChat10K/eval_results/real/[llama3.2-3b-instruct]BEAT[gemma2-2b-it]"
with open("Datasets/WildChat10K/eval_results/real/[llama3.2-3b-instruct]BEAT[gemma2-2b-it]/results.json".format(model)) as fin :
    RESULTS = np.array([(int(result[0] == 1) + int(result[1] == 1)) / 2.0 for result in json.load(fin)])
with open("Datasets/WildChat10K/splits/8k-2k.json".format(model)) as fin :
    split = sorted(list(set(range(len(RESULTS))) - set(json.load(fin))))
Averages.append(np.mean(RESULTS[split]))


titles = [r"\textbf{Llama 3.1 8B Instruct}", r"\textbf{DART-Math-Llama3-8B (Uniform)}", r"\textbf{Llama 3.2 3B Instruct}"]
ylabels = ["Accuracy", "Accuracy", "Win-Rate"]

plt.rcParams["font.family"] = "Palatino"
mpl.rcParams["text.usetex"] = True
mpl.rcParams["mathtext.default"] = "regular"
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
fig, axs = plt.subplots(2, 3, figsize = (6 * 3, 8))

legend_lines = []
legend_labels = []

colors = ["#C86862", "#4575B4", "#D9A441"]
markers = ["o", "s", "*"]

for col in range(3) :
    for row in range(2) :
        ax = axs[row, col]

        method_paths_list = get_paths_for(col, row)

        if all_models[col][1] :
            prefix = "Assessments/LowPerformance/results/MATH/real/{}".format(all_models[col][0])
        else :
            prefix = "Assessments/LowPerformance/results/WildChat10K/real/[llama3.2-3b-instruct]BEAT[gemma2-2b-it]"

        for idx, path_template in enumerate(method_paths_list) :
            full_path = os.path.join(prefix, path_template)
            with open(full_path, "r") as fin :
                data = json.load(fin)

            def process(metrics):
                return [min(metrics[i:]) * 100.0 for i in range(len(metrics))]

            ln, = ax.plot(
                data["size" if row == 0 else "num"],
                process(data["val1" if row == 0 else "val2"]),
                color = colors[idx % len(colors)],
                linestyle = "-",
                marker = markers[idx % len(markers)],
                linewidth = 2.5,
                markersize = 6,
            )
            if (col == 0) and (row == 0) :
                legend_lines.append(ln)
                legend_labels.append(method_names_list[idx])

        avg_line = ax.axhline(
            y = Averages[col] * 100.0,
            color = "#5A5A5A",
            linestyle = "--",
            linewidth = 3
        )
        if (col == 0) and (row == 0) :
            legend_lines.append(avg_line)
            legend_labels.append("Accuracy/Win-Rate on All Instances")

        ax.grid(True, linestyle = "--", alpha = 0.6)

        if row == 1 :
            ax.set_xlabel(r"Number of Associated Instances ($N'$)", fontsize = 20)
            if col != 1 :
                ax.set_ylabel("{} (\%)".format(ylabels[col]), fontsize = 20)
        else:
            ax.set_xlabel(r"Size of Weakness Profile ($M'$)", fontsize = 20)
            ax.set_xticks(np.arange(1, 20 + 1))
            if col != 1 :
                ax.set_ylabel("{} (\%)".format(ylabels[col]), fontsize = 20)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer = True, nbins = 6))

        if row == 0 :
            ax.set_title(titles[col], fontsize = 22, pad = 10)

line = Line2D(
    [0.6665, 0.6665],  # x in figure coords
    [0.11, 0.98],    # y in figure coords
    color = "black",
    linestyle = ":",
    linewidth = 2,
    transform = fig.transFigure,
    clip_on = False
)
fig.add_artist(line)

fig.text(1 / 3, 0.12, r"\textbf{(a) MATH}", fontsize = 24, ha = "center", va = "center")
fig.text(1 - 1 / 3 / 2, 0.12, r"\textbf{(b) WildChat10K}", fontsize = 24, ha = "center", va = "center")

plt.tight_layout()
fig.subplots_adjust(bottom = 0.23, hspace = 0.35)

fig.legend(
    legend_lines,
    legend_labels,
    loc = "lower center",
    ncol = 4,
    framealpha = 0.9,
    fontsize = 24
)

plt.savefig("Assessments/LowPerformance/results/figure.pdf")