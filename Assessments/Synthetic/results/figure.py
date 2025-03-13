import os
import json
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

parser = argparse.ArgumentParser()
parser.add_argument("--metrics", type = str, default = "F1", choices = ("F1", "Precision", "Recall", ))
args = parser.parse_args()

prob_drates = (0.2, 0.4, 0.5, )

all_data = {
    "MATH" : [],
    "WildChat10K" : [],
}
for prob_drate in prob_drates :
    for dataset in ("MATH", "WildChat10K", ) :
        prefix = "Assessments/Synthetic/results/{}/[base=0.7]_[drate={}]_[seed=0]".format(dataset, prob_drate)
        METHOD2PATH = {
            r"\textsc{TextDiff}" : "[method=TextDiff][size={}].json",
            r"\textsc{QualEval}" : "[method=QualEval][size={}].json",
            r"\textsc{EvalTree}" : "[method=EvalTree][size={}].json",
        }

        METHOD2DATA = {method : dict(number = [], performance = []) for method in METHOD2PATH.keys()}
        for number in range(1, 20 + 1) :
            for method, path in METHOD2PATH.items() :
                try :
                    with open(os.path.join(prefix, path.format(number)), "r") as fin :
                        results = json.load(fin)
                    METHOD2DATA[method]["number"].append(number)
                    def extract(results, metrics) :
                        if metrics == "F1" :
                            return results["harmonic mean (F1)"]
                        elif metrics == "Precision" :
                            return results["Precision"]["average"]
                        elif metrics == "Recall" :
                            return results["Recall"]["average"]
                        else :
                            raise NotImplementedError("metrics = {}".format(metrics))
                    METHOD2DATA[method]["performance"].append(extract(results, args.metrics))
                except FileNotFoundError :
                    pass
                except :
                    assert False
        
        all_data[dataset].append(METHOD2DATA)


plt.rcParams["font.family"] = "Palatino"
mpl.rcParams["text.usetex"] = True
mpl.rcParams["mathtext.default"] = "regular"
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
assert len(prob_drates) == 3
fig, axs = plt.subplots(2, len(prob_drates), figsize = (6 * len(prob_drates), 8))

for col, drate in enumerate(prob_drates) :
    for row, dataset in enumerate(all_data.keys()) :
        ax1 = axs[row, col]

        markers, colors = ["o", "s", "*"], ["#C86862", "#4575B4", "#D9A441"]
        for idx, (method, data) in enumerate(all_data[dataset][col].items()):
            ax1.plot(data["number"], data["performance"], marker = markers[idx], color = colors[idx], linestyle = "-", label = method, linewidth = 2.5, markersize = 8)
            if args.metrics == "F1" :
                ax1.axhline(y = max(data["performance"]), color = colors[idx], linestyle = '--')
        
        ax1.set_xticks(np.arange(1, 20 + 1))
        ax1.grid(True, linestyle = "--", alpha = 0.6)
        ax1.set_xlabel(r"Size of Weakness Profile  ($M'$)", fontsize = 20)
        if col == 0 :
            ax1.set_ylabel(args.metrics, fontsize = 20)

        ax1.yaxis.set_major_locator(mticker.MaxNLocator(nbins = 6))
        ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))


        if row == 0 :
            ax1.set_title("$d = {}$".format(drate), fontsize = 24, pad = 10)

plt.tight_layout()
fig.subplots_adjust(left = 0.06, bottom = 0.18, hspace = 0.35)

handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc = "lower center", ncol = len(handles), framealpha = 0.9, fontsize = 24)

y_positions = [0.78, 0.37]
for label, y_pos in zip(all_data.keys(), y_positions):
    fig.text(
        0.004,
        y_pos,
        r"\textbf{" + "{}".format(label) + r"}",
        va = "center",
        rotation = "vertical",
        fontsize = 22,
    )

plt.savefig("Assessments/Synthetic/results/{}.pdf".format(args.metrics))