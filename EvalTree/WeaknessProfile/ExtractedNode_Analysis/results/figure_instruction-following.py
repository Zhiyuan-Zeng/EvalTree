import json
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


InstanceTypes = {
    "lower" : "Weakness",
    "higher" : "Strength",
}

id_data = {
    "lower" : None,
    "higher" : None,
}
ood_data = {
    "lower" : [],
    "higher" : [],
}
for direction in ("lower", "higher") :
    with open("EvalTree/WeaknessProfile/ExtractedNode_Analysis/results/WildChat10K->WildChat10K/[direction={}][llama3.2-3b-instruct]BEAT[gemma2-2b-it].json".format(direction), "r") as fin :
        id_data[direction] = json.load(fin)
    for ood_dataset in ("ShareGPT10K", "Chatbot-Arena", ) :
        with open("EvalTree/WeaknessProfile/ExtractedNode_Analysis/results/WildChat10K->{}/[direction={}][llama3.2-3b-instruct]BEAT[gemma2-2b-it].json".format(ood_dataset, direction), "r") as fin :
            ood_data[direction].append(json.load(fin))


plt.rcParams["font.family"] = "Palatino"
mpl.rcParams["text.usetex"] = True
mpl.rcParams["mathtext.default"] = "regular"
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
fig, axs = plt.subplots(2, 3, figsize = (6 * 3, 8))

column_titles = [
    "WildChat10K (ID)",
    "ShareGPT10K (OOD)",
    "Chatbot Arena (OOD)",
]

for col in range(3) :
    for row, direction in enumerate(("lower", "higher")) :
        ax1 = axs[row, col]

        if col == 0 :
            current_data = id_data[direction]
        else :
            current_data = ood_data[direction][col - 1]

        THRESHOLD = [value * 100.0 for value in current_data["THRESHOLD"]]
        PERFORMANCE = [value * 100.0 for value in current_data["PERFORMANCE"]]
        all_mean = current_data["all_mean"] * 100.0
        INSTANCE_NUMBER = current_data["INSTANCE_NUMBER"]

        ax1.plot(
            THRESHOLD,
            PERFORMANCE,
            label = "{} on ".format("Win-Rate") + r"$\bf{" + "{}".format(InstanceTypes[direction]) + "}$" + " Instances",
            color = "#4575B4",
            linestyle = "-",
            linewidth = 3,
        )
        ax1.axhline(
            y = all_mean,
            color = "#C86862",
            linestyle = "--",
            label = "{} on All Instances ({:.2f}\\%)".format("Win-Rate", all_mean),
            linewidth = 3,
        )

        ax1.set_xlabel("Threshold $\\tau$ (\\%)")
        ax1.set_ylabel("{} (\\%)".format("Win-Rate"), color = "#4575B4")
        ax1.tick_params(axis = "y", labelcolor = "#4575B4")
        ax1.grid(True, linestyle = "--", alpha = 0.6)

        if col == 0 :
            ax1.plot(THRESHOLD, THRESHOLD, label = "{} Threshold".format("Win-Rate"), color = "#5A5A5A", linestyle = ":", linewidth = 3)

        ax2 = ax1.twinx()
        ax2.plot(
            THRESHOLD,
            INSTANCE_NUMBER,
            label = "Number of " + r"$\bf{" + "{}".format(InstanceTypes[direction]) + "}$" + " Instances",
            color = "#D9A441",
            linestyle = ":",
            linewidth = 1.5,
        )
        ax2.set_ylabel("Instance Number", color = "#D9A441")
        ax2.tick_params(axis = "y", labelcolor = "#D9A441")

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(
            lines_1 + lines_2,
            labels_1 + labels_2,
            loc = "best",
            framealpha = 0.9,
        )

        if row == 0 :
            ax1.set_title(r"\textbf{" + column_titles[col] + r"}", fontsize = 20, pad = 10)

plt.tight_layout()

line = Line2D(
    [0.325, 0.325],  # x-coordinates in figure space
    [0.03, 0.97],  # y-coordinates in figure space
    color = "black",
    linestyle = ":",
    linewidth = 2,
    transform = fig.transFigure,
    clip_on = False,
)
fig.add_artist(line)

fig.text(0.16, 0.015, r"\textbf{(a)}", fontsize = 16, ha = "center", va = "center")
fig.text(0.66, 0.015, r"\textbf{(b)}", fontsize = 16, ha = "center", va = "center")
plt.subplots_adjust(bottom = 0.10)  # Adjust the bottom margin as needed

plt.savefig("EvalTree/WeaknessProfile/ExtractedNode_Analysis/results/WildChat10K.pdf")