import json
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--predictor_dataset", type = str, required = True, choices = ("MATH", "MMLU", "DS-1000", ))
parser.add_argument("--target_dataset", type = str, required = True, choices = ("MATH", "MMLU", "DS-1000", ) + ("CollegeMath", ))
args = parser.parse_args()


InstanceTypes = {
    "lower" : "Weakness",
    "higher" : "Strength",
}

datasets2models = {
    ("MATH", "MATH") : ("GPT-4o mini", "Llama 3.1 8B Instruct", "DART-Math-Llama3-8B (Uniform)", ),
    ("MMLU", "MMLU") : ("GPT-4o mini", "Llama 3.1 8B Instruct", "TÜLU 3 8B", ),
    ("DS-1000", "DS-1000") : ("GPT-4o", "GPT-3.5 Turbo", "DeepSeek-Coder-Base 6.7B", ),
    ("MATH", "CollegeMath") : ("GPT-4o mini", "Llama 3.1 8B Instruct", "DART-Math-Llama3-8B (Uniform)", ),
}
model2path = {
    "GPT-4o mini" : "gpt-4o-mini-2024-07-18",
    "Llama 3.1 8B Instruct" : "Llama-3.1-8B-Instruct",
    "DART-Math-Llama3-8B (Uniform)" : "dart-math-llama3-8b-uniform",
    "TÜLU 3 8B" : "Llama-3.1-Tulu-3-8B",
    "GPT-4o" : "gpt-4o-2024-08-06",
    "GPT-3.5 Turbo" : "gpt-3.5-turbo-0613",
    "DeepSeek-Coder-Base 6.7B" : "deepseek-coder-6.7b-base",
}


data = {
    "lower" : [],
    "higher" : [],
}
for model in datasets2models[(args.predictor_dataset, args.target_dataset)] :
    for direction in ("lower", "higher") :
        with open("EvalTree/WeaknessProfile/ExtractedNode_Analysis/results/{}->{}/[direction={}]{}.json".format(args.predictor_dataset, args.target_dataset, direction, model2path[model]), "r") as fin :
            data[direction].append(json.load(fin))


plt.rcParams["font.family"] = "Palatino"
mpl.rcParams["text.usetex"] = True
mpl.rcParams["mathtext.default"] = "regular"
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
fig, axs = plt.subplots(2, len(datasets2models[(args.predictor_dataset, args.target_dataset)]), figsize = (6 * len(datasets2models[(args.predictor_dataset, args.target_dataset)]), 8))

for col, model in enumerate(datasets2models[(args.predictor_dataset, args.target_dataset)]) :
    for row, direction in enumerate(("lower", "higher")) :
        ax1 = axs[row, col]

        current_data = data[direction][col]
        THRESHOLD = [value * 100.0 for value in current_data["THRESHOLD"]]
        PERFORMANCE = [value * 100.0 for value in current_data["PERFORMANCE"]]
        all_mean = current_data["all_mean"] * 100.0
        INSTANCE_NUMBER = current_data["INSTANCE_NUMBER"]

        ax1.plot(THRESHOLD, PERFORMANCE, label = "{} on ".format("Accuracy") + r"$\bf{" + "{}".format(InstanceTypes[direction]) + "}$" + " Instances", color = "#4575B4", linestyle = "-", linewidth = 3)
        ax1.axhline(y = all_mean, color = "#C86862", linestyle = "--", label = "{} on All Instances ({:.2f}\\%)".format("Accuracy", all_mean), linewidth = 3)
        if args.predictor_dataset == args.target_dataset :
            ax1.plot(THRESHOLD, THRESHOLD, label = "{} Threshold".format("Accuracy"), color = "#5A5A5A", linestyle = ":", linewidth = 3)

        ax1.set_xlabel("Threshold $\\tau$ (\\%)")
        ax1.set_ylabel("{} (\\%)".format("Accuracy"), color = "#4575B4")
        ax1.tick_params(axis = "y", labelcolor = "#4575B4")
        ax1.grid(True, linestyle = "--", alpha = 0.6)

        ax2 = ax1.twinx()
        ax2.plot(THRESHOLD, INSTANCE_NUMBER, label = "Number of " + r"$\bf{" + "{}".format(InstanceTypes[direction]) + "}$" + " Instances", color = "#D9A441", linestyle = ":", linewidth = 1.5)
        ax2.set_ylabel("Instance Number", color = "#D9A441")
        ax2.tick_params(axis = "y", labelcolor = "#D9A441")

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc = "best", framealpha = 0.9)

        if row == 0 :
            ax1.set_title(r"\textbf{" + model + r"}", fontsize = 20, pad = 10)


plt.tight_layout()

plt.savefig("EvalTree/WeaknessProfile/ExtractedNode_Analysis/results/{}_{}.pdf".format(args.predictor_dataset, args.target_dataset))